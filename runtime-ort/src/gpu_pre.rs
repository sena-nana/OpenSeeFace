//! GPU preprocess: ONNX Resize+Normalize on the EP.
//!
//! CoreML fuses uint8 NHWC → Resize+Normalize+model (ORT has no CoreML device allocator).
//! Landmark crop stays on CPU because CoreML wants static spatial dims.
//! CUDA uploads the uint8 frame once (PINNED), keeps NCHW on device, and only
//! readbacks boxes/heatmaps.

use std::path::Path;
#[cfg(feature = "gpu")]
use std::path::PathBuf;
#[cfg(feature = "gpu")]
use std::process::Command;

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use crate::decode::decode_landmarks_data;
use crate::decode::LmSpec;
use crate::enhance::EnhanceCfg;
#[cfg(feature = "gpu")]
use crate::enhance_gpu::GpuEnhance;
#[cfg(feature = "gpu")]
use crate::preprocess::crop_box_pad;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use crate::preprocess::crop_slice;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use crate::preprocess::resize_bgr;
use crate::preprocess::BgrImage;
#[cfg(feature = "gpu")]
use crate::session::Device;
#[cfg(feature = "gpu")]
use crate::session::{make_session, oe};
#[cfg(feature = "gpu")]
use anyhow::Context;
use anyhow::{bail, Result};
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use half::f16;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ort::memory::Allocator;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ort::session::IoBinding;
#[cfg(feature = "gpu")]
use ort::session::Session;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ort::value::Tensor;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use ort::value::TensorRef;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ort::value::ValueType;

#[cfg(feature = "gpu")]
fn pre_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("pre")
}

#[cfg(feature = "gpu")]
fn ensure_pre_models(models_dir: &Path) -> Result<PathBuf> {
    let out = pre_dir(models_dir);
    if out.join("imagenet_224.onnx").is_file() {
        return Ok(out);
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/wrap_preprocess.py");
    if !script.is_file() {
        bail!("missing {}", script.display());
    }
    let run = |bin: &str, extra: &[&str]| -> Result<bool> {
        let mut cmd = Command::new(bin);
        cmd.args(extra)
            .arg(&script)
            .arg("--models-dir")
            .arg(models_dir)
            .arg("--out-dir")
            .arg(&out);
        Ok(cmd
            .status()
            .with_context(|| format!("spawn {bin}"))?
            .success())
    };
    let ok = run("uv", &["run", "python"]).or_else(|_| run("python3", &[]))?;
    if !ok || !out.join("imagenet_224.onnx").is_file() {
        bail!(
            "failed to generate GPU preprocess graphs in {} (need Python + onnx). Run runtime-ort/scripts/wrap_preprocess.py or use --device cpu",
            out.display()
        );
    }
    Ok(out)
}

#[cfg(feature = "gpu")]
fn in_name(s: &Session) -> String {
    s.inputs()
        .first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "input".into())
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
struct GpuCfg {
    models_dir: PathBuf,
    spec: LmSpec,
    threads: usize,
    enhance: EnhanceCfg,
    threshold: f32,
    max_faces: usize,
    width: u32,
    height: u32,
}

#[cfg(not(feature = "gpu"))]
fn need_gpu<T>() -> Result<T> {
    bail!("GPU requested; rebuild with `--features gpu` or use `--device cpu`")
}

/// GPU pipeline: preprocess on the EP; only small tensors come back to the CPU.
pub struct GpuTracker {
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    pipe: CoreMlPipe,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pipe: CudaPipe,
    #[cfg(feature = "gpu")]
    cfg: GpuCfg,
    #[cfg(not(feature = "gpu"))]
    _no_gpu: (),
}

impl GpuTracker {
    pub fn with_enhance(
        models_dir: &Path,
        spec: LmSpec,
        threads: usize,
        frame: &BgrImage,
        enhance: EnhanceCfg,
    ) -> Result<Self> {
        Self::open(models_dir, spec, threads, frame, enhance, 0.6, 1)
    }

    pub(crate) fn open(
        models_dir: &Path,
        spec: LmSpec,
        threads: usize,
        frame: &BgrImage,
        enhance: EnhanceCfg,
        threshold: f32,
        max_faces: usize,
    ) -> Result<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (
                models_dir, spec, threads, frame, enhance, threshold, max_faces,
            );
            need_gpu()
        }
        #[cfg(feature = "gpu")]
        Self::from_cfg(
            GpuCfg {
                models_dir: models_dir.to_path_buf(),
                spec,
                threads,
                enhance,
                threshold,
                max_faces: max_faces.max(1),
                width: frame.width,
                height: frame.height,
            },
            frame,
        )
    }

    #[cfg(feature = "gpu")]
    fn from_cfg(cfg: GpuCfg, frame: &BgrImage) -> Result<Self> {
        let pre = ensure_pre_models(&cfg.models_dir)?;
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                pipe: CoreMlPipe::open(&pre, cfg.spec, cfg.threads, frame, cfg.enhance)?,
                cfg,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                pipe: CudaPipe::open(
                    &cfg.models_dir,
                    &pre,
                    cfg.spec,
                    cfg.threads,
                    frame,
                    cfg.enhance,
                )?,
                cfg,
            })
        }
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> Result<()> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (width, height);
            need_gpu()
        }
        #[cfg(feature = "gpu")]
        {
            if width == self.cfg.width && height == self.cfg.height {
                return Ok(());
            }
            let mut cfg = self.cfg.clone();
            cfg.width = width;
            cfg.height = height;
            *self = Self::from_cfg(cfg, &BgrImage::zeros(width, height))?;
            Ok(())
        }
    }

    pub fn detect(&mut self, frame: &BgrImage) -> Result<Vec<[f32; 5]>> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = frame;
            need_gpu()
        }
        #[cfg(feature = "gpu")]
        self.pipe
            .detect(frame, self.cfg.threshold, self.cfg.max_faces)
    }

    pub fn landmarks(
        &mut self,
        frame: &BgrImage,
        det: &[f32; 5],
        spec: LmSpec,
        pad_x: f32,
        pad_y: f32,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (frame, det, spec, pad_x, pad_y);
            need_gpu()
        }
        #[cfg(feature = "gpu")]
        {
            let (x1, y1, x2, y2) = crop_box_pad(frame, det, pad_x, pad_y);
            self.pipe.landmarks_roi(frame, x1, y1, x2, y2, spec)
        }
    }

    pub(crate) fn landmarks_roi(
        &mut self,
        frame: &BgrImage,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (frame, x1, y1, x2, y2, spec);
            need_gpu()
        }
        #[cfg(feature = "gpu")]
        self.pipe.landmarks_roi(frame, x1, y1, x2, y2, spec)
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
struct CoreMlPipe {
    det: Session,
    det_name: String,
    lm: Session,
    lm_name: String,
    enhance: GpuEnhance,
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
impl CoreMlPipe {
    fn open(
        pre: &Path,
        spec: LmSpec,
        threads: usize,
        frame: &BgrImage,
        enhance: EnhanceCfg,
    ) -> Result<Self> {
        let fused_det = pre.join("mnv3_detection_opt.onnx");
        let fused_lm = pre.join(spec.file);
        if !fused_det.is_file() || !fused_lm.is_file() {
            bail!(
                "fused CoreML graphs not in {} (run runtime-ort/scripts/wrap_preprocess.py or use --device cpu)",
                pre.display()
            );
        }
        let (det, _) = make_session(
            &fused_det,
            threads,
            Device::Gpu,
            1,
            &[
                ("height", frame.height as i64),
                ("width", frame.width as i64),
            ],
        )?;
        let (lm, _) = make_session(
            &fused_lm,
            threads,
            Device::Gpu,
            1,
            &[("height", spec.size as i64), ("width", spec.size as i64)],
        )?;
        Ok(Self {
            det_name: in_name(&det),
            det,
            lm_name: in_name(&lm),
            lm,
            enhance: GpuEnhance::new(frame.width, frame.height, enhance)?,
        })
    }

    fn detect(
        &mut self,
        frame: &BgrImage,
        threshold: f32,
        max_faces: usize,
    ) -> Result<Vec<[f32; 5]>> {
        let name = self.det_name.clone();
        let shape = [1i64, frame.height as i64, frame.width as i64, 3];
        let bytes = self.enhance.run(frame)?;
        let t = TensorRef::from_array_view((shape.as_slice(), bytes)).map_err(oe)?;
        let outs = self.det.run(ort::inputs![name.as_str() => t]).map_err(oe)?;
        let a = crate::session::output_f16(&outs, 0)?;
        let b = crate::session::output_f16(&outs, 1)?;
        Ok(crate::decode::detect_faces_data(
            a,
            b,
            frame.width,
            frame.height,
            threshold,
            max_faces,
        ))
    }

    fn landmarks_roi(
        &mut self,
        frame: &BgrImage,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        let name = self.lm_name.clone();
        let bytes = self.enhance.run(frame)?;
        let crop = crop_slice(bytes, frame.width, frame.height, x1, y1, x2, y2);
        if crop.width < 4 || crop.height < 4 {
            bail!("crop too small");
        }
        let sized = if crop.width == spec.size && crop.height == spec.size {
            crop
        } else {
            resize_bgr(&crop, spec.size, spec.size)
        };
        let shape = [1i64, sized.height as i64, sized.width as i64, 3];
        let t =
            TensorRef::from_array_view((shape.as_slice(), sized.data.as_slice())).map_err(oe)?;
        let outs = self.lm.run(ort::inputs![name.as_str() => t]).map_err(oe)?;
        let data = crate::session::output_f16(&outs, 0)?;
        let scale_x = (x2 - x1) as f32 / spec.size as f32;
        let scale_y = (y2 - y1) as f32 / spec.size as f32;
        Ok(crate::decode::decode_landmarks_data(
            data,
            [x1 as f32, y1 as f32, scale_x, scale_y],
            spec,
        ))
    }
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
struct CudaPipe {
    w: u32,
    h: u32,
    frame: Tensor<u8>,
    uploaded_ptr: usize,
    pre_det: Session,
    pre_det_name: String,
    det: Session,
    det_in: String,
    pre_lm: Session,
    pre_lm_names: Vec<String>,
    starts: Tensor<i64>,
    ends: Tensor<i64>,
    lm: Session,
    lm_in: String,
    cuda_info: ort::memory::MemoryInfo<'static>,
    pin_out: Allocator,
    enhance: GpuEnhance,
    enhanced: bool,
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
impl CudaPipe {
    fn open(
        models_dir: &Path,
        pre: &Path,
        spec: LmSpec,
        threads: usize,
        frame: &BgrImage,
        enhance: EnhanceCfg,
    ) -> Result<Self> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

        let hw = [
            ("height", frame.height as i64),
            ("width", frame.width as i64),
        ];
        let (pre_det, _) =
            make_session(&pre.join("imagenet_224.onnx"), threads, Device::Gpu, 1, &hw)?;
        let (det, _) = make_session(
            &crate::metrics::model_path(models_dir, "mnv3_detection_opt.onnx"),
            threads,
            Device::Gpu,
            1,
            &[],
        )?;
        let (pre_lm, _) = make_session(
            &pre.join(format!("imagenet_crop_{}.onnx", spec.size)),
            threads,
            Device::Gpu,
            1,
            &hw,
        )?;
        let (lm, _) = make_session(
            &crate::metrics::model_path(models_dir, spec.file),
            threads,
            Device::Gpu,
            1,
            &[],
        )?;
        let pin_in = Allocator::new(
            &pre_det,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                0,
                AllocatorType::Device,
                MemoryType::CPUInput,
            )
            .map_err(oe)?,
        )
        .map_err(oe)?;
        let pin_out = Allocator::new(
            &det,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                0,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )
            .map_err(oe)?,
        )
        .map_err(oe)?;
        let cuda_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(oe)?;
        let mut frame_t =
            Tensor::<u8>::new(&pin_in, [1i64, frame.height as i64, frame.width as i64, 3])
                .map_err(oe)?;
        {
            let (_, buf) = frame_t.extract_tensor_mut();
            buf.copy_from_slice(&frame.data);
        }
        let mut starts = Tensor::<i64>::new(&pin_in, [4i64]).map_err(oe)?;
        let mut ends = Tensor::<i64>::new(&pin_in, [4i64]).map_err(oe)?;
        {
            let (_, b) = starts.extract_tensor_mut();
            b.copy_from_slice(&[0, 0, 0, 0]);
        }
        {
            let (_, b) = ends.extract_tensor_mut();
            b.copy_from_slice(&[1, 1, 1, 3]);
        }
        Ok(Self {
            w: frame.width,
            h: frame.height,
            frame: frame_t,
            uploaded_ptr: frame.data.as_ptr() as usize,
            pre_det_name: in_name(&pre_det),
            pre_det,
            det_in: in_name(&det),
            det,
            pre_lm_names: pre_lm
                .inputs()
                .iter()
                .map(|i| i.name().to_string())
                .collect(),
            pre_lm,
            starts,
            ends,
            lm_in: in_name(&lm),
            lm,
            cuda_info,
            pin_out,
            enhance: GpuEnhance::new(frame.width, frame.height, enhance)?,
            enhanced: false,
        })
    }

    fn upload_frame(&mut self, frame: &BgrImage, force: bool) -> Result<()> {
        if frame.width != self.w || frame.height != self.h {
            bail!("frame size changed");
        }
        let key = frame.data.as_ptr() as usize;
        if !force && key == self.uploaded_ptr {
            return Ok(());
        }
        let (_, buf) = self.frame.extract_tensor_mut();
        buf.copy_from_slice(&frame.data);
        self.uploaded_ptr = key;
        self.enhanced = false;
        Ok(())
    }

    fn ensure_enhanced(&mut self) -> Result<()> {
        if self.enhanced || !self.enhance.enabled() {
            return Ok(());
        }
        let (w, h) = (self.w, self.h);
        let (ptr, len) = {
            let (_, buf) = self.frame.extract_tensor_mut();
            (buf.as_mut_ptr(), buf.len())
        };
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        self.enhance.run_in_place(buf, w, h)?;
        self.enhanced = true;
        Ok(())
    }

    fn bind_f16_outs(sess: &Session, bind: &mut IoBinding, pin: &Allocator) -> Result<()> {
        for o in sess.outputs() {
            let ValueType::Tensor { shape, .. } = o.dtype() else {
                continue;
            };
            let mut dims: Vec<i64> = shape.iter().copied().collect();
            if dims.first().is_some_and(|d| *d <= 0) {
                dims[0] = 1;
            }
            bind.bind_output(
                o.name().to_string(),
                Tensor::<f16>::new(pin, dims).map_err(oe)?,
            )
            .map_err(oe)?;
        }
        Ok(())
    }

    fn detect(
        &mut self,
        frame: &BgrImage,
        threshold: f32,
        max_faces: usize,
    ) -> Result<Vec<[f32; 5]>> {
        self.upload_frame(frame, true)?;
        self.ensure_enhanced()?;
        let mut pre_bind = self.pre_det.create_binding().map_err(oe)?;
        pre_bind
            .bind_input(self.pre_det_name.clone(), &self.frame)
            .map_err(oe)?;
        pre_bind
            .bind_output_to_device("nchw", &self.cuda_info)
            .map_err(oe)?;
        let mut pre_out = self.pre_det.run_binding(&pre_bind).map_err(oe)?;
        let nchw = pre_out.remove("nchw").context("pre nchw")?;

        let mut det_bind = self.det.create_binding().map_err(oe)?;
        det_bind
            .bind_input(self.det_in.clone(), &nchw)
            .map_err(oe)?;
        Self::bind_f16_outs(&self.det, &mut det_bind, &self.pin_out)?;
        let det_out = self.det.run_binding(&det_bind).map_err(oe)?;
        let a = crate::session::output_f16(&det_out, 0)?;
        let b = crate::session::output_f16(&det_out, 1)?;
        Ok(crate::decode::detect_faces_data(
            a,
            b,
            frame.width,
            frame.height,
            threshold,
            max_faces,
        ))
    }

    fn landmarks_roi(
        &mut self,
        frame: &BgrImage,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        self.upload_frame(frame, false)?;
        self.ensure_enhanced()?;
        if x2 - x1 < 4 || y2 - y1 < 4 {
            bail!("crop too small");
        }
        {
            let (_, s) = self.starts.extract_tensor_mut();
            s.copy_from_slice(&[0, y1 as i64, x1 as i64, 0]);
        }
        {
            let (_, e) = self.ends.extract_tensor_mut();
            e.copy_from_slice(&[1, y2 as i64, x2 as i64, 3]);
        }
        let names = self.pre_lm_names.clone();
        let mut pre_bind = self.pre_lm.create_binding().map_err(oe)?;
        for n in &names {
            match n.as_str() {
                "starts" => pre_bind.bind_input(n.clone(), &self.starts).map_err(oe)?,
                "ends" => pre_bind.bind_input(n.clone(), &self.ends).map_err(oe)?,
                _ => pre_bind.bind_input(n.clone(), &self.frame).map_err(oe)?,
            }
        }
        pre_bind
            .bind_output_to_device("nchw", &self.cuda_info)
            .map_err(oe)?;
        let mut pre_out = self.pre_lm.run_binding(&pre_bind).map_err(oe)?;
        let nchw = pre_out.remove("nchw").context("lm nchw")?;

        let mut lm_bind = self.lm.create_binding().map_err(oe)?;
        lm_bind.bind_input(self.lm_in.clone(), &nchw).map_err(oe)?;
        Self::bind_f16_outs(&self.lm, &mut lm_bind, &self.pin_out)?;
        let lm_out = self.lm.run_binding(&lm_bind).map_err(oe)?;
        let data = crate::session::output_f16(&lm_out, 0)?;
        let scale_x = (x2 - x1) as f32 / spec.size as f32;
        let scale_y = (y2 - y1) as f32 / spec.size as f32;
        Ok(decode_landmarks_data(
            data,
            [x1 as f32, y1 as f32, scale_x, scale_y],
            spec,
        ))
    }
}
