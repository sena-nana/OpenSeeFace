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

use crate::decode::LmSpec;
#[cfg(feature = "gpu")]
use crate::decode::{decode_landmarks, detect_faces};
use crate::preprocess::BgrImage;
#[cfg(feature = "gpu")]
use crate::preprocess::{crop_box, crop_img};
#[cfg(feature = "gpu")]
use crate::session::Device;
#[cfg(feature = "gpu")]
use crate::session::{collect_f16, make_session, oe};
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
#[cfg(feature = "gpu")]
use ort::value::Tensor;
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
            "failed to generate GPU preprocess graphs in {} (need Python + onnx)",
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

/// GPU pipeline: preprocess on the EP; only small tensors come back to the CPU.
pub struct GpuTracker {
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    pipe: CoreMlPipe,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pipe: CudaPipe,
    #[cfg(not(feature = "gpu"))]
    _no_gpu: (),
}

impl GpuTracker {
    pub fn open(models_dir: &Path, spec: LmSpec, threads: usize, frame: &BgrImage) -> Result<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (models_dir, spec, threads, frame);
            bail!("GPU requested; rebuild with `--features gpu`");
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let pre = ensure_pre_models(models_dir)?;
            Ok(Self {
                pipe: CoreMlPipe::open(&pre, spec, threads, frame)?,
            })
        }
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        {
            let pre = ensure_pre_models(models_dir)?;
            Ok(Self {
                pipe: CudaPipe::open(models_dir, &pre, spec, threads, frame)?,
            })
        }
    }

    pub fn detect(&mut self, frame: &BgrImage) -> Result<Vec<[f32; 5]>> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = frame;
            bail!("GPU requested; rebuild with `--features gpu`");
        }
        #[cfg(feature = "gpu")]
        self.pipe.detect(frame)
    }

    pub fn landmarks(
        &mut self,
        frame: &BgrImage,
        det: &[f32; 5],
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (frame, det, spec);
            bail!("GPU requested; rebuild with `--features gpu`");
        }
        #[cfg(feature = "gpu")]
        self.pipe.landmarks(frame, det, spec)
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
struct CoreMlPipe {
    det: Session,
    det_name: String,
    lm: Session,
    lm_name: String,
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
impl CoreMlPipe {
    fn open(pre: &Path, spec: LmSpec, threads: usize, frame: &BgrImage) -> Result<Self> {
        let fused_det = pre.join("mnv3_detection_opt.onnx");
        let fused_lm = pre.join(spec.file);
        if !fused_det.is_file() || !fused_lm.is_file() {
            bail!("fused CoreML graphs not in {}", pre.display());
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
        })
    }

    fn detect(&mut self, frame: &BgrImage) -> Result<Vec<[f32; 5]>> {
        let shape = [1i64, frame.height as i64, frame.width as i64, 3];
        let t = Tensor::<u8>::from_array((shape.as_slice(), frame.data.clone())).map_err(oe)?;
        let outs = self
            .det
            .run(ort::inputs![self.det_name.as_str() => t])
            .map_err(oe)?;
        let got = collect_f16(&outs)?;
        Ok(detect_faces(
            &got[0],
            &got[1],
            frame.width,
            frame.height,
            0.6,
        ))
    }

    fn landmarks(
        &mut self,
        frame: &BgrImage,
        det: &[f32; 5],
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        let (x1, y1, x2, y2) = crop_box(frame, det);
        let crop = crop_img(frame, x1, y1, x2, y2);
        if crop.width < 4 || crop.height < 4 {
            bail!("crop too small");
        }
        let sized = if crop.width == spec.size && crop.height == spec.size {
            crop
        } else {
            cpu_resize_u8(&crop, spec.size, spec.size)
        };
        let shape = [1i64, sized.height as i64, sized.width as i64, 3];
        let t = Tensor::<u8>::from_array((shape.as_slice(), sized.data)).map_err(oe)?;
        let outs = self
            .lm
            .run(ort::inputs![self.lm_name.as_str() => t])
            .map_err(oe)?;
        let got = collect_f16(&outs)?;
        let scale_x = (x2 - x1) as f32 / spec.size as f32;
        let scale_y = (y2 - y1) as f32 / spec.size as f32;
        Ok(decode_landmarks(
            &got[0],
            [x1 as f32, y1 as f32, scale_x, scale_y],
            spec,
        ))
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn cpu_resize_u8(bgr: &BgrImage, dw: u32, dh: u32) -> BgrImage {
    if dw == bgr.width && dh == bgr.height {
        return bgr.clone();
    }
    let fx = bgr.width as f32 / dw as f32;
    let fy = bgr.height as f32 / dh as f32;
    let mut data = vec![0u8; dw as usize * dh as usize * 3];
    for y in 0..dh {
        let sy = (y as f32 + 0.5) * fy - 0.5;
        let y0 = sy.floor().clamp(0.0, (bgr.height - 1) as f32) as u32;
        let y1 = (y0 + 1).min(bgr.height - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..dw {
            let sx = (x as f32 + 0.5) * fx - 0.5;
            let x0 = sx.floor().clamp(0.0, (bgr.width - 1) as f32) as u32;
            let x1 = (x0 + 1).min(bgr.width - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i00 = ((y0 * bgr.width + x0) * 3) as usize;
            let i10 = ((y0 * bgr.width + x1) * 3) as usize;
            let i01 = ((y1 * bgr.width + x0) * 3) as usize;
            let i11 = ((y1 * bgr.width + x1) * 3) as usize;
            let o = ((y * dw + x) * 3) as usize;
            for c in 0..3 {
                let v = bgr.data[i00 + c] as f32 * (1.0 - wx) * (1.0 - wy)
                    + bgr.data[i10 + c] as f32 * wx * (1.0 - wy)
                    + bgr.data[i01 + c] as f32 * (1.0 - wx) * wy
                    + bgr.data[i11 + c] as f32 * wx * wy;
                data[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    BgrImage {
        width: dw,
        height: dh,
        data,
    }
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
struct CudaPipe {
    w: u32,
    h: u32,
    frame: Tensor<u8>,
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
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
impl CudaPipe {
    fn open(
        models_dir: &Path,
        pre: &Path,
        spec: LmSpec,
        threads: usize,
        frame: &BgrImage,
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
        })
    }

    fn upload_frame(&mut self, frame: &BgrImage) -> Result<()> {
        if frame.width != self.w || frame.height != self.h {
            bail!("frame size changed");
        }
        let (_, buf) = self.frame.extract_tensor_mut();
        buf.copy_from_slice(&frame.data);
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

    fn detect(&mut self, frame: &BgrImage) -> Result<Vec<[f32; 5]>> {
        self.upload_frame(frame)?;
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
        let got = collect_f16(&det_out)?;
        Ok(detect_faces(
            &got[0],
            &got[1],
            frame.width,
            frame.height,
            0.6,
        ))
    }

    fn landmarks(
        &mut self,
        frame: &BgrImage,
        det: &[f32; 5],
        spec: LmSpec,
    ) -> Result<(f32, Vec<[f32; 3]>)> {
        self.upload_frame(frame)?;
        let (x1, y1, x2, y2) = crop_box(frame, det);
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
        let mut pre_bind = self.pre_lm.create_binding().map_err(oe)?;
        let names = self.pre_lm_names.clone();
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
        let got = collect_f16(&lm_out)?;
        let scale_x = (x2 - x1) as f32 / spec.size as f32;
        let scale_y = (y2 - y1) as f32 / spec.size as f32;
        Ok(decode_landmarks(
            &got[0],
            [x1 as f32, y1 as f32, scale_x, scale_y],
            spec,
        ))
    }
}
