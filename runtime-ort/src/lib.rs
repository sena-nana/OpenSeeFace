//! ort inference for the same ONNX files used by the Python tracker.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use half::f16;
#[cfg(feature = "gpu")]
use ort::ep::ExecutionProvider;
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::{Tensor, TensorRef, ValueType};
use ort::{api, AsPointer};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const RETINA_MEAN: [f32; 3] = [104.0, 117.0, 123.0];

const fn bake_lut(scale: [f32; 3], bias: [f32; 3]) -> [[f16; 256]; 3] {
    let mut lut = [[f16::from_f32_const(0.0); 256]; 3];
    let mut c = 0;
    while c < 3 {
        let mut v = 0;
        while v < 256 {
            lut[c][v] = f16::from_f32_const((v as f32) * scale[c] + bias[c]);
            v += 1;
        }
        c += 1;
    }
    lut
}

/// Baked BGR-source → NCHW dest normalization.
///
/// ImageNet dest planes are RGB (`src = [2,1,0]`) with `(x/255 - mean) / std`.
/// RetinaFace dest planes stay BGR and subtract `(104, 117, 123)`.
#[derive(Clone, Copy, Debug)]
pub struct ColorNorm {
    pub scale: [f32; 3],
    pub bias: [f32; 3],
    /// BGR channel index for destination planes 0..2.
    pub src: [usize; 3],
    lut: [[f16; 256]; 3],
}

impl ColorNorm {
    pub const IMAGENET: Self = {
        let scale = [
            1.0 / (IMAGENET_STD[0] * 255.0),
            1.0 / (IMAGENET_STD[1] * 255.0),
            1.0 / (IMAGENET_STD[2] * 255.0),
        ];
        let bias = [
            -(IMAGENET_MEAN[0] / IMAGENET_STD[0]),
            -(IMAGENET_MEAN[1] / IMAGENET_STD[1]),
            -(IMAGENET_MEAN[2] / IMAGENET_STD[2]),
        ];
        Self {
            scale,
            bias,
            src: [2, 1, 0],
            lut: bake_lut(scale, bias),
        }
    };

    pub const RETINA: Self = {
        let scale = [1.0, 1.0, 1.0];
        let bias = [-RETINA_MEAN[0], -RETINA_MEAN[1], -RETINA_MEAN[2]];
        Self {
            scale,
            bias,
            src: [0, 1, 2],
            lut: bake_lut(scale, bias),
        }
    };
}

/// Matches `Tracker.model_type`.
#[derive(Clone, Copy, Debug)]
pub struct LmSpec {
    pub model_type: i32,
    pub file: &'static str,
    pub size: u32,
    pub out_res: i32,
    pub points: usize,
    pub logit: f32,
}

impl LmSpec {
    pub fn from_type(t: i32) -> Result<Self> {
        Ok(match t {
            -3 => Self {
                model_type: t,
                file: "lm_modelU_opt.onnx",
                size: 112,
                out_res: 13,
                points: 66,
                logit: 16.0,
            },
            -2 => Self {
                model_type: t,
                file: "lm_modelV_opt.onnx",
                size: 112,
                out_res: 13,
                points: 66,
                logit: 16.0,
            },
            -1 => Self {
                model_type: t,
                file: "lm_modelT_opt.onnx",
                size: 56,
                out_res: 6,
                points: 30,
                logit: 8.0,
            },
            0 => Self {
                model_type: t,
                file: "lm_model0_opt.onnx",
                size: 224,
                out_res: 27,
                points: 66,
                logit: 16.0,
            },
            1 => Self {
                model_type: t,
                file: "lm_model1_opt.onnx",
                size: 224,
                out_res: 27,
                points: 66,
                logit: 16.0,
            },
            2 => Self {
                model_type: t,
                file: "lm_model2_opt.onnx",
                size: 224,
                out_res: 27,
                points: 66,
                logit: 16.0,
            },
            3 => Self {
                model_type: t,
                file: "lm_model3_opt.onnx",
                size: 224,
                out_res: 27,
                points: 66,
                logit: 16.0,
            },
            4 => Self {
                model_type: t,
                file: "lm_model4_opt.onnx",
                size: 224,
                out_res: 27,
                points: 66,
                logit: 16.0,
            },
            _ => bail!("invalid model type {t}"),
        })
    }

    pub fn grid(self) -> i32 {
        self.out_res + 1
    }
}

#[derive(Clone, Debug)]
pub struct TensorF16 {
    pub shape: Vec<i64>,
    pub data: Vec<f16>,
}

impl TensorF16 {
    pub fn zeros(shape: Vec<i64>) -> Self {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        Self {
            shape,
            data: vec![f16::ZERO; n],
        }
    }

    pub fn from_f32(shape: Vec<i64>, data: impl IntoIterator<Item = f32>) -> Self {
        Self {
            shape,
            data: data.into_iter().map(f16::from_f32).collect(),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|x| x.to_f32()).collect()
    }
}

#[derive(Clone)]
pub struct BgrImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl BgrImage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let img = image::open(path).or_else(|_| image::load_from_memory(&std::fs::read(path)?))?;
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for p in rgb.pixels() {
            data.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn resize(&self, dw: u32, dh: u32) -> Self {
        if dw == self.width && dh == self.height {
            return self.clone();
        }
        let fx = self.width as f32 / dw as f32;
        let fy = self.height as f32 / dh as f32;
        let mut data = vec![0u8; dw as usize * dh as usize * 3];
        for y in 0..dh {
            let sy = (y as f32 + 0.5) * fy - 0.5;
            let y0 = sy.floor().clamp(0.0, (self.height - 1) as f32) as u32;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = (sy - y0 as f32).clamp(0.0, 1.0);
            for x in 0..dw {
                let sx = (x as f32 + 0.5) * fx - 0.5;
                let x0 = sx.floor().clamp(0.0, (self.width - 1) as f32) as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = (sx - x0 as f32).clamp(0.0, 1.0);
                let i00 = ((y0 * self.width + x0) * 3) as usize;
                let i10 = ((y0 * self.width + x1) * 3) as usize;
                let i01 = ((y1 * self.width + x0) * 3) as usize;
                let i11 = ((y1 * self.width + x1) * 3) as usize;
                let o = ((y * dw + x) * 3) as usize;
                for c in 0..3 {
                    let v = self.data[i00 + c] as f32 * (1.0 - wx) * (1.0 - wy)
                        + self.data[i10 + c] as f32 * wx * (1.0 - wy)
                        + self.data[i01 + c] as f32 * (1.0 - wx) * wy
                        + self.data[i11 + c] as f32 * wx * wy;
                    data[o + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Self {
            width: dw,
            height: dh,
            data,
        }
    }
}

fn apply_lut(src_bgr: &[u8], n: usize, norm: &ColorNorm, data: &mut [f16]) {
    let src = norm.src;
    let lut = &norm.lut;
    for i in 0..n {
        let p = i * 3;
        data[i] = lut[0][src_bgr[p + src[0]] as usize];
        data[n + i] = lut[1][src_bgr[p + src[1]] as usize];
        data[2 * n + i] = lut[2][src_bgr[p + src[2]] as usize];
    }
}

/// Fused bilinear resize + BGR/RGB remap + baked mean/std → f16 NCHW.
pub fn nchw(bgr: &BgrImage, width: u32, height: u32, norm: &ColorNorm) -> TensorF16 {
    let n = (width * height) as usize;
    let mut data = vec![f16::ZERO; 3 * n];
    let src = norm.src;
    let lut = &norm.lut;
    if width == bgr.width && height == bgr.height {
        apply_lut(&bgr.data, n, norm, &mut data);
        return TensorF16 {
            shape: vec![1, 3, height as i64, width as i64],
            data,
        };
    }
    let fx = bgr.width as f32 / width as f32;
    let fy = bgr.height as f32 / height as f32;
    for y in 0..height {
        let sy = (y as f32 + 0.5) * fy - 0.5;
        let y0 = sy.floor().clamp(0.0, (bgr.height - 1) as f32) as u32;
        let y1 = (y0 + 1).min(bgr.height - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..width {
            let sx = (x as f32 + 0.5) * fx - 0.5;
            let x0 = sx.floor().clamp(0.0, (bgr.width - 1) as f32) as u32;
            let x1 = (x0 + 1).min(bgr.width - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i00 = ((y0 * bgr.width + x0) * 3) as usize;
            let i10 = ((y0 * bgr.width + x1) * 3) as usize;
            let i01 = ((y1 * bgr.width + x0) * 3) as usize;
            let i11 = ((y1 * bgr.width + x1) * 3) as usize;
            let o = (y * width + x) as usize;
            for c in 0..3 {
                let s = src[c];
                let v = bgr.data[i00 + s] as f32 * (1.0 - wx) * (1.0 - wy)
                    + bgr.data[i10 + s] as f32 * wx * (1.0 - wy)
                    + bgr.data[i01 + s] as f32 * (1.0 - wx) * wy
                    + bgr.data[i11 + s] as f32 * wx * wy;
                let u = v.round().clamp(0.0, 255.0) as u8 as usize;
                data[c * n + o] = lut[c][u];
            }
        }
    }
    TensorF16 {
        shape: vec![1, 3, height as i64, width as i64],
        data,
    }
}

pub fn imagenet_nchw(bgr: &BgrImage, size: u32) -> TensorF16 {
    nchw(bgr, size, size, &ColorNorm::IMAGENET)
}

pub fn retina_nchw(bgr: &BgrImage) -> TensorF16 {
    nchw(bgr, 640, 640, &ColorNorm::RETINA)
}

/// `cpu` or `gpu` (CoreML on Apple, CUDA on NVIDIA).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => bail!("unknown device {s:?}, expected cpu|gpu"),
        }
    }
}

fn oe(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

fn gpu_eps() -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    #[cfg(not(feature = "gpu"))]
    bail!("GPU requested; rebuild with `--features gpu`");

    #[cfg(feature = "gpu")]
    {
        #[allow(unused_mut)]
        let mut eps = Vec::new();
        #[cfg(target_os = "macos")]
        if ort::ep::CoreML::default().is_available().unwrap_or(false) {
            eps.push(
                ort::ep::CoreML::default()
                    .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .build()
                    .error_on_failure(),
            );
        }
        #[cfg(not(target_os = "macos"))]
        if ort::ep::CUDA::default().is_available().unwrap_or(false) {
            eps.push(ort::ep::CUDA::default().build().error_on_failure());
        }
        if eps.is_empty() {
            bail!("GPU requested but no GPU execution provider is available");
        }
        Ok(eps)
    }
}

struct BoundIo {
    binding: IoBinding,
    input: Tensor<f16>,
    input_shape: Vec<i64>,
}

pub struct OrtModel {
    session: Session,
    bound: Option<BoundIo>,
    pub input_name: String,
    pub load_ms: f64,
}

impl OrtModel {
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        Self::open(path, threads, Device::Cpu, 1)
    }

    pub fn open(
        path: impl AsRef<Path>,
        threads: usize,
        device: Device,
        batch: i64,
    ) -> Result<Self> {
        let path = path.as_ref();
        let start = Instant::now();
        let mut builder = Session::builder()
            .map_err(oe)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(oe)?
            .with_intra_threads(threads.max(1))
            .map_err(oe)?
            .with_inter_threads(1)
            .map_err(oe)?
            .with_parallel_execution(false)
            .map_err(oe)?
            .with_memory_pattern(true)
            .map_err(oe)?
            .with_log_level(LogLevel::Error)
            .map_err(oe)?;
        if device == Device::Gpu {
            builder = builder
                .with_dimension_override("batch_size", batch.max(1))
                .map_err(oe)?
                .with_execution_providers(gpu_eps()?)
                .map_err(oe)?;
        }
        let session = builder.commit_from_file(path).map_err(oe)?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".into());
        Ok(Self {
            session,
            bound: None,
            input_name,
            load_ms,
        })
    }

    fn bind(&mut self, input: &TensorF16) -> Result<()> {
        if self.bound.as_ref().is_some_and(|b| b.input_shape == input.shape) {
            return self.write_input(input);
        }
        let batch = input.shape.first().copied().unwrap_or(1);
        let mut specs = Vec::new();
        for o in self.session.outputs() {
            let ValueType::Tensor { shape, .. } = o.dtype() else {
                bail!("expected tensor output {}", o.name());
            };
            let mut dims: Vec<i64> = shape.iter().copied().collect();
            if dims.first().is_some_and(|d| *d <= 0) {
                dims[0] = batch;
            }
            specs.push((o.name().to_string(), dims));
        }
        if specs.iter().any(|(_, s)| s.iter().any(|d| *d <= 0)) {
            let probe = self.run_unbound(input)?;
            specs = self
                .session
                .outputs()
                .iter()
                .zip(&probe)
                .map(|(o, t)| (o.name().to_string(), t.shape.clone()))
                .collect();
        }
        let input_t = Tensor::from_array((input.shape.clone(), input.data.clone())).map_err(oe)?;
        let mut binding = self.session.create_binding().map_err(oe)?;
        binding
            .bind_input(self.input_name.clone(), &input_t)
            .map_err(oe)?;
        for (name, shape) in specs {
            let n: usize = shape.iter().map(|d| *d as usize).product();
            binding
                .bind_output(name, Tensor::from_array((shape, vec![f16::ZERO; n])).map_err(oe)?)
                .map_err(oe)?;
        }
        self.bound = Some(BoundIo {
            binding,
            input: input_t,
            input_shape: input.shape.clone(),
        });
        Ok(())
    }

    fn write_input(&mut self, input: &TensorF16) -> Result<()> {
        let name = self.input_name.clone();
        let bound = self.bound.as_mut().context("unbound")?;
        {
            let (_, buf) = bound.input.try_extract_tensor_mut::<f16>().map_err(oe)?;
            buf.copy_from_slice(&input.data);
        }
        bound.binding.bind_input(name, &bound.input).map_err(oe)
    }

    fn run_unbound(&mut self, input: &TensorF16) -> Result<Vec<TensorF16>> {
        let tensor = TensorRef::from_array_view((input.shape.as_slice(), input.data.as_slice()))
            .map_err(oe)?;
        collect_f16(
            &self
                .session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(oe)?,
        )
    }

    pub fn infer(&mut self) -> Result<()> {
        let OrtModel { session, bound, .. } = self;
        let bound = bound.as_ref().context("infer() needs run() first")?;
        let status = unsafe {
            (api().RunWithBinding)(session.ptr().cast_mut(), core::ptr::null(), bound.binding.ptr())
        };
        unsafe { ort::Error::result_from_status(status) }.map_err(oe)
    }

    pub fn run(&mut self, input: &TensorF16) -> Result<Vec<TensorF16>> {
        self.bind(input)?;
        let OrtModel { session, bound, .. } = self;
        collect_f16(
            &session
                .run_binding(&bound.as_ref().unwrap().binding)
                .map_err(oe)?,
        )
    }
}

fn collect_f16(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<TensorF16>> {
    let mut out = Vec::with_capacity(outputs.len());
    for i in 0..outputs.len() {
        let (shape, data) = outputs[i]
            .try_extract_tensor::<f16>()
            .map_err(oe)?;
        out.push(TensorF16 {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        });
    }
    if out.is_empty() {
        bail!("empty model output");
    }
    Ok(out)
}

pub fn detect_faces(
    outputs: &TensorF16,
    maxpool: &TensorF16,
    fw: u32,
    fh: u32,
    thresh: f32,
) -> Vec<[f32; 5]> {
    let plane = 56 * 56;
    let mut heat: Vec<f32> = outputs.data[..plane].iter().map(|x| x.to_f32()).collect();
    for i in 0..plane {
        if (heat[i] - maxpool.data[i].to_f32()).abs() > 1e-6 {
            heat[i] = 0.0;
        }
    }
    let mut order: Vec<usize> = (0..plane).collect();
    order.sort_by(|a, b| {
        heat[*b]
            .partial_cmp(&heat[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut dets = Vec::new();
    if let Some(&i) = order.first() {
        if heat[i] >= thresh {
            let y = (i / 56) as f32;
            let x = (i % 56) as f32;
            let r = outputs.data[plane + i].to_f32() * 112.0;
            let sx = fw as f32 / 224.0;
            let sy = fh as f32 / 224.0;
            dets.push([
                (x * 4.0 - r) * sx,
                (y * 4.0 - r) * sy,
                2.0 * r * sx,
                2.0 * r * sy,
                heat[i],
            ]);
        }
    }
    dets
}

pub fn decode_landmarks(t: &TensorF16, crop: [f32; 4], spec: LmSpec) -> (f32, Vec<[f32; 3]>) {
    let c0 = spec.points;
    let g = spec.grid() as usize;
    let cells = g * g;
    let data = &t.data;
    let res = spec.size as f32 - 1.0;
    let mut pts = Vec::with_capacity(c0);
    let mut sum = 0.0f32;
    for p in 0..c0 {
        let base = p * cells;
        let (best, conf) = (0..cells)
            .map(|i| (i, data[base + i].to_f32()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        let off_x = logit(data[(c0 + p) * cells + best].to_f32(), spec.logit) * res;
        let off_y = logit(data[(2 * c0 + p) * cells + best].to_f32(), spec.logit) * res;
        let tm = best as f32;
        let x = crop[1] + crop[3] * (res * (tm / g as f32).floor() / spec.out_res as f32 + off_x);
        let y = crop[0] + crop[2] * (res * (tm % g as f32).floor() / spec.out_res as f32 + off_y);
        sum += conf;
        pts.push([x, y, conf]);
    }
    (sum / c0 as f32, pts)
}

fn logit(p: f32, factor: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln() / factor
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

pub fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .take(n)
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / n as f32
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Latency {
    pub warmup: u32,
    pub iters: u32,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl Latency {
    pub fn from_samples(warmup: u32, s: &[f64]) -> Self {
        let mut v = s.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len() as f64;
        let mean = if n == 0.0 {
            0.0
        } else {
            v.iter().sum::<f64>() / n
        };
        Self {
            warmup,
            iters: s.len() as u32,
            mean_ms: mean,
            p50_ms: pct(&v, 50.0),
            p90_ms: pct(&v, 90.0),
            p99_ms: pct(&v, 99.0),
            min_ms: v.first().copied().unwrap_or(0.0),
            max_ms: v.last().copied().unwrap_or(0.0),
        }
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let r = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = r.floor() as usize;
    let hi = r.ceil() as usize;
    let w = r - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Rss {
    pub rss_bytes: u64,
    pub rss_peak_bytes: u64,
}

pub fn rss() -> Rss {
    let peak = unsafe {
        let mut u = std::mem::zeroed::<libc::rusage>();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        #[cfg(target_os = "macos")]
        {
            u.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            (u.ru_maxrss as u64) * 1024
        }
    };
    let cur = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|kb| kb * 1024)
        .unwrap_or(peak);
    Rss {
        rss_bytes: cur,
        rss_peak_bytes: peak,
    }
}

pub fn read_f32_le(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let b = std::fs::read(path.as_ref()).with_context(|| path.as_ref().display().to_string())?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn model_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(b: u8, g: u8, r: u8) -> BgrImage {
        BgrImage {
            width: 2,
            height: 2,
            data: vec![b, g, r].repeat(4),
        }
    }

    #[test]
    fn imagenet_lut_matches_affine() {
        let n = ColorNorm::IMAGENET;
        for v in [0u8, 1, 17, 128, 255] {
            for c in 0..3 {
                let got = n.lut[c][v as usize].to_f32();
                let exp = v as f32 * n.scale[c] + n.bias[c];
                assert!((got - exp).abs() < 2e-3, "c={c} v={v} {got} vs {exp}");
            }
        }
    }

    #[test]
    fn imagenet_swaps_bgr_to_rgb() {
        let red = imagenet_nchw(&solid(0, 0, 255), 2);
        let n = ColorNorm::IMAGENET;
        let exp_r = 255.0 * n.scale[0] + n.bias[0];
        let exp_g = n.bias[1];
        let exp_b = n.bias[2];
        assert!((red.data[0].to_f32() - exp_r).abs() < 2e-3);
        assert!((red.data[4].to_f32() - exp_g).abs() < 2e-3);
        assert!((red.data[8].to_f32() - exp_b).abs() < 2e-3);
    }

    #[test]
    fn retina_keeps_bgr() {
        let blue = retina_nchw(&BgrImage {
            width: 640,
            height: 640,
            data: vec![200u8, 10, 3].repeat(640 * 640),
        });
        assert!((blue.data[0].to_f32() - (200.0 - 104.0)).abs() < 2e-3);
        assert!((blue.data[640 * 640].to_f32() - (10.0 - 117.0)).abs() < 2e-3);
        assert!((blue.data[2 * 640 * 640].to_f32() - (3.0 - 123.0)).abs() < 2e-3);
    }

}
