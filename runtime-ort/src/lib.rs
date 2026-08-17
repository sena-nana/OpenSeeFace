//! ort inference for the same ONNX files used by the Python tracker.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const RETINA_MEAN: [f32; 3] = [104.0, 117.0, 123.0];

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
pub struct TensorF32 {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
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

fn imagenet_affine() -> ([f32; 3], [f32; 3]) {
    let mut mean = [0.0; 3];
    let mut std = [0.0; 3];
    for c in 0..3 {
        mean[c] = -(IMAGENET_MEAN[c] / IMAGENET_STD[c]);
        std[c] = 1.0 / (IMAGENET_STD[c] * 255.0);
    }
    (mean, std)
}

pub fn imagenet_nchw(bgr: &BgrImage, size: u32) -> TensorF32 {
    let im = bgr.resize(size, size);
    let (mean, std) = imagenet_affine();
    let n = (size * size) as usize;
    let mut data = vec![0.0f32; 3 * n];
    for i in 0..n {
        let b = im.data[i * 3] as f32;
        let g = im.data[i * 3 + 1] as f32;
        let r = im.data[i * 3 + 2] as f32;
        data[i] = r * std[0] + mean[0];
        data[n + i] = g * std[1] + mean[1];
        data[2 * n + i] = b * std[2] + mean[2];
    }
    TensorF32 {
        shape: vec![1, 3, size as i64, size as i64],
        data,
    }
}

pub fn retina_nchw(bgr: &BgrImage) -> TensorF32 {
    let im = bgr.resize(640, 640);
    let n = 640 * 640;
    let mut data = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            data[c * n + i] = im.data[i * 3 + c] as f32 - RETINA_MEAN[c];
        }
    }
    TensorF32 {
        shape: vec![1, 3, 640, 640],
        data,
    }
}

pub struct OrtModel {
    session: Session,
    pub input_name: String,
    pub load_ms: f64,
}

impl OrtModel {
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let path = path.as_ref();
        let start = Instant::now();
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_log_level(LogLevel::Error)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".into());
        Ok(Self {
            session,
            input_name,
            load_ms,
        })
    }

    pub fn run(&mut self, input: &TensorF32) -> Result<Vec<TensorF32>> {
        let tensor = Tensor::from_array((input.shape.clone(), input.data.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let name = self.input_name.clone();
        let outputs = self
            .session
            .run(ort::inputs![name => tensor])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut out = Vec::with_capacity(outputs.len());
        for i in 0..outputs.len() {
            let (shape, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.push(TensorF32 {
                shape: shape.iter().copied().collect(),
                data: data.to_vec(),
            });
        }
        if out.is_empty() {
            bail!("empty model output");
        }
        Ok(out)
    }
}

pub fn detect_faces(
    outputs: &TensorF32,
    maxpool: &TensorF32,
    fw: u32,
    fh: u32,
    thresh: f32,
) -> Vec<[f32; 5]> {
    let plane = 56 * 56;
    let mut heat = outputs.data[..plane].to_vec();
    for i in 0..plane {
        if (heat[i] - maxpool.data[i]).abs() > 1e-6 {
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
            let r = outputs.data[plane + i] * 112.0;
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

pub fn decode_landmarks(t: &TensorF32, crop: [f32; 4], spec: LmSpec) -> (f32, Vec<[f32; 3]>) {
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
            .map(|i| (i, data[base + i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        let off_x = logit(data[(c0 + p) * cells + best], spec.logit) * res;
        let off_y = logit(data[(2 * c0 + p) * cells + best], spec.logit) * res;
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
