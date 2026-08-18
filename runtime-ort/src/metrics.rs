use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

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

pub fn read_f32_le(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let b = std::fs::read(path.as_ref()).with_context(|| path.as_ref().display().to_string())?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn model_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}
