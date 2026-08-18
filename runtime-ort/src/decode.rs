use anyhow::{bail, Result};
use half::f16;

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
        let ox = logit(data[(c0 + p) * cells + best].to_f32(), spec.logit) * res;
        let oy = logit(data[(2 * c0 + p) * cells + best].to_f32(), spec.logit) * res;
        let tm = best as f32;
        let x = crop[1] + crop[3] * (res * (tm / g as f32).floor() / spec.out_res as f32 + ox);
        let y = crop[0] + crop[2] * (res * (tm % g as f32).floor() / spec.out_res as f32 + oy);
        sum += conf;
        pts.push([x, y, conf]);
    }
    (sum / c0 as f32, pts)
}

/// Image-space xywh from OpenSeeFace landmarks stored as (row, col, conf).
pub fn landmark_bbox(pts: &[[f32; 3]]) -> Option<[f32; 5]> {
    if pts.is_empty() {
        return None;
    }
    let mut min_r = f32::MAX;
    let mut max_r = f32::MIN;
    let mut min_c = f32::MAX;
    let mut max_c = f32::MIN;
    for p in pts {
        min_r = min_r.min(p[0]);
        max_r = max_r.max(p[0]);
        min_c = min_c.min(p[1]);
        max_c = max_c.max(p[1]);
    }
    Some([
        min_c,
        min_r,
        (max_c - min_c).max(1.0),
        (max_r - min_r).max(1.0),
        1.0,
    ])
}

pub const EYE_IDX: [usize; 12] = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47];

pub fn mean_conf(pts: &[[f32; 3]], idx: &[usize]) -> Option<f32> {
    let v: Vec<f32> = idx
        .iter()
        .filter_map(|&i| pts.get(i).map(|p| p[2]))
        .collect();
    (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
}

fn logit(p: f32, factor: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln() / factor
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
