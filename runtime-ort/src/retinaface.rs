//! RetinaFace ONNX detector with prior-box decode + NMS.

use std::path::Path;

use anyhow::{Context, Result};

use crate::decode::TensorF16;
use crate::preprocess::{retina_nchw, BgrImage};
use crate::session::OrtModel;

pub struct RetinaFace {
    model: OrtModel,
    priors: Vec<[f32; 4]>,
    min_conf: f32,
    nms_threshold: f32,
    top_k: usize,
    pending: Option<BgrImage>,
    ready: Option<Vec<[f32; 4]>>,
}

fn nms(dets: &[[f32; 5]], thresh: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|a, b| {
        dets[*b][4]
            .partial_cmp(&dets[*a][4])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        let mut rest = Vec::new();
        let (x1, y1, x2, y2) = (dets[i][0], dets[i][1], dets[i][2], dets[i][3]);
        let area_i = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
        for &j in order.iter().skip(1) {
            let xx1 = x1.max(dets[j][0]);
            let yy1 = y1.max(dets[j][1]);
            let xx2 = x2.min(dets[j][2]);
            let yy2 = y2.min(dets[j][3]);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let area_j = (dets[j][2] - dets[j][0] + 1.0) * (dets[j][3] - dets[j][1] + 1.0);
            let ovr = inter / (area_i + area_j - inter);
            if ovr <= thresh {
                rest.push(j);
            }
        }
        order = rest;
    }
    keep
}

fn decode_boxes(loc: &[[f32; 4]], priors: &[[f32; 4]]) -> Vec<[f32; 4]> {
    loc.iter()
        .zip(priors)
        .map(|(l, p)| {
            let cx = p[0] + l[0] * 0.1 * p[2];
            let cy = p[1] + l[1] * 0.1 * p[3];
            let w = p[2] * (l[2] * 0.2).exp();
            let h = p[3] * (l[3] * 0.2).exp();
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
        })
        .collect()
}

fn parse_priors(path: &Path) -> Result<Vec<[f32; 4]>> {
    let s = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let raw: Vec<Vec<f32>> = serde_json::from_str(&s)?;
    Ok(raw
        .into_iter()
        .filter(|r| r.len() >= 4)
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect())
}

fn flatten4(t: &TensorF16) -> Vec<[f32; 4]> {
    let d: Vec<f32> = t.data.iter().map(|x| x.to_f32()).collect();
    d.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

fn flatten_conf(t: &TensorF16) -> Vec<f32> {
    let d: Vec<f32> = t.data.iter().map(|x| x.to_f32()).collect();
    let shape = &t.shape;
    if shape.len() >= 2 && *shape.last().unwrap_or(&1) == 2 {
        d.chunks_exact(2).map(|c| c[1]).collect()
    } else if d.len() >= 2 {
        d.chunks_exact(d.len() / (d.len() / 2).max(1))
            .map(|c| *c.last().unwrap_or(&0.0))
            .collect()
    } else {
        d
    }
}

impl RetinaFace {
    pub fn load(
        model_path: impl AsRef<Path>,
        prior_path: impl AsRef<Path>,
        threads: usize,
        top_k: usize,
    ) -> Result<Self> {
        let model = OrtModel::load(model_path, threads.max(1))?;
        let priors = parse_priors(prior_path.as_ref())?;
        Ok(Self {
            model,
            priors,
            min_conf: 0.4,
            nms_threshold: 0.4,
            top_k: top_k.max(1),
            pending: None,
            ready: None,
        })
    }

    pub fn detect(&mut self, frame: &BgrImage) -> Result<Vec<[f32; 4]>> {
        let w = frame.width as f32;
        let h = frame.height as f32;
        let input = retina_nchw(frame);
        let out = self.model.run(&input)?;
        let loc_t = flatten4(&out[0]);
        let conf_t = if out.len() > 1 {
            flatten_conf(&out[1])
        } else {
            vec![1.0; loc_t.len()]
        };
        let n = self.priors.len().min(loc_t.len()).min(conf_t.len());
        let boxes = decode_boxes(&loc_t[..n], &self.priors[..n]);
        let mut dets = Vec::new();
        for i in 0..n {
            let score = conf_t[i];
            if score <= self.min_conf {
                continue;
            }
            dets.push([
                boxes[i][0] * w,
                boxes[i][1] * h,
                boxes[i][2] * w,
                boxes[i][3] * h,
                score,
            ]);
        }
        let keep = nms(&dets, self.nms_threshold);
        let mut out_boxes = Vec::new();
        for &i in keep.iter().take(self.top_k) {
            let mut x1 = dets[i][0];
            let mut y1 = dets[i][1];
            let mut bw = dets[i][2] - dets[i][0];
            let mut bh = dets[i][3] - dets[i][1];
            let ux = bw * 0.15;
            let uy = bh * 0.2;
            x1 -= ux;
            y1 -= uy;
            bw += ux * 2.0;
            bh += uy * 2.0;
            out_boxes.push([x1, y1, bw, bh]);
        }
        Ok(out_boxes)
    }

    pub fn background_detect(&mut self, frame: &BgrImage) {
        if self.pending.is_some() || self.ready.is_some() {
            return;
        }
        self.pending = Some(frame.clone());
    }

    pub fn get_results(&mut self) -> Vec<[f32; 4]> {
        if let Some(frame) = self.pending.take() {
            self.ready = self.detect(&frame).ok();
        }
        self.ready.take().unwrap_or_default()
    }
}
