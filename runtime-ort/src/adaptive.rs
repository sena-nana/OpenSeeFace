//! Face-size adaptive detection ROI and 224/112 landmark ladder.
//!
//! ONNX graphs have fixed spatial dims, so "dynamic resolution" is a zoomed
//! 224 detector window plus switching `lm_modelV` (112) on large faces.

pub const DET_INPUT: f32 = 224.0;
pub const FAST_LM: i32 = -2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveCfg {
    pub ceiling: i32,
    pub t_fast_px: f32,
    pub hyst_px: f32,
    pub ema_alpha: f32,
    pub det_target_frac: f32,
    pub det_zoom_below: f32,
}

impl Default for AdaptiveCfg {
    /// From `osf-bench --suite scale`: zoom recovers far-face recall; 112px
    /// on faces taller than 200px. GPU ignores the landmark ladder.
    fn default() -> Self {
        Self {
            ceiling: 3,
            t_fast_px: 200.0,
            hyst_px: 24.0,
            ema_alpha: 0.4,
            det_target_frac: 0.42,
            det_zoom_below: 36.0,
        }
    }
}

impl AdaptiveCfg {
    pub fn with_ceiling(mut self, ceiling: i32) -> Self {
        self.ceiling = ceiling;
        self
    }

    pub fn fast_rung(self) -> Option<i32> {
        (self.ceiling >= 0).then_some(FAST_LM)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveState {
    pub ema_h: f32,
    pub lm_type: i32,
}

impl AdaptiveState {
    pub fn new(cfg: AdaptiveCfg) -> Self {
        Self {
            ema_h: 0.0,
            lm_type: cfg.ceiling,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetWindow {
    Full,
    Roi { x1: i32, y1: i32, x2: i32, y2: i32 },
}

impl DetWindow {
    pub fn apply_offset(self, dets: &mut [[f32; 5]]) {
        let (ox, oy) = match self {
            Self::Full => return,
            Self::Roi { x1, y1, .. } => (x1 as f32, y1 as f32),
        };
        for d in dets {
            d[0] += ox;
            d[1] += oy;
        }
    }
}

pub fn face_on_224(face_h: f32, frame_h: u32) -> f32 {
    face_h * DET_INPUT / (frame_h.max(1) as f32)
}

pub fn det_window(
    frame_w: u32,
    frame_h: u32,
    last_box: Option<&[f32; 5]>,
    cfg: &AdaptiveCfg,
) -> DetWindow {
    let Some(b) = last_box else {
        return DetWindow::Full;
    };
    if cfg.det_zoom_below <= 0.0 || face_on_224(b[3], frame_h) >= cfg.det_zoom_below {
        return DetWindow::Full;
    }
    roi_around_face(frame_w, frame_h, b, cfg.det_target_frac)
}

pub fn center_2x(frame_w: u32, frame_h: u32) -> DetWindow {
    let rw = (frame_w as f32 * 0.5).max(8.0);
    let rh = (frame_h as f32 * 0.5).max(8.0);
    let x1 = ((frame_w as f32 - rw) * 0.5).round().max(0.0);
    let y1 = ((frame_h as f32 - rh) * 0.5).round().max(0.0);
    DetWindow::Roi {
        x1: x1 as i32,
        y1: y1 as i32,
        x2: (x1 + rw).min(frame_w as f32) as i32,
        y2: (y1 + rh).min(frame_h as f32) as i32,
    }
}

fn roi_around_face(frame_w: u32, frame_h: u32, b: &[f32; 5], target_frac: f32) -> DetWindow {
    let fw = frame_w as f32;
    let fh = frame_h as f32;
    let cx = b[0] + b[2] * 0.5;
    let cy = b[1] + b[3] * 0.5;
    let side = (b[3].max(1.0) / target_frac.max(0.05))
        .max(8.0)
        .min(fw.max(fh));
    let mut x1 = cx - side * 0.5;
    let mut y1 = cy - side * 0.5;
    if x1 + side > fw {
        x1 = fw - side;
    }
    if y1 + side > fh {
        y1 = fh - side;
    }
    x1 = x1.max(0.0);
    y1 = y1.max(0.0);
    let x2 = (x1 + side).min(fw);
    let y2 = (y1 + side).min(fh);
    if x2 - x1 < 8.0
        || y2 - y1 < 8.0
        || (x1 <= 0.5 && y1 <= 0.5 && x2 >= fw - 0.5 && y2 >= fh - 0.5)
    {
        return DetWindow::Full;
    }
    DetWindow::Roi {
        x1: x1.round() as i32,
        y1: y1.round() as i32,
        x2: x2.round() as i32,
        y2: y2.round() as i32,
    }
}

pub fn pick_lm(state: &mut AdaptiveState, face_h: f32, cfg: &AdaptiveCfg) -> i32 {
    if state.ema_h <= 0.0 {
        state.ema_h = face_h;
    } else {
        state.ema_h = (1.0 - cfg.ema_alpha) * state.ema_h + cfg.ema_alpha * face_h;
    }
    let Some(fast) = cfg.fast_rung() else {
        state.lm_type = cfg.ceiling;
        return state.lm_type;
    };
    if state.lm_type == fast {
        if state.ema_h < cfg.t_fast_px - cfg.hyst_px {
            state.lm_type = cfg.ceiling;
        }
    } else if state.ema_h > cfg.t_fast_px + cfg.hyst_px {
        state.lm_type = fast;
    }
    state.lm_type
}

/// Mean pixel error / inter-ocular distance. Landmarks are `(row, col, conf)`.
pub fn nme(pred: &[[f32; 3]], gt: &[[f32; 3]]) -> Option<f32> {
    let n = pred.len().min(gt.len());
    if n < 46 {
        return None;
    }
    let iod = (gt[36][1] - gt[45][1]).hypot(gt[36][0] - gt[45][0]);
    if iod < 1e-3 {
        return None;
    }
    let s: f32 = (0..n)
        .map(|i| (pred[i][0] - gt[i][0]).hypot(pred[i][1] - gt[i][1]))
        .sum();
    Some(s / n as f32 / iod)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AdaptiveCfg {
        AdaptiveCfg::default()
    }

    #[test]
    fn face_on_224_scales_with_frame() {
        assert!((face_on_224(216.0, 1080) - 44.8).abs() < 1e-3);
        assert!((face_on_224(224.0, 224) - 224.0).abs() < 1e-3);
    }

    #[test]
    fn det_window_full_without_box_or_when_large() {
        let c = AdaptiveCfg {
            det_zoom_below: 48.0,
            ..cfg()
        };
        assert_eq!(det_window(1280, 720, None, &c), DetWindow::Full);
        let large = [400.0, 100.0, 300.0, 400.0, 1.0];
        assert_eq!(det_window(1280, 720, Some(&large), &c), DetWindow::Full);
    }

    #[test]
    fn det_window_centers_roi_on_small_face() {
        let c = AdaptiveCfg {
            det_zoom_below: 80.0,
            det_target_frac: 0.40,
            ..cfg()
        };
        let b = [540.0, 300.0, 160.0, 160.0, 1.0];
        let DetWindow::Roi { x1, y1, x2, y2 } = det_window(1280, 720, Some(&b), &c) else {
            panic!("expected ROI");
        };
        assert!(((x2 - x1) as f32 - 160.0 / 0.40).abs() < 2.0);
        assert!(((x1 + x2) as f32 * 0.5 - 620.0).abs() < 2.0);
        assert!(((y1 + y2) as f32 * 0.5 - 380.0).abs() < 2.0);
    }

    #[test]
    fn det_window_clamps_at_edge() {
        let c = AdaptiveCfg {
            det_zoom_below: 80.0,
            det_target_frac: 0.40,
            ..cfg()
        };
        let b = [0.0, 200.0, 80.0, 80.0, 1.0];
        let DetWindow::Roi { x1, y1, x2, y2 } = det_window(1280, 720, Some(&b), &c) else {
            panic!("expected ROI");
        };
        assert_eq!(x1, 0);
        assert!(y1 >= 0 && y2 <= 720 && x2 > x1);
    }

    #[test]
    fn apply_offset_maps_roi_box_to_image() {
        let w = DetWindow::Roi {
            x1: 100,
            y1: 50,
            x2: 300,
            y2: 250,
        };
        let mut dets = [[10.0, 20.0, 40.0, 40.0, 0.9]];
        w.apply_offset(&mut dets);
        assert!((dets[0][0] - 110.0).abs() < 1e-5);
        assert!((dets[0][1] - 70.0).abs() < 1e-5);
    }

    #[test]
    fn hysteresis_does_not_chatter_near_threshold() {
        let c = AdaptiveCfg {
            ceiling: 3,
            t_fast_px: 280.0,
            hyst_px: 24.0,
            ema_alpha: 1.0,
            ..cfg()
        };
        let mut s = AdaptiveState::new(c);
        assert_eq!(pick_lm(&mut s, 280.0, &c), 3);
        assert_eq!(pick_lm(&mut s, 305.0, &c), FAST_LM);
        assert_eq!(pick_lm(&mut s, 280.0, &c), FAST_LM);
        assert_eq!(pick_lm(&mut s, 255.0, &c), 3);
    }

    #[test]
    fn pick_lm_stays_on_ceiling_when_already_fast_model() {
        let c = AdaptiveCfg {
            ceiling: -2,
            t_fast_px: 10.0,
            hyst_px: 1.0,
            ema_alpha: 1.0,
            ..cfg()
        };
        let mut s = AdaptiveState::new(c);
        assert_eq!(pick_lm(&mut s, 400.0, &c), -2);
    }

    #[test]
    fn nme_is_normalized_by_iod() {
        let mut gt = vec![[0.0, 0.0, 1.0]; 66];
        gt[36] = [10.0, 0.0, 1.0];
        gt[45] = [10.0, 100.0, 1.0];
        let mut pred = gt.clone();
        for p in &mut pred {
            p[1] += 2.0;
        }
        assert!((nme(&pred, &gt).unwrap() - 0.02).abs() < 1e-5);
    }
}
