//! Expression features + remedian, matching `tracker.py` / `remedian.py`.
//!
//! Eye openness (`eye_l` / `eye_r`) uses a continuous calibrator instead of
//! the original 15% snap-to-closed path, which jumped on half-open lids.

use crate::geom::{angle, rotate};

pub const FEATURE_COUNT: usize = 20;
pub type FeatureVec = [f32; FEATURE_COUNT];

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "eye_l",
    "eye_r",
    "eyebrow_steepness_l",
    "eyebrow_updown_l",
    "eyebrow_quirk_l",
    "eyebrow_steepness_r",
    "eyebrow_updown_r",
    "eyebrow_quirk_r",
    "mouth_corner_updown_l",
    "mouth_corner_inout_l",
    "mouth_corner_updown_r",
    "mouth_corner_inout_r",
    "mouth_open",
    "mouth_wide",
    "mouth_pucker",
    "mouth_offset_x",
    "cheek_puff",
    "jaw_open",
    "mouth_funnel",
    "mouth_press_lip_open",
];

/// Official UDP feature slots 14–19.
pub const FEAT_MOUTH_PUCKER: usize = 14;
pub const FEAT_MOUTH_OFFSET_X: usize = 15;
pub const FEAT_CHEEK_PUFF: usize = 16;
pub const FEAT_JAW_OPEN: usize = 17;
pub const FEAT_MOUTH_FUNNEL: usize = 18;
pub const FEAT_MOUTH_PRESS_LIP_OPEN: usize = 19;

/// Bias so [`Feature`] median stays away from 0 (signed offset around rest).
const OFFSET_BIAS: f32 = 1.0;
const PUCKER_Z_WEIGHT: f32 = 0.5;
const CHEEK_Z_WEIGHT: f32 = 0.5;
const CHEEK_SMILE_GATE: f32 = 0.3;
const CHEEK_OPEN_GATE: f32 = 0.5;
const FUNNEL_NARROW_FRAC: f32 = 0.92;
const FUNNEL_GAP_MIN: f32 = 0.2;
const PRESS_W: f32 = 0.8;
const PRESS_LIP_BIAS: f32 = 6.0;

struct Remedian {
    k: usize,
    all: Vec<f32>,
    more: Option<Box<Remedian>>,
    cached: Option<f32>,
}

impl Remedian {
    fn new() -> Self {
        Self {
            k: 64,
            all: Vec::new(),
            more: None,
            cached: None,
        }
    }

    fn add(&mut self, x: f32) {
        self.cached = None;
        self.all.push(x);
        if self.all.len() == self.k {
            let m = median(&self.all);
            let more = self.more.get_or_insert_with(|| Box::new(Remedian::new()));
            more.add(m);
            self.all.clear();
        }
    }

    fn median(&mut self) -> f32 {
        if let Some(more) = self.more.as_mut() {
            more.median()
        } else {
            median(&self.all)
        }
    }
}

fn median(lst: &[f32]) -> f32 {
    if lst.is_empty() {
        return 0.0;
    }
    let mut v = lst.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n < 3 {
        return (v[0] + v[n - 1]) / 2.0;
    }
    let p = n / 2;
    if n % 2 == 0 {
        (v[p] + v[p - 1]) / 2.0
    } else {
        v[p]
    }
}

struct Feature {
    median: Remedian,
    min: Option<f32>,
    max: Option<f32>,
    hard_min: Option<f32>,
    hard_max: Option<f32>,
    threshold: f32,
    alpha: f32,
    hard_factor: f32,
    decay: f32,
    last: f32,
    current_median: f32,
    max_feature_updates: f32,
    first_seen: f32,
    updating: bool,
    one_sided: bool,
}

impl Feature {
    fn new(threshold: f32, max_feature_updates: f32) -> Self {
        Self::new_ex(threshold, max_feature_updates, false)
    }

    fn new_positive(threshold: f32, max_feature_updates: f32) -> Self {
        Self::new_ex(threshold, max_feature_updates, true)
    }

    fn new_ex(threshold: f32, max_feature_updates: f32, one_sided: bool) -> Self {
        Self {
            median: Remedian::new(),
            min: None,
            max: None,
            hard_min: None,
            hard_max: None,
            threshold,
            alpha: 0.2,
            hard_factor: 0.15,
            decay: 0.001,
            last: 0.0,
            current_median: 0.0,
            max_feature_updates,
            first_seen: -1.0,
            updating: true,
            one_sided,
        }
    }

    fn update(&mut self, x: f32, now: f32) -> f32 {
        if self.max_feature_updates > 0.0 && self.first_seen < 0.0 {
            self.first_seen = now;
        }
        let new = self.update_state(x, now);
        let filtered = self.last * self.alpha + new * (1.0 - self.alpha);
        self.last = filtered;
        filtered
    }

    fn raw_median(&self) -> f32 {
        self.current_median
    }

    fn update_state(&mut self, x: f32, now: f32) -> f32 {
        let updating = self.updating
            && (self.max_feature_updates == 0.0
                || now - self.first_seen < self.max_feature_updates);
        if updating {
            self.median.add(x);
            self.current_median = self.median.median();
        } else {
            self.updating = false;
        }
        let median = self.current_median;
        if self.one_sided {
            return self.update_positive(x, median, updating);
        }

        if self.min.is_none() {
            if x < median && median != 0.0 && (median - x) / median > self.threshold {
                if updating {
                    self.min = Some(x);
                    self.hard_min = Some(x + self.hard_factor * (median - x));
                }
                return -1.0;
            }
            return 0.0;
        }
        if x < self.min.unwrap() {
            if updating {
                self.min = Some(x);
                self.hard_min = Some(x + self.hard_factor * (median - x));
            }
            return -1.0;
        }
        if self.max.is_none() {
            if x > median && median != 0.0 && (x - median) / median > self.threshold {
                if updating {
                    self.max = Some(x);
                    self.hard_max = Some(x - self.hard_factor * (x - median));
                }
                return 1.0;
            }
            return 0.0;
        }
        if x > self.max.unwrap() {
            if updating {
                self.max = Some(x);
                self.hard_max = Some(x - self.hard_factor * (x - median));
            }
            return 1.0;
        }

        if updating {
            if let (Some(min), Some(hard_min)) = (self.min, self.hard_min) {
                if min < hard_min {
                    self.min = Some(hard_min * self.decay + min * (1.0 - self.decay));
                }
            }
            if let (Some(max), Some(hard_max)) = (self.max, self.hard_max) {
                if max > hard_max {
                    self.max = Some(hard_max * self.decay + max * (1.0 - self.decay));
                }
            }
        }

        if x < median {
            let min = self.min.unwrap();
            if (median - min).abs() < 1e-8 {
                0.0
            } else {
                -(1.0 - (x - min) / (median - min))
            }
        } else if x > median {
            let max = self.max.unwrap();
            if (max - median).abs() < 1e-8 {
                0.0
            } else {
                (x - median) / (max - median)
            }
        } else {
            0.0
        }
    }

    /// Rest is 0; only values above the median count.
    fn update_positive(&mut self, x: f32, median: f32, updating: bool) -> f32 {
        if median == 0.0 || x <= median {
            return 0.0;
        }
        if self.max.is_none() {
            if (x - median) / median > self.threshold {
                if updating {
                    self.max = Some(x);
                    self.hard_max = Some(x - self.hard_factor * (x - median));
                }
                return 1.0;
            }
            return 0.0;
        }
        if x > self.max.unwrap() {
            if updating {
                self.max = Some(x);
                self.hard_max = Some(x - self.hard_factor * (x - median));
            }
            return 1.0;
        }
        if updating {
            if let (Some(max), Some(hard_max)) = (self.max, self.hard_max) {
                if max > hard_max {
                    self.max = Some(hard_max * self.decay + max * (1.0 - self.decay));
                }
            }
        }
        let max = self.max.unwrap();
        if (max - median).abs() < 1e-8 {
            0.0
        } else {
            (x - median) / (max - median)
        }
    }
}

/// Landmark EAR at a fully shut lid (~8% of the open-eye median).
const EYE_CLOSED_FRAC: f32 = 0.08;
const EYE_ALPHA: f32 = 0.6;
const EYE_ALPHA_OUTLIER: f32 = 0.82;
/// Ignore blinks / held squints when updating the open-eye baseline.
const EYE_MEDIAN_GATE: f32 = 0.70;
/// Frame height looks like an extra-open eye; skip median updates.
const EYE_FRAME_FRAC: f32 = 1.12;

/// Continuous eye openness. Mouth / brow features keep [`Feature`].
struct EyeFeature {
    median: Remedian,
    last: f32,
    current_median: f32,
    max_feature_updates: f32,
    first_seen: f32,
    updating: bool,
    outlier_run: u8,
}

impl EyeFeature {
    fn new(max_feature_updates: f32) -> Self {
        Self {
            median: Remedian::new(),
            last: 0.0,
            current_median: 0.0,
            max_feature_updates,
            first_seen: -1.0,
            updating: true,
            outlier_run: 0,
        }
    }

    fn update(&mut self, x: f32, now: f32) -> f32 {
        self.update_ex(x, now, None)
    }

    fn update_ex(&mut self, x: f32, now: f32, eye_conf: Option<f32>) -> f32 {
        if self.max_feature_updates > 0.0 && self.first_seen < 0.0 {
            self.first_seen = now;
        }
        let updating = self.updating
            && (self.max_feature_updates == 0.0
                || now - self.first_seen < self.max_feature_updates);
        let weak = eye_conf.map(|c| c < 0.4).unwrap_or(false);
        let too_wide = self.current_median > 1e-8 && x > self.current_median * EYE_FRAME_FRAC;
        let outlier = weak || too_wide;
        if outlier {
            self.outlier_run = self.outlier_run.saturating_add(1);
        } else {
            self.outlier_run = 0;
        }
        if updating {
            let allow_median = !weak && !too_wide;
            if allow_median
                && (self.current_median.abs() < 1e-8 || x >= self.current_median * EYE_MEDIAN_GATE)
            {
                self.median.add(x);
                self.current_median = self.median.median();
            }
        } else {
            self.updating = false;
        }
        let new = eye_value(x, self.current_median);
        let alpha = if self.outlier_run >= 2 {
            EYE_ALPHA_OUTLIER
        } else {
            EYE_ALPHA
        };
        self.last = self.last * alpha + new * (1.0 - alpha);
        self.last
    }
}

/// Open median → 0, shut (`EYE_CLOSED_FRAC`) → −1. Half-open and a lid slit stay analog.
fn eye_value(x: f32, median: f32) -> f32 {
    if median.abs() < 1e-8 || x >= median {
        0.0
    } else {
        ((x - median) / (median * (1.0 - EYE_CLOSED_FRAC))).clamp(-1.0, 0.0)
    }
}

/// Vertical lid gap over inter-corner width after aligning the eye horizontally.
fn eye_aspect(outer: [f32; 2], inner: [f32; 2], lids: [[f32; 2]; 4]) -> (f32, f32) {
    let (alpha, f_pts) = FeatureExtractor::align_points(outer, inner, &lids);
    let gap = ((f_pts[0][1] + f_pts[1][1]) / 2.0 - (f_pts[2][1] + f_pts[3][1]) / 2.0).abs();
    let eye_w = (inner[0] - outer[0]).hypot(inner[1] - outer[1]).max(1e-6);
    (alpha, gap / eye_w)
}

pub struct FeatureExtractor {
    eye_l: EyeFeature,
    eye_r: EyeFeature,
    eyebrow_updown_l: Feature,
    eyebrow_updown_r: Feature,
    eyebrow_quirk_l: Feature,
    eyebrow_quirk_r: Feature,
    eyebrow_steepness_l: Feature,
    eyebrow_steepness_r: Feature,
    mouth_corner_updown_l: Feature,
    mouth_corner_updown_r: Feature,
    mouth_corner_inout_l: Feature,
    mouth_corner_inout_r: Feature,
    mouth_open: Feature,
    mouth_wide: Feature,
    mouth_pucker: Feature,
    mouth_offset_x: Feature,
    cheek_puff: Feature,
    jaw_open: Feature,
    mouth_funnel: Feature,
    mouth_press_lip_open: Feature,
}

impl FeatureExtractor {
    pub fn new(max_feature_updates: f32) -> Self {
        Self {
            eye_l: EyeFeature::new(max_feature_updates),
            eye_r: EyeFeature::new(max_feature_updates),
            eyebrow_updown_l: Feature::new(0.15, max_feature_updates),
            eyebrow_updown_r: Feature::new(0.15, max_feature_updates),
            eyebrow_quirk_l: Feature::new(0.05, max_feature_updates),
            eyebrow_quirk_r: Feature::new(0.05, max_feature_updates),
            eyebrow_steepness_l: Feature::new(0.05, max_feature_updates),
            eyebrow_steepness_r: Feature::new(0.05, max_feature_updates),
            mouth_corner_updown_l: Feature::new(0.15, max_feature_updates),
            mouth_corner_updown_r: Feature::new(0.15, max_feature_updates),
            mouth_corner_inout_l: Feature::new(0.02, max_feature_updates),
            mouth_corner_inout_r: Feature::new(0.02, max_feature_updates),
            mouth_open: Feature::new(0.15, max_feature_updates),
            mouth_wide: Feature::new(0.02, max_feature_updates),
            mouth_pucker: Feature::new(0.02, max_feature_updates),
            mouth_offset_x: Feature::new(0.02, max_feature_updates),
            cheek_puff: Feature::new(0.02, max_feature_updates),
            jaw_open: Feature::new_positive(0.15, max_feature_updates),
            mouth_funnel: Feature::new_positive(0.05, max_feature_updates),
            mouth_press_lip_open: Feature::new(0.05, max_feature_updates),
        }
    }

    fn align_points(a: [f32; 2], b: [f32; 2], pts: &[[f32; 2]]) -> (f32, Vec<[f32; 2]>) {
        let mut alpha = angle(a, b).to_degrees();
        if alpha >= 90.0 {
            alpha = -(alpha - 180.0);
        }
        if alpha <= -90.0 {
            alpha = -(alpha + 180.0);
        }
        let alpha = alpha.to_radians();
        let aligned = pts
            .iter()
            .map(|pt| {
                let (x, y) = rotate((a[0], a[1]), (pt[0], pt[1]), alpha);
                [x, y]
            })
            .collect();
        (alpha, aligned)
    }

    pub fn update(&mut self, pts: &[[f32; 3]], full: bool) -> FeatureVec {
        self.update_ex(pts, full, None)
    }

    pub fn update_ex(&mut self, pts: &[[f32; 3]], full: bool, eye_conf: Option<f32>) -> FeatureVec {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f32())
            .unwrap_or(0.0);

        let p = |i: usize| [pts[i][0], pts[i][1]];
        let norm_distance_x = ((pts[0][0] - pts[16][0]) + (pts[1][0] - pts[15][0])) / 2.0;
        let norm_distance_y =
            ((pts[27][1] - pts[28][1]) + (pts[28][1] - pts[29][1]) + (pts[29][1] - pts[30][1]))
                / 3.0;
        let ny = norm_distance_y.abs().max(1e-6);
        let nx = norm_distance_x.abs().max(1e-6);

        let (a1, f) = eye_aspect(p(42), p(45), [p(43), p(44), p(47), p(46)]);
        let eye_l = self.eye_l.update_ex(f, now, eye_conf);

        let (a2, f) = eye_aspect(p(36), p(39), [p(37), p(38), p(41), p(40)]);
        let eye_r = self.eye_r.update_ex(f, now, eye_conf);

        let (steep_l, quirk_l, steep_r, quirk_r) = if full {
            let (a3, _) = Self::align_points(p(0), p(16), &[]);
            let (a4, _) = Self::align_points(p(31), p(35), &[]);
            let norm_angle =
                (a1.to_degrees() + a2.to_degrees() + a3.to_degrees() + a4.to_degrees()) / 4.0;
            let (a, f_pts) = Self::align_points(p(22), p(26), &[p(22), p(23), p(24), p(25), p(26)]);
            let steep_l = self
                .eyebrow_steepness_l
                .update(-a.to_degrees() - norm_angle, now);
            let f = f_pts[1..4]
                .iter()
                .map(|q| (q[1] - f_pts[0][1]).abs())
                .fold(0.0f32, f32::max)
                / ny;
            let quirk_l = self.eyebrow_quirk_l.update(f, now);

            let (a, f_pts) = Self::align_points(p(17), p(21), &[p(17), p(18), p(19), p(20), p(21)]);
            let steep_r = self
                .eyebrow_steepness_r
                .update(a.to_degrees() - norm_angle, now);
            let f = f_pts[1..4]
                .iter()
                .map(|q| (q[1] - f_pts[0][1]).abs())
                .fold(0.0f32, f32::max)
                / ny;
            let quirk_r = self.eyebrow_quirk_r.update(f, now);
            (steep_l, quirk_l, steep_r, quirk_r)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let f = ((pts[22][1] + pts[26][1]) / 2.0 - pts[27][1]) / ny;
        let up_l = self.eyebrow_updown_l.update(f, now);
        let f = ((pts[17][1] + pts[21][1]) / 2.0 - pts[27][1]) / ny;
        let up_r = self.eyebrow_updown_r.update(f, now);

        let upper_mouth = (pts[49][1] + pts[50][1] + pts[51][1]) / 3.0;
        let center_line =
            (pts[50][0] + pts[60][0] + pts[27][0] + pts[30][0] + pts[64][0] + pts[55][0]) / 6.0;

        let f = (upper_mouth - pts[62][1]) / ny;
        let m_ud_l = self.mouth_corner_updown_l.update(f, now);
        let m_io_l = if full {
            self.mouth_corner_inout_l
                .update((center_line - pts[62][0]).abs() / nx, now)
        } else {
            0.0
        };
        let f = (upper_mouth - pts[58][1]) / ny;
        let m_ud_r = self.mouth_corner_updown_r.update(f, now);
        let m_io_r = if full {
            self.mouth_corner_inout_r
                .update((center_line - pts[58][0]).abs() / nx, now)
        } else {
            0.0
        };

        let inner_gap = ((pts[59][1] + pts[60][1] + pts[61][1]) / 3.0
            - (pts[63][1] + pts[64][1] + pts[65][1]) / 3.0)
            .abs()
            / ny;
        let mouth_open = self.mouth_open.update(inner_gap, now);
        let mouth_w = (pts[58][0] - pts[62][0]).abs() / nx;
        let mouth_wide = self.mouth_wide.update(mouth_w, now);

        let lip_z = (pts[48][2] + pts[52][2] + pts[58][2] + pts[62][2]) / 4.0 - pts[30][2];
        let mouth_pucker = self
            .mouth_pucker
            .update(1.0 - mouth_w + PUCKER_Z_WEIGHT * lip_z, now);

        let jaw_span = (pts[30][1] - pts[8][1]).abs() / ny;
        let jaw_open = self.jaw_open.update(jaw_span, now);

        let mut mouth_funnel = self.mouth_funnel.update(inner_gap * (1.0 - mouth_w), now);
        let rest_w = self.mouth_wide.raw_median();
        if rest_w <= 1e-6 || mouth_w >= rest_w * FUNNEL_NARROW_FRAC || inner_gap < FUNNEL_GAP_MIN {
            mouth_funnel = 0.0;
        }

        let upper_outer = (pts[49][1] + pts[50][1] + pts[51][1]) / 3.0;
        let lower_outer = (pts[54][1] + pts[55][1] + pts[56][1]) / 3.0;
        let inner_upper = (pts[59][1] + pts[60][1] + pts[61][1]) / 3.0;
        let inner_lower = (pts[63][1] + pts[64][1] + pts[65][1]) / 3.0;
        let mut press_raw = -(pts[30][1] - upper_outer) / ny - (lower_outer - pts[8][1]) / ny
            + PRESS_W * ((upper_outer - inner_upper) + (inner_lower - lower_outer)) / ny
            + PRESS_LIP_BIAS;
        let med = self.mouth_press_lip_open.raw_median();
        let block_minus = mouth_pucker > CHEEK_SMILE_GATE
            || mouth_wide > CHEEK_SMILE_GATE
            || inner_gap > FUNNEL_GAP_MIN
            || jaw_open > CHEEK_SMILE_GATE;
        if med.abs() > 1e-6 {
            if mouth_funnel > 0.0 {
                press_raw = press_raw.min(med);
            }
            if block_minus {
                press_raw = press_raw.max(med);
            }
        }
        let mut mouth_press_lip_open = self.mouth_press_lip_open.update(press_raw, now);
        if mouth_funnel > 0.0 {
            mouth_press_lip_open = mouth_press_lip_open.min(0.0);
        }
        if block_minus {
            mouth_press_lip_open = mouth_press_lip_open.max(0.0);
        }

        let mouth_cx =
            (pts[50][0] + pts[55][0] + pts[58][0] + pts[60][0] + pts[62][0] + pts[64][0]) / 6.0;
        let mouth_offset_x = self.mouth_offset_x.update(mouth_cx / nx + OFFSET_BIAS, now);

        let eye_span = (pts[36][0] - pts[45][0]).abs().max(1e-6);
        let cheek_span = ((pts[2][0] - pts[14][0]).abs() + (pts[3][0] - pts[13][0]).abs()) / 2.0;
        let cheek_z =
            (pts[2][2] + pts[3][2] + pts[4][2] + pts[12][2] + pts[13][2] + pts[14][2]) / 6.0;
        let mut cheek_puff = self
            .cheek_puff
            .update(cheek_span / eye_span + CHEEK_Z_WEIGHT * cheek_z, now);
        if mouth_wide > CHEEK_SMILE_GATE || mouth_open > CHEEK_OPEN_GATE {
            cheek_puff = 0.0;
        } else {
            cheek_puff = cheek_puff.max(0.0);
        }

        [
            eye_l,
            eye_r,
            steep_l,
            up_l,
            quirk_l,
            steep_r,
            up_r,
            quirk_r,
            m_ud_l,
            m_io_l,
            m_ud_r,
            m_io_r,
            mouth_open,
            mouth_wide,
            mouth_pucker,
            mouth_offset_x,
            cheek_puff,
            jaw_open,
            mouth_funnel,
            mouth_press_lip_open,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(eye: &mut EyeFeature, x: f32, n: u32) -> f32 {
        let mut v = 0.0;
        for i in 0..n {
            v = eye.update(x, i as f32 * 0.03);
        }
        v
    }

    #[test]
    fn lid_close_is_continuous() {
        let m = 1.0;
        let half = eye_value(0.55, m);
        let slit = eye_value(0.25, m);
        let shut = eye_value(0.08, m);
        assert!(half < -0.2 && half > slit);
        assert!(slit < -0.5 && slit > shut);
        assert!((shut + 1.0).abs() < 1e-5);
        assert_eq!(eye_value(1.0, m), 0.0);
    }

    #[test]
    fn squint_does_not_snap_like_legacy() {
        let mut old = Feature::new(0.15, 0.0);
        let mut eye = EyeFeature::new(0.0);
        for i in 0..80 {
            let t = i as f32 * 0.03;
            old.update(1.0, t);
            eye.update(1.0, t);
        }
        let mut seed = 1u32;
        let mut old_cross = 0;
        let mut prev_old = 0.0;
        let mut new_vals = Vec::new();
        for i in 0..80 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (seed >> 8) as f32 / u32::MAX as f32 * 2.0 - 1.0;
            let x = 0.55 * (1.0 + 0.05 * n);
            let t = (80 + i) as f32 * 0.03;
            let o = old.update(x, t);
            if (prev_old > -0.1 && o < -0.7) || (prev_old < -0.7 && o > -0.1) {
                old_cross += 1;
            }
            prev_old = o;
            new_vals.push(eye.update(x, t));
        }
        assert!(old_cross > 0, "legacy Feature should bounce 0/-1");
        for v in &new_vals[8..] {
            assert!(*v < -0.2 && *v > -0.8, "half-open snapped: {v}");
        }
    }

    #[test]
    fn blink_reaches_shut_then_opens() {
        let mut eye = EyeFeature::new(0.0);
        feed(&mut eye, 1.0, 80);
        let shut = feed(&mut eye, 0.10, 8);
        assert!(shut < -0.85, "blink {shut}");
        let open = feed(&mut eye, 1.0, 20);
        assert!(open.abs() < 0.2, "recover {open}");
    }

    #[test]
    fn extractor_uses_eye_width_and_holds_squint() {
        let (_, ear) = eye_aspect(
            [0.0, 0.0],
            [2.0, 0.0],
            [[0.5, 0.4], [1.5, 0.4], [1.5, 0.0], [0.5, 0.0]],
        );
        assert!((ear - 0.2).abs() < 1e-5);

        let mut pts = vec![[0.0; 3]; 66];
        pts[0] = [0.45, 0.30, 0.0];
        pts[16] = [-0.45, 0.30, 0.0];
        pts[27] = [0.0, 0.29, 0.0];
        pts[28] = [0.0, 0.19, 0.0];
        pts[29] = [0.0, 0.10, 0.0];
        let set_eye = |pts: &mut [[f32; 3]], outer: usize, sign: f32, lid: f32| {
            pts[outer] = [sign * 0.32, 0.30, 0.0];
            pts[outer + 1] = [sign * 0.27, 0.30 + 0.04 * lid, 0.0];
            pts[outer + 2] = [sign * 0.18, 0.30 + 0.04 * lid, 0.0];
            pts[outer + 3] = [sign * 0.13, 0.28, 0.0];
            pts[outer + 4] = [sign * 0.18, 0.26, 0.0];
            pts[outer + 5] = [sign * 0.27, 0.26, 0.0];
        };
        let mut ext = FeatureExtractor::new(0.0);
        set_eye(&mut pts, 36, 1.0, 1.0);
        set_eye(&mut pts, 42, -1.0, 1.0);
        for _ in 0..80 {
            ext.update(&pts, false);
        }
        set_eye(&mut pts, 36, 1.0, 0.45);
        set_eye(&mut pts, 42, -1.0, 0.45);
        for _ in 0..20 {
            let f = ext.update(&pts, false);
            assert!(f[0] < -0.05 && f[0] > -0.95);
            assert!(f[1] < -0.05 && f[1] > -0.95);
        }
    }

    #[test]
    fn outlier_keeps_calibrated_median() {
        let mut eye = EyeFeature::new(0.0);
        feed(&mut eye, 0.25, 80);
        for i in 0..40 {
            eye.update_ex(0.45, 3.0 + i as f32 * 0.03, Some(0.9));
        }
        let open = eye.update_ex(0.25, 5.0, Some(0.9));
        assert!(
            open.abs() < 0.2,
            "true-open EAR treated as closed after frame-height: {open}"
        );
    }

    fn canonical_face() -> Vec<[f32; 3]> {
        let mut p = vec![[0.0f32; 3]; 66];
        for i in 0..17 {
            let t = i as f32 / 16.0;
            let x = 0.45 * (1.0 - 2.0 * t);
            let y = 0.30 - (std::f32::consts::PI * t).sin() * 0.90;
            p[i] = [x, y, -0.72];
        }
        p[27] = [0.0, 0.29, -0.14];
        p[28] = [0.0, 0.19, -0.07];
        p[29] = [0.0, 0.10, -0.01];
        p[30] = [0.0, 0.00, 0.00];
        p[36] = [0.32, 0.30, -0.28];
        p[37] = [0.27, 0.34, -0.25];
        p[38] = [0.18, 0.34, -0.24];
        p[39] = [0.13, 0.28, -0.23];
        p[40] = [0.18, 0.26, -0.24];
        p[41] = [0.27, 0.26, -0.25];
        p[42] = [-0.13, 0.28, -0.23];
        p[43] = [-0.18, 0.34, -0.24];
        p[44] = [-0.27, 0.34, -0.25];
        p[45] = [-0.32, 0.30, -0.28];
        p[46] = [-0.27, 0.26, -0.25];
        p[47] = [-0.18, 0.26, -0.24];
        p[48] = [0.12, -0.21, -0.16];
        p[49] = [0.04, -0.19, -0.10];
        p[50] = [0.00, -0.21, -0.08];
        p[51] = [-0.04, -0.19, -0.10];
        p[52] = [-0.12, -0.21, -0.16];
        p[53] = [-0.13, -0.29, -0.19];
        p[54] = [-0.06, -0.33, -0.16];
        p[55] = [0.00, -0.34, -0.11];
        p[56] = [0.06, -0.33, -0.16];
        p[57] = [0.13, -0.29, -0.19];
        p[58] = [0.18, -0.24, -0.23];
        p[59] = [0.08, -0.24, -0.16];
        p[60] = [0.00, -0.26, -0.10];
        p[61] = [-0.08, -0.24, -0.16];
        p[62] = [-0.18, -0.24, -0.23];
        p[63] = [-0.07, -0.25, -0.18];
        p[64] = [0.00, -0.26, -0.11];
        p[65] = [0.07, -0.25, -0.18];
        p
    }

    fn last_after(ext: &mut FeatureExtractor, pts: &[[f32; 3]], n: u32) -> FeatureVec {
        let mut f = [0.0; FEATURE_COUNT];
        for _ in 0..n {
            f = ext.update(pts, true);
        }
        f
    }

    fn shift_mouth_x(pts: &[[f32; 3]], dx: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        for i in 48..66 {
            p[i][0] += dx;
        }
        p
    }

    fn pucker_lips(pts: &[[f32; 3]], corner_scale: f32, forward: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        p[58][0] *= corner_scale;
        p[62][0] *= corner_scale;
        for i in [48, 52, 58, 62] {
            p[i][2] += forward;
        }
        p
    }

    fn scale_cheeks(pts: &[[f32; 3]], scale: f32, forward: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        for i in [2, 3, 4, 12, 13, 14] {
            p[i][0] *= scale;
            p[i][2] += forward;
        }
        p
    }

    const DETECT_TH: f32 = 0.3;

    #[derive(Clone, Copy, Debug)]
    enum MouthLabel {
        Neutral,
        Pucker,
        OffsetRight,
        OffsetLeft,
        Puff,
        Smile,
        Open,
    }

    const LABELS: [MouthLabel; 7] = [
        MouthLabel::Neutral,
        MouthLabel::Pucker,
        MouthLabel::OffsetRight,
        MouthLabel::OffsetLeft,
        MouthLabel::Puff,
        MouthLabel::Smile,
        MouthLabel::Open,
    ];

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 8) as f32 / (u32::MAX >> 8) as f32
    }

    fn jitter(pts: &[[f32; 3]], seed: &mut u32, sigma: f32) -> Vec<[f32; 3]> {
        pts.iter()
            .map(|p| {
                [
                    p[0] + (lcg(seed) * 2.0 - 1.0) * sigma,
                    p[1] + (lcg(seed) * 2.0 - 1.0) * sigma,
                    p[2] + (lcg(seed) * 2.0 - 1.0) * sigma,
                ]
            })
            .collect()
    }

    fn lerp_pts(a: &[[f32; 3]], b: &[[f32; 3]], t: f32) -> Vec<[f32; 3]> {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| {
                [
                    p[0] + (q[0] - p[0]) * t,
                    p[1] + (q[1] - p[1]) * t,
                    p[2] + (q[2] - p[2]) * t,
                ]
            })
            .collect()
    }

    fn smile_mouth(pts: &[[f32; 3]], corner_scale: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        p[58][0] *= corner_scale;
        p[62][0] *= corner_scale;
        p
    }

    fn open_mouth(pts: &[[f32; 3]], gap_scale: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        for i in [59, 60, 61] {
            p[i][1] += 0.04 * (gap_scale - 1.0);
        }
        for i in [63, 64, 65, 53, 54, 55, 56, 57] {
            p[i][1] -= 0.05 * (gap_scale - 1.0);
        }
        p
    }

    fn drop_chin(pts: &[[f32; 3]], dy: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        for i in [6, 7, 8, 9, 10] {
            let w = 1.0 - (i as f32 - 8.0).abs() / 3.0;
            p[i][1] -= dy * w;
        }
        p
    }

    fn funnel_mouth(pts: &[[f32; 3]], corner_scale: f32, gap_scale: f32) -> Vec<[f32; 3]> {
        let mut p = pucker_lips(pts, corner_scale, 0.04);
        for i in [59, 60, 61] {
            p[i][1] += 0.04 * (gap_scale - 1.0);
        }
        for i in [63, 64, 65, 53, 54, 55, 56, 57] {
            p[i][1] -= 0.05 * (gap_scale - 1.0);
        }
        p
    }

    fn bare_teeth(pts: &[[f32; 3]], dy: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        for i in [48, 49, 50, 51, 52, 59, 60, 61] {
            p[i][1] += dy;
        }
        for i in [53, 54, 55, 56, 57, 63, 64, 65] {
            p[i][1] -= dy;
        }
        p
    }

    fn press_lips(pts: &[[f32; 3]], collapse: f32) -> Vec<[f32; 3]> {
        let mut p = pts.to_vec();
        let ou = (p[49][1] + p[50][1] + p[51][1]) / 3.0;
        let iu = (p[59][1] + p[60][1] + p[61][1]) / 3.0;
        let il = (p[63][1] + p[64][1] + p[65][1]) / 3.0;
        let ol = (p[54][1] + p[55][1] + p[56][1]) / 3.0;
        for i in [48, 49, 50, 51, 52] {
            p[i][1] += (iu - ou) * collapse;
        }
        for i in [53, 54, 55, 56, 57] {
            p[i][1] += (il - ol) * collapse;
        }
        p
    }

    fn apply_label(rest: &[[f32; 3]], label: MouthLabel, intensity: f32) -> Vec<[f32; 3]> {
        let t = intensity.clamp(0.0, 1.0);
        let target = match label {
            MouthLabel::Neutral => rest.to_vec(),
            MouthLabel::Pucker => pucker_lips(rest, 0.45, 0.10),
            MouthLabel::OffsetRight => shift_mouth_x(rest, 0.12),
            MouthLabel::OffsetLeft => shift_mouth_x(rest, -0.12),
            MouthLabel::Puff => scale_cheeks(rest, 1.25, 0.08),
            MouthLabel::Smile => smile_mouth(rest, 1.6),
            MouthLabel::Open => open_mouth(rest, 3.0),
        };
        lerp_pts(rest, &target, t)
    }

    fn calibrate_mouth(ext: &mut FeatureExtractor, rest: &[[f32; 3]]) {
        last_after(ext, rest, 80);
        last_after(ext, &pucker_lips(rest, 1.5, -0.04), 12);
        last_after(ext, &pucker_lips(rest, 0.45, 0.10), 12);
        last_after(ext, &shift_mouth_x(rest, -0.12), 12);
        last_after(ext, &shift_mouth_x(rest, 0.12), 12);
        last_after(ext, &scale_cheeks(rest, 0.85, -0.04), 12);
        last_after(ext, &scale_cheeks(rest, 1.25, 0.08), 12);
        last_after(ext, &pucker_lips(rest, 0.7, 0.0), 12);
        last_after(ext, &open_mouth(rest, 3.0), 12);
        last_after(ext, &drop_chin(rest, 0.14), 12);
        last_after(ext, &funnel_mouth(rest, 0.55, 2.8), 12);
        last_after(ext, &bare_teeth(rest, 0.05), 12);
        last_after(ext, &press_lips(rest, 0.7), 12);
        last_after(ext, rest, 16);
    }

    fn read_noisy(
        ext: &mut FeatureExtractor,
        pts: &[[f32; 3]],
        n: u32,
        seed: &mut u32,
        sigma: f32,
    ) -> FeatureVec {
        let mut f = [0.0; FEATURE_COUNT];
        for _ in 0..n {
            let j = jitter(pts, seed, sigma);
            f = ext.update(&j, true);
        }
        f
    }

    #[derive(Default, Clone, Copy)]
    struct Conf {
        tp: u32,
        fp: u32,
        fn_: u32,
        tn: u32,
    }

    impl Conf {
        fn add(&mut self, pred: bool, truth: bool) {
            match (pred, truth) {
                (true, true) => self.tp += 1,
                (true, false) => self.fp += 1,
                (false, true) => self.fn_ += 1,
                (false, false) => self.tn += 1,
            }
        }

        fn recall(self) -> f32 {
            let d = self.tp + self.fn_;
            if d == 0 {
                1.0
            } else {
                self.tp as f32 / d as f32
            }
        }

        fn precision(self) -> f32 {
            let d = self.tp + self.fp;
            if d == 0 {
                1.0
            } else {
                self.tp as f32 / d as f32
            }
        }

        fn fpr(self) -> f32 {
            let d = self.fp + self.tn;
            if d == 0 {
                0.0
            } else {
                self.fp as f32 / d as f32
            }
        }

        fn accuracy(self) -> f32 {
            let d = self.tp + self.fp + self.fn_ + self.tn;
            if d == 0 {
                1.0
            } else {
                (self.tp + self.tn) as f32 / d as f32
            }
        }
    }

    fn detect(f: &FeatureVec) -> (bool, i8, bool) {
        let pucker = f[FEAT_MOUTH_PUCKER] > DETECT_TH;
        let offset = if f[FEAT_MOUTH_OFFSET_X] > DETECT_TH {
            1
        } else if f[FEAT_MOUTH_OFFSET_X] < -DETECT_TH {
            -1
        } else {
            0
        };
        let puff = f[FEAT_CHEEK_PUFF] > DETECT_TH;
        (pucker, offset, puff)
    }

    /// Labeled noisy poses: recall / precision / FPR / accuracy for the three new features.
    #[test]
    fn mouth_features_detection_accuracy() {
        const N_ID: u32 = 24;
        const SIGMA: f32 = 0.0035;
        let rest0 = canonical_face();
        let mut pucker_c = Conf::default();
        let mut offset_c = Conf::default();
        let mut puff_c = Conf::default();
        let mut puff_on_smile_fp = 0u32;
        let mut puff_on_smile_n = 0u32;
        let mut seed: u32 = 0xC0FFEE;

        for _ in 0..N_ID {
            let identity = jitter(&rest0, &mut seed, 0.006);
            let mut ext = FeatureExtractor::new(0.0);
            calibrate_mouth(&mut ext, &identity);
            let intensity = 0.85 + 0.15 * lcg(&mut seed);
            for label in LABELS {
                let pose = apply_label(&identity, label, intensity);
                let f = read_noisy(&mut ext, &pose, 16, &mut seed, SIGMA);
                let (pucker, offset, puff) = detect(&f);
                pucker_c.add(pucker, matches!(label, MouthLabel::Pucker));
                let want_off = match label {
                    MouthLabel::OffsetRight => 1,
                    MouthLabel::OffsetLeft => -1,
                    _ => 0,
                };
                if want_off != 0 {
                    offset_c.add(offset == want_off, true);
                } else {
                    offset_c.add(offset != 0, false);
                }
                puff_c.add(puff, matches!(label, MouthLabel::Puff));
                if matches!(label, MouthLabel::Smile) {
                    puff_on_smile_n += 1;
                    if puff {
                        puff_on_smile_fp += 1;
                    }
                }
                last_after(&mut ext, &identity, 10);
            }
        }

        let smile_puff_fpr = puff_on_smile_fp as f32 / puff_on_smile_n.max(1) as f32;
        eprintln!(
            "mouth feature accuracy (ids={N_ID}, th={DETECT_TH}):\n  \
             pucker recall={:.3} prec={:.3} fpr={:.3} acc={:.3}\n  \
             offset recall={:.3} prec={:.3} fpr={:.3} acc={:.3}\n  \
             puff   recall={:.3} prec={:.3} fpr={:.3} acc={:.3} smile_fpr={:.3}",
            pucker_c.recall(),
            pucker_c.precision(),
            pucker_c.fpr(),
            pucker_c.accuracy(),
            offset_c.recall(),
            offset_c.precision(),
            offset_c.fpr(),
            offset_c.accuracy(),
            puff_c.recall(),
            puff_c.precision(),
            puff_c.fpr(),
            puff_c.accuracy(),
            smile_puff_fpr
        );

        assert!(
            pucker_c.recall() >= 0.90 && pucker_c.fpr() <= 0.10 && pucker_c.accuracy() >= 0.90,
            "pucker recall={} fpr={} acc={}",
            pucker_c.recall(),
            pucker_c.fpr(),
            pucker_c.accuracy()
        );
        assert!(
            offset_c.recall() >= 0.90 && offset_c.fpr() <= 0.10 && offset_c.accuracy() >= 0.90,
            "offset recall={} fpr={} acc={}",
            offset_c.recall(),
            offset_c.fpr(),
            offset_c.accuracy()
        );
        assert!(
            puff_c.recall() >= 0.85 && puff_c.fpr() <= 0.15 && puff_c.accuracy() >= 0.85,
            "puff recall={} fpr={} acc={}",
            puff_c.recall(),
            puff_c.fpr(),
            puff_c.accuracy()
        );
        assert!(
            smile_puff_fpr <= 0.10,
            "cheek puff fired on smile fpr={smile_puff_fpr}"
        );
    }

    #[test]
    fn mouth_features_rest_stays_below_detect_threshold() {
        let rest = canonical_face();
        let mut ext = FeatureExtractor::new(0.0);
        calibrate_mouth(&mut ext, &rest);
        let mut seed = 7u32;
        let f = read_noisy(&mut ext, &rest, 20, &mut seed, 0.0035);
        let (pucker, offset, puff) = detect(&f);
        assert!(!pucker, "rest pucker {}", f[FEAT_MOUTH_PUCKER]);
        assert_eq!(offset, 0, "rest offset {}", f[FEAT_MOUTH_OFFSET_X]);
        assert!(!puff, "rest puff {}", f[FEAT_CHEEK_PUFF]);
        assert!(
            f[FEAT_JAW_OPEN] < DETECT_TH,
            "rest jaw {}",
            f[FEAT_JAW_OPEN]
        );
        assert!(
            f[FEAT_MOUTH_FUNNEL] < DETECT_TH,
            "rest funnel {}",
            f[FEAT_MOUTH_FUNNEL]
        );
        assert!(
            f[FEAT_MOUTH_PRESS_LIP_OPEN].abs() < DETECT_TH,
            "rest press-lip-open {}",
            f[FEAT_MOUTH_PRESS_LIP_OPEN]
        );
    }

    #[test]
    fn jaw_open_tracks_chin_not_lip_gap() {
        let rest = canonical_face();
        let mut ext = FeatureExtractor::new(0.0);
        calibrate_mouth(&mut ext, &rest);
        let mut seed = 11u32;

        let lip = read_noisy(&mut ext, &open_mouth(&rest, 3.0), 16, &mut seed, 0.0035);
        last_after(&mut ext, &rest, 10);
        let jaw = read_noisy(&mut ext, &drop_chin(&rest, 0.14), 16, &mut seed, 0.0035);

        assert!(
            lip[FEAT_JAW_OPEN] < DETECT_TH,
            "lip-only open fired jaw {}",
            lip[FEAT_JAW_OPEN]
        );
        assert!(
            jaw[FEAT_JAW_OPEN] > DETECT_TH,
            "chin drop missed jaw {}",
            jaw[FEAT_JAW_OPEN]
        );
    }

    #[test]
    fn mouth_press_lip_open_tracks_teeth_not_jaw() {
        let rest = canonical_face();
        let mut ext = FeatureExtractor::new(0.0);
        calibrate_mouth(&mut ext, &rest);
        let mut seed = 11u32;
        let slot = FEAT_MOUTH_PRESS_LIP_OPEN;
        let teeth = read_noisy(&mut ext, &bare_teeth(&rest, 0.05), 16, &mut seed, 0.0035);
        last_after(&mut ext, &rest, 10);
        let press = read_noisy(&mut ext, &press_lips(&rest, 0.7), 16, &mut seed, 0.0035);
        last_after(&mut ext, &rest, 10);
        let jaw = read_noisy(&mut ext, &drop_chin(&rest, 0.14), 16, &mut seed, 0.0035);
        last_after(&mut ext, &rest, 10);
        let funnel = read_noisy(
            &mut ext,
            &funnel_mouth(&rest, 0.55, 2.8),
            16,
            &mut seed,
            0.0035,
        );
        last_after(&mut ext, &rest, 10);
        let puck = read_noisy(
            &mut ext,
            &pucker_lips(&rest, 0.45, 0.10),
            16,
            &mut seed,
            0.0035,
        );
        last_after(&mut ext, &rest, 10);
        let smile = read_noisy(&mut ext, &smile_mouth(&rest, 1.6), 16, &mut seed, 0.0035);

        assert!(teeth[slot] > DETECT_TH, "teeth {}", teeth[slot]);
        assert!(press[slot] < -DETECT_TH, "press {}", press[slot]);
        assert!(jaw[slot] < DETECT_TH, "jaw {}", jaw[slot]);
        assert!(funnel[slot] < DETECT_TH, "funnel {}", funnel[slot]);
        assert!(puck[slot].abs() < DETECT_TH, "pucker {}", puck[slot]);
        assert!(smile[slot] > -DETECT_TH, "smile {}", smile[slot]);
    }

    #[test]
    fn mouth_funnel_fires_on_round_open_not_pucker_or_smile() {
        const N_ID: u32 = 24;
        const SIGMA: f32 = 0.0035;
        let rest0 = canonical_face();
        let mut funnel_c = Conf::default();
        let mut funnel_on_pucker = 0u32;
        let mut funnel_on_smile = 0u32;
        let mut funnel_on_open = 0u32;
        let mut n_ctrl = 0u32;
        let mut seed: u32 = 0xF00D;

        for _ in 0..N_ID {
            let identity = jitter(&rest0, &mut seed, 0.006);
            let mut ext = FeatureExtractor::new(0.0);
            calibrate_mouth(&mut ext, &identity);
            let intensity = 0.85 + 0.15 * lcg(&mut seed);

            let funnel_pose = lerp_pts(&identity, &funnel_mouth(&identity, 0.55, 2.8), intensity);
            let f = read_noisy(&mut ext, &funnel_pose, 16, &mut seed, SIGMA);
            funnel_c.add(f[FEAT_MOUTH_FUNNEL] > DETECT_TH, true);
            last_after(&mut ext, &identity, 10);

            let f = read_noisy(
                &mut ext,
                &lerp_pts(&identity, &pucker_lips(&identity, 0.45, 0.10), intensity),
                16,
                &mut seed,
                SIGMA,
            );
            n_ctrl += 1;
            if f[FEAT_MOUTH_FUNNEL] > DETECT_TH {
                funnel_on_pucker += 1;
            }
            funnel_c.add(f[FEAT_MOUTH_FUNNEL] > DETECT_TH, false);
            last_after(&mut ext, &identity, 10);

            let f = read_noisy(
                &mut ext,
                &lerp_pts(&identity, &smile_mouth(&identity, 1.6), intensity),
                16,
                &mut seed,
                SIGMA,
            );
            n_ctrl += 1;
            if f[FEAT_MOUTH_FUNNEL] > DETECT_TH {
                funnel_on_smile += 1;
            }
            funnel_c.add(f[FEAT_MOUTH_FUNNEL] > DETECT_TH, false);
            last_after(&mut ext, &identity, 10);

            let f = read_noisy(
                &mut ext,
                &lerp_pts(&identity, &open_mouth(&identity, 3.0), intensity),
                16,
                &mut seed,
                SIGMA,
            );
            n_ctrl += 1;
            if f[FEAT_MOUTH_FUNNEL] > DETECT_TH {
                funnel_on_open += 1;
            }
            funnel_c.add(f[FEAT_MOUTH_FUNNEL] > DETECT_TH, false);
            last_after(&mut ext, &identity, 10);
        }

        let pucker_fpr = funnel_on_pucker as f32 / N_ID as f32;
        let smile_fpr = funnel_on_smile as f32 / N_ID as f32;
        let open_fpr = funnel_on_open as f32 / N_ID as f32;
        eprintln!(
            "funnel recall={:.3} prec={:.3} fpr={:.3} acc={:.3} \
             pucker_fpr={pucker_fpr:.3} smile_fpr={smile_fpr:.3} open_fpr={open_fpr:.3} n_ctrl={n_ctrl}",
            funnel_c.recall(),
            funnel_c.precision(),
            funnel_c.fpr(),
            funnel_c.accuracy()
        );
        assert!(
            funnel_c.recall() >= 0.85 && funnel_c.fpr() <= 0.15 && funnel_c.accuracy() >= 0.85,
            "funnel recall={} fpr={} acc={}",
            funnel_c.recall(),
            funnel_c.fpr(),
            funnel_c.accuracy()
        );
        assert!(pucker_fpr <= 0.15, "funnel on pucker fpr={pucker_fpr}");
        assert!(smile_fpr <= 0.10, "funnel on smile fpr={smile_fpr}");
        assert!(open_fpr <= 0.15, "funnel on lip-open fpr={open_fpr}");
    }

    fn series_mean_std(
        ext: &mut FeatureExtractor,
        pose: &[[f32; 3]],
        slot: usize,
        seed: &mut u32,
    ) -> (f32, f32) {
        let mut vals = Vec::new();
        for i in 0..40 {
            let f = ext.update(&jitter(pose, seed, 0.0035), true);
            if i >= 8 {
                vals.push(f[slot]);
            }
        }
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn jaw_and_funnel_hold_stable() {
        let rest = canonical_face();
        let mut ext = FeatureExtractor::new(0.0);
        calibrate_mouth(&mut ext, &rest);
        let mut seed = 0x51ABu32;
        let (rj, sj) = series_mean_std(&mut ext, &rest, FEAT_JAW_OPEN, &mut seed);
        let (rf, sf) = series_mean_std(&mut ext, &rest, FEAT_MOUTH_FUNNEL, &mut seed);
        last_after(&mut ext, &rest, 8);
        let (hj, hjs) =
            series_mean_std(&mut ext, &drop_chin(&rest, 0.14), FEAT_JAW_OPEN, &mut seed);
        last_after(&mut ext, &rest, 8);
        let (hf, hfs) = series_mean_std(
            &mut ext,
            &funnel_mouth(&rest, 0.55, 2.8),
            FEAT_MOUTH_FUNNEL,
            &mut seed,
        );
        last_after(&mut ext, &rest, 8);
        let lip = read_noisy(&mut ext, &open_mouth(&rest, 3.0), 16, &mut seed, 0.0035);
        last_after(&mut ext, &rest, 8);
        let puck = read_noisy(
            &mut ext,
            &pucker_lips(&rest, 0.45, 0.10),
            16,
            &mut seed,
            0.0035,
        );
        last_after(&mut ext, &rest, 8);
        let smile = read_noisy(&mut ext, &smile_mouth(&rest, 1.6), 16, &mut seed, 0.0035);
        assert!(rj < DETECT_TH && sj < 0.08, "rest jaw {rj} std {sj}");
        assert!(rf < DETECT_TH && sf < 0.08, "rest funnel {rf} std {sf}");
        assert!(hj > 0.5 && hjs < 0.15, "hold jaw {hj} std {hjs}");
        assert!(hf > 0.5 && hfs < 0.15, "hold funnel {hf} std {hfs}");
        assert!(lip[FEAT_JAW_OPEN] < DETECT_TH && lip[FEAT_MOUTH_FUNNEL] < DETECT_TH);
        assert!(puck[FEAT_MOUTH_FUNNEL] < DETECT_TH && smile[FEAT_MOUTH_FUNNEL] < DETECT_TH);
        assert!(puck[FEAT_MOUTH_PRESS_LIP_OPEN].abs() < DETECT_TH);
        assert!(smile[FEAT_MOUTH_PRESS_LIP_OPEN] > -DETECT_TH);
    }
}
