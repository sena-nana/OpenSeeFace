//! Expression features + remedian, matching upstream `tracker.py` / `remedian.py`.

use crate::geom::{angle, rotate};

pub const FEATURE_COUNT: usize = 14;
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
];

pub const FEAT_EYE_L: usize = 0;
pub const FEAT_EYE_R: usize = 1;
pub const FEAT_BROW_STEEP_L: usize = 2;
pub const FEAT_BROW_UD_L: usize = 3;
pub const FEAT_BROW_STEEP_R: usize = 5;
pub const FEAT_BROW_UD_R: usize = 6;
pub const FEAT_CORNER_UD_L: usize = 8;
pub const FEAT_CORNER_IO_L: usize = 9;
pub const FEAT_CORNER_UD_R: usize = 10;
pub const FEAT_CORNER_IO_R: usize = 11;
pub const FEAT_MOUTH_OPEN: usize = 12;
pub const FEAT_MOUTH_WIDE: usize = 13;

/// Face-width (`nx`) and nose-bridge (`ny`) scale from 3D landmarks.
pub fn face_norm_basis(pts: &[[f32; 3]]) -> (f32, f32) {
    let nx = ((pts[0][0] - pts[16][0]) + (pts[1][0] - pts[15][0])) / 2.0;
    let ny =
        ((pts[27][1] - pts[28][1]) + (pts[28][1] - pts[29][1]) + (pts[29][1] - pts[30][1])) / 3.0;
    (nx.abs().max(1e-6), ny.abs().max(1e-6))
}

struct Remedian {
    k: usize,
    all: Vec<f32>,
    more: Option<Box<Remedian>>,
}

impl Remedian {
    fn new() -> Self {
        Self {
            k: 64,
            all: Vec::new(),
            more: None,
        }
    }

    fn add(&mut self, x: f32) {
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
    if n.is_multiple_of(2) {
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
}

impl Feature {
    fn new(threshold: f32, max_feature_updates: f32) -> Self {
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
}

pub struct FeatureExtractor {
    eye_l: Feature,
    eye_r: Feature,
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
}

impl FeatureExtractor {
    pub fn new(max_feature_updates: f32) -> Self {
        Self {
            eye_l: Feature::new(0.15, max_feature_updates),
            eye_r: Feature::new(0.15, max_feature_updates),
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f32())
            .unwrap_or(0.0);

        let p = |i: usize| [pts[i][0], pts[i][1]];
        let (nx, ny) = face_norm_basis(pts);

        let (a1, f_pts) = Self::align_points(p(42), p(45), &[p(43), p(44), p(47), p(46)]);
        let f = ((f_pts[0][1] + f_pts[1][1]) / 2.0 - (f_pts[2][1] + f_pts[3][1]) / 2.0).abs() / ny;
        let eye_l = self.eye_l.update(f, now);

        let (a2, f_pts) = Self::align_points(p(36), p(39), &[p(37), p(38), p(41), p(40)]);
        let f = ((f_pts[0][1] + f_pts[1][1]) / 2.0 - (f_pts[2][1] + f_pts[3][1]) / 2.0).abs() / ny;
        let eye_r = self.eye_r.update(f, now);

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

        [
            eye_l, eye_r, steep_l, up_l, quirk_l, steep_r, up_r, quirk_r, m_ud_l, m_io_l, m_ud_r,
            m_io_r, mouth_open, mouth_wide,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_face() -> Vec<[f32; 3]> {
        let mut p = vec![[0.0f32; 3]; 66];
        for (i, slot) in p.iter_mut().enumerate().take(17) {
            let t = i as f32 / 16.0;
            let x = 0.45 * (1.0 - 2.0 * t);
            let y = 0.30 - (std::f32::consts::PI * t).sin() * 0.90;
            *slot = [x, y, -0.72];
        }
        p[27] = [0.0, 0.29, -0.14];
        p[28] = [0.0, 0.19, -0.07];
        p[29] = [0.0, 0.10, -0.01];
        p[30] = [0.0, 0.00, 0.00];
        p[31] = [0.08, -0.02, -0.08];
        p[35] = [-0.08, -0.02, -0.08];
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
        p[17] = [0.36, 0.42, -0.25];
        p[18] = [0.30, 0.45, -0.24];
        p[19] = [0.22, 0.46, -0.23];
        p[20] = [0.15, 0.45, -0.22];
        p[21] = [0.09, 0.42, -0.20];
        p[22] = [-0.09, 0.42, -0.20];
        p[23] = [-0.15, 0.45, -0.22];
        p[24] = [-0.22, 0.46, -0.23];
        p[25] = [-0.30, 0.45, -0.24];
        p[26] = [-0.36, 0.42, -0.25];
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

    #[test]
    fn remedian_tracks_running_median() {
        let mut r = Remedian::new();
        for i in 1..=63 {
            r.add(i as f32);
        }
        assert!((r.median() - 32.0).abs() < 1e-5);
        r.add(64.0);
        // Once 64 items roll up into the parent Remedian, its sole median is 32.5.
        assert!((r.median() - 32.5).abs() < 1e-5);
    }

    #[test]
    fn feature_blinks_and_recovers_with_upstream_threshold() {
        let mut f = Feature::new(0.15, 0.0);
        for i in 0..80 {
            f.update(1.0, i as f32 * 0.03);
        }
        // Small variation below the 15% threshold stays at rest (0.0).
        assert_eq!(f.update(0.90, 80.0 * 0.03), 0.0);
        // Once min and max have been seen, closing toward 0.10 holds near -1.0 and recovers.
        for i in 0..2 {
            f.update(0.80, (81 + i) as f32 * 0.03);
        }
        for i in 0..2 {
            f.update(1.25, (83 + i) as f32 * 0.03);
        }
        let mut shut = 0.0;
        for i in 0..12 {
            shut = f.update(0.10, (85 + i) as f32 * 0.03);
        }
        assert!(shut < -0.85, "shut {shut}");
        let mut open = shut;
        for i in 0..20 {
            open = f.update(1.0, (97 + i) as f32 * 0.03);
        }
        assert!(open.abs() < 0.05, "recover {open}");
    }

    #[test]
    fn extractor_outputs_14_features_and_respects_full_flag() {
        let rest = canonical_face();
        let mut ext = FeatureExtractor::new(0.0);
        for _ in 0..70 {
            let f = ext.update(&rest, true);
            assert_eq!(f.len(), FEATURE_COUNT);
        }
        // Seed min then max on the left eye before holding it shut.
        let mut squint = rest.clone();
        squint[43][1] -= 0.025;
        squint[44][1] -= 0.025;
        ext.update(&squint, true);
        let mut wide = rest.clone();
        wide[43][1] += 0.03;
        wide[44][1] += 0.03;
        ext.update(&wide, true);

        // Dropping the upper lids on the left eye closes eye_l toward -1.
        let mut closed = rest.clone();
        closed[43][1] = closed[47][1];
        closed[44][1] = closed[46][1];
        let mut f_closed = [0.0; FEATURE_COUNT];
        for _ in 0..12 {
            f_closed = ext.update(&closed, false);
        }
        assert!(
            f_closed[FEAT_EYE_L] < -0.8,
            "eye_l {}",
            f_closed[FEAT_EYE_L]
        );
        assert!(
            f_closed[FEAT_EYE_R].abs() < 0.1,
            "eye_r {}",
            f_closed[FEAT_EYE_R]
        );
        assert_eq!(f_closed[FEAT_BROW_STEEP_L], 0.0);
        assert_eq!(f_closed[FEAT_CORNER_IO_L], 0.0);
    }
}
