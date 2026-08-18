//! Expression features + remedian, matching `tracker.py` / `remedian.py`.

use crate::geom::{angle, rotate};

pub const FEATURE_NAMES: [&str; 14] = [
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
            && (self.max_feature_updates == 0.0 || now - self.first_seen < self.max_feature_updates);
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

    pub fn update(&mut self, pts: &[[f32; 3]], full: bool) -> [f32; 14] {
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
            let (a, f_pts) =
                Self::align_points(p(22), p(26), &[p(22), p(23), p(24), p(25), p(26)]);
            let steep_l = self
                .eyebrow_steepness_l
                .update(-a.to_degrees() - norm_angle, now);
            let f = f_pts[1..4]
                .iter()
                .map(|q| (q[1] - f_pts[0][1]).abs())
                .fold(0.0f32, f32::max)
                / ny;
            let quirk_l = self.eyebrow_quirk_l.update(f, now);

            let (a, f_pts) =
                Self::align_points(p(17), p(21), &[p(17), p(18), p(19), p(20), p(21)]);
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

        let f = ((pts[59][1] + pts[60][1] + pts[61][1]) / 3.0
            - (pts[63][1] + pts[64][1] + pts[65][1]) / 3.0)
            .abs()
            / ny;
        let mouth_open = self.mouth_open.update(f, now);
        let mouth_wide = self
            .mouth_wide
            .update((pts[58][0] - pts[62][0]).abs() / nx, now);

        [
            eye_l, eye_r, steep_l, up_l, quirk_l, steep_r, up_r, quirk_r, m_ud_l, m_io_l, m_ud_r,
            m_io_r, mouth_open, mouth_wide,
        ]
    }
}
