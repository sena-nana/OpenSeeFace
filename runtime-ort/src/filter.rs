//! One Euro post-process on UDP pose and 2D landmarks.
//!
//! Crop and PnP keep raw measurements. Expression features are not filtered
//! here (`features.rs` already uses EMA). See `benchmarks/filter-eval.md`.

use std::f32::consts::PI;
use std::str::FromStr;

use nalgebra::Matrix3;

use crate::geom::matrix_to_quaternion;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterKind {
    #[default]
    None,
    OneEuro,
}

impl FilterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OneEuro => "one-euro",
        }
    }
}

impl FromStr for FilterKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "none" | "off" | "0" => Self::None,
            "one-euro" | "oneeuro" | "1e" | "euro" => Self::OneEuro,
            other => anyhow::bail!("unknown --filter {other} (none|one-euro)"),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FilterCfg {
    pub kind: FilterKind,
    pub mincutoff: f32,
    pub beta: f32,
    pub dcutoff: f32,
}

impl Default for FilterCfg {
    fn default() -> Self {
        Self {
            kind: FilterKind::None,
            mincutoff: 1.0,
            beta: 0.007,
            dcutoff: 1.0,
        }
    }
}

impl FilterCfg {
    pub fn new(kind: FilterKind, mincutoff: f32, beta: f32) -> Self {
        Self {
            kind,
            mincutoff,
            beta,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug)]
struct OneEuro {
    xhat: Option<f32>,
    dxhat: f32,
    xprev: Option<f32>,
}

impl OneEuro {
    fn new() -> Self {
        Self {
            xhat: None,
            dxhat: 0.0,
            xprev: None,
        }
    }

    fn filter(&mut self, x: f32, dt: f32, cfg: &FilterCfg) -> f32 {
        let dt = dt.max(1e-6);
        let dx = match self.xprev {
            Some(p) => (x - p) / dt,
            None => 0.0,
        };
        let a_d = alpha(cfg.dcutoff, dt);
        self.dxhat = if self.xprev.is_none() {
            dx
        } else {
            a_d * dx + (1.0 - a_d) * self.dxhat
        };
        let cutoff = (cfg.mincutoff + cfg.beta * self.dxhat.abs()).max(1e-4);
        let a = alpha(cutoff, dt);
        let y = match self.xhat {
            Some(p) => a * x + (1.0 - a) * p,
            None => x,
        };
        self.xhat = Some(y);
        self.xprev = Some(x);
        y
    }
}

fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff.max(1e-4));
    1.0 / (1.0 + tau / dt.max(1e-6))
}

#[derive(Clone, Debug)]
pub struct OutputFilter {
    cfg: FilterCfg,
    trans: [OneEuro; 3],
    euler: [OneEuro; 3],
    lms: Vec<[OneEuro; 2]>,
    pts: Vec<[OneEuro; 3]>,
    unwrapped: Option<[f32; 3]>,
}

impl OutputFilter {
    pub fn new(cfg: FilterCfg) -> Self {
        Self {
            cfg,
            trans: [OneEuro::new(), OneEuro::new(), OneEuro::new()],
            euler: [OneEuro::new(), OneEuro::new(), OneEuro::new()],
            lms: Vec::new(),
            pts: Vec::new(),
            unwrapped: None,
        }
    }

    pub fn reset(&mut self) {
        self.trans = [OneEuro::new(), OneEuro::new(), OneEuro::new()];
        self.euler = [OneEuro::new(), OneEuro::new(), OneEuro::new()];
        self.lms.clear();
        self.pts.clear();
        self.unwrapped = None;
    }

    pub fn apply(
        &mut self,
        euler: &mut [f32; 3],
        translation: &mut [f32; 3],
        quaternion: &mut [f32; 4],
        lms: &mut [[f32; 3]],
        pts_3d: &mut [[f32; 3]],
        dt: f32,
    ) {
        if self.cfg.kind == FilterKind::None {
            return;
        }
        let dt = dt.clamp(1.0 / 240.0, 0.25);
        let raw_e = *euler;
        let unwrapped = match self.unwrapped {
            Some(p) => [
                unwrap_deg(p[0], raw_e[0]),
                unwrap_deg(p[1], raw_e[1]),
                unwrap_deg(p[2], raw_e[2]),
            ],
            None => raw_e,
        };
        let mut fe = [0.0; 3];
        for i in 0..3 {
            fe[i] = self.euler[i].filter(unwrapped[i], dt, &self.cfg);
            translation[i] = self.trans[i].filter(translation[i], dt, &self.cfg);
        }
        self.unwrapped = Some(fe);
        *euler = [wrap_deg(fe[0]), wrap_deg(fe[1]), wrap_deg(fe[2])];
        *quaternion = euler_to_quat(fe);

        while self.lms.len() < lms.len() {
            self.lms.push([OneEuro::new(), OneEuro::new()]);
        }
        for (i, p) in lms.iter_mut().enumerate() {
            p[0] = self.lms[i][0].filter(p[0], dt, &self.cfg);
            p[1] = self.lms[i][1].filter(p[1], dt, &self.cfg);
        }
        while self.pts.len() < pts_3d.len() {
            self.pts
                .push([OneEuro::new(), OneEuro::new(), OneEuro::new()]);
        }
        for (i, p) in pts_3d.iter_mut().enumerate() {
            p[0] = self.pts[i][0].filter(p[0], dt, &self.cfg);
            p[1] = self.pts[i][1].filter(p[1], dt, &self.cfg);
            p[2] = self.pts[i][2].filter(p[2], dt, &self.cfg);
        }
    }
}

pub fn unwrap_deg(prev: f32, meas: f32) -> f32 {
    let mut d = meas - prev;
    d -= 360.0 * (d / 360.0).round();
    prev + d
}

fn wrap_deg(a: f32) -> f32 {
    let mut x = (a + 180.0) % 360.0;
    if x < 0.0 {
        x += 360.0;
    }
    x - 180.0
}

fn euler_to_rmat(e: [f32; 3]) -> Matrix3<f32> {
    let (x, y, z) = (e[0].to_radians(), e[1].to_radians(), e[2].to_radians());
    let (cx, sx) = (x.cos(), x.sin());
    let (cy, sy) = (y.cos(), y.sin());
    let (cz, sz) = (z.cos(), z.sin());
    Matrix3::new(
        cy * cz,
        cz * sx * sy - cx * sz,
        sx * sz + cx * cz * sy,
        cy * sz,
        cx * cz + sx * sy * sz,
        cx * sy * sz - cz * sx,
        -sy,
        cy * sx,
        cx * cy,
    )
}

fn euler_to_quat(e: [f32; 3]) -> [f32; 4] {
    let q = matrix_to_quaternion(&euler_to_rmat(e));
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-8);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        let m = v.iter().sum::<f32>() / v.len() as f32;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
    }

    fn q_norm(q: [f32; 4]) -> f32 {
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
    }

    #[test]
    fn one_euro_damps_static_noise() {
        let mut f = OneEuro::new();
        let cfg = FilterCfg {
            kind: FilterKind::OneEuro,
            ..FilterCfg::default()
        };
        let dt = 1.0 / 30.0;
        let mut xin = Vec::new();
        let mut yout = Vec::new();
        for i in 0..180 {
            let x = 0.15 * ((i as f32) * 1.7).sin();
            xin.push(x);
            yout.push(f.filter(x, dt, &cfg));
        }
        assert!(
            var(&yout) < 0.45 * var(&xin),
            "out var {} vs in {}",
            var(&yout),
            var(&xin)
        );
    }

    #[test]
    fn one_euro_follows_step_in_three_frames() {
        let mut f = OneEuro::new();
        let cfg = FilterCfg {
            kind: FilterKind::OneEuro,
            beta: 1.0,
            mincutoff: 1.0,
            ..FilterCfg::default()
        };
        let dt = 1.0 / 30.0;
        for _ in 0..8 {
            f.filter(0.0, dt, &cfg);
        }
        let mut y = 0.0;
        for _ in 0..3 {
            y = f.filter(1.0, dt, &cfg);
        }
        assert!(y >= 0.85, "step y={y}");
    }

    #[test]
    fn euler_unwrap_does_not_jump() {
        let mut flt = OutputFilter::new(FilterCfg::new(FilterKind::OneEuro, 1.0, 0.05));
        let dt = 1.0 / 30.0;
        let seq = [170.0, 175.0, 179.0, -179.0, -175.0, -170.0];
        let mut prev: Option<f32> = None;
        for &m in &seq {
            let mut e = [m, 0.0, 0.0];
            let mut t = [0.0; 3];
            let mut q = [0.0, 0.0, 0.0, 1.0];
            let mut lms = vec![[0.0; 3]; 4];
            let mut pts = vec![[0.0; 3]; 4];
            flt.apply(&mut e, &mut t, &mut q, &mut lms, &mut pts, dt);
            let u = flt.unwrapped.unwrap()[0];
            if let Some(p) = prev {
                assert!((u - p).abs() < 20.0, "unwrap jump {p} -> {u} from meas {m}");
            }
            prev = Some(u);
            assert!((q_norm(q) - 1.0).abs() < 1e-4, "quat norm {}", q_norm(q));
        }
    }

    #[test]
    fn dt_change_stays_finite() {
        let mut f = OneEuro::new();
        let cfg = FilterCfg::new(FilterKind::OneEuro, 1.0, 0.007);
        for (i, dt) in [1.0 / 15.0, 1.0 / 30.0, 1.0 / 60.0, 1.0 / 24.0]
            .into_iter()
            .enumerate()
        {
            let y = f.filter(i as f32 * 0.1, dt, &cfg);
            assert!(y.is_finite());
        }
    }

    #[test]
    fn reset_first_frame_equals_measurement() {
        let mut flt = OutputFilter::new(FilterCfg::new(FilterKind::OneEuro, 1.0, 0.007));
        let dt = 1.0 / 30.0;
        let mut e = [3.0, 1.0, -2.0];
        let mut t = [10.0, 20.0, 30.0];
        let mut q = [0.0, 0.0, 0.0, 1.0];
        let mut lms = vec![[5.0, 6.0, 0.9]];
        let mut pts = vec![[1.0, 2.0, 3.0]];
        flt.apply(&mut e, &mut t, &mut q, &mut lms, &mut pts, dt);
        for _ in 0..5 {
            let mut e2 = [0.0, 0.0, 0.0];
            let mut t2 = [0.0; 3];
            let mut q2 = q;
            flt.apply(&mut e2, &mut t2, &mut q2, &mut lms, &mut pts, dt);
        }
        flt.reset();
        let mut e = [12.0, -4.0, 8.0];
        let mut t = [100.0, 50.0, 7.0];
        let mut q = [0.0, 0.0, 0.0, 1.0];
        let mut lms = vec![[9.0, 11.0, 0.8]];
        let mut pts = vec![[4.0, 5.0, 6.0]];
        flt.apply(&mut e, &mut t, &mut q, &mut lms, &mut pts, dt);
        assert!((t[0] - 100.0).abs() < 1e-4);
        assert!((lms[0][0] - 9.0).abs() < 1e-4);
        assert!((q_norm(q) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn none_is_identity() {
        let mut flt = OutputFilter::new(FilterCfg::default());
        let mut e = [11.0, -8.0, 3.0];
        let mut t = [1.0, 2.0, 3.0];
        let mut q = [0.1, 0.2, 0.3, 0.9];
        let mut lms = vec![[4.0, 5.0, 0.7]];
        let mut pts = vec![[6.0, 7.0, 8.0]];
        flt.apply(&mut e, &mut t, &mut q, &mut lms, &mut pts, 1.0 / 30.0);
        assert_eq!(e, [11.0, -8.0, 3.0]);
        assert_eq!(t, [1.0, 2.0, 3.0]);
        assert_eq!(lms[0], [4.0, 5.0, 0.7]);
    }

    #[test]
    fn parse_kinds() {
        assert_eq!(
            "one-euro".parse::<FilterKind>().unwrap(),
            FilterKind::OneEuro
        );
        assert_eq!("none".parse::<FilterKind>().unwrap(), FilterKind::None);
        assert!("ema".parse::<FilterKind>().is_err());
        assert!("kalman".parse::<FilterKind>().is_err());
    }
}
