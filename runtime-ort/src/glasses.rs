//! Eye-region robustness: glare, pupil hold, and synthetic rims for benches.
//!
//! Crop, blink, and gaze key off local distrust (fit disagreement, EAR
//! outliers, heatmap conf, glare fraction). There is no glasses classifier.

use crate::preprocess::BgrImage;

const GAZE_MIN_CONF: f32 = 0.18;
const RIGHT_EYE: [usize; 6] = [36, 37, 38, 39, 40, 41];
const LEFT_EYE: [usize; 6] = [42, 43, 44, 45, 46, 47];

/// Keep the previous pupil when heatmap confidence collapses.
pub fn hold_gaze(prev: [[f32; 4]; 2], mut next: [[f32; 4]; 2]) -> [[f32; 4]; 2] {
    for i in 0..2 {
        if next[i][3] < GAZE_MIN_CONF && prev[i][3] >= GAZE_MIN_CONF {
            next[i][1] = prev[i][1];
            next[i][2] = prev[i][2];
            next[i][3] = prev[i][3];
        }
    }
    next
}

pub const GLARE_FRAC_THRESH: f32 = 0.015;

/// Fraction of pixels with luma > 235 (specular / blown highlights).
pub fn glare_frac(im: &BgrImage) -> f32 {
    if im.data.len() < 3 {
        return 0.0;
    }
    let mut n = 0.0;
    let mut hot = 0.0;
    for px in im.data.chunks_exact(3) {
        n += 1.0;
        if luma([px[0], px[1], px[2]]) > 235.0 {
            hot += 1.0;
        }
    }
    if n <= 0.0 {
        0.0
    } else {
        hot / n
    }
}

pub fn region_glare_frac(im: &BgrImage, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let xa = x1.max(0.0).floor() as i32;
    let ya = y1.max(0.0).floor() as i32;
    let xb = x2.min(im.width as f32 - 1.0).ceil() as i32;
    let yb = y2.min(im.height as f32 - 1.0).ceil() as i32;
    if xb <= xa || yb <= ya {
        return 0.0;
    }
    let mut n = 0.0;
    let mut hot = 0.0;
    for y in ya..=yb {
        for x in xa..=xb {
            n += 1.0;
            if luma(im.get(x, y)) > 235.0 {
                hot += 1.0;
            }
        }
    }
    if n <= 0.0 {
        0.0
    } else {
        hot / n
    }
}

fn luma(c: [u8; 3]) -> f32 {
    0.114 * c[0] as f32 + 0.587 * c[1] as f32 + 0.299 * c[2] as f32
}

struct EyeGeom {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
}

fn eye_geom(lms: &[[f32; 3]], idx: &[usize; 6]) -> Option<EyeGeom> {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut n = 0.0;
    for &i in idx {
        let p = lms.get(i)?;
        if p[2] < 0.15 {
            continue;
        }
        cx += p[1];
        cy += p[0];
        n += 1.0;
    }
    if n < 3.0 {
        return None;
    }
    cx /= n;
    cy /= n;
    let mut rx: f32 = 0.0;
    let mut ry: f32 = 0.0;
    for &i in idx {
        let p = lms.get(i)?;
        if p[2] < 0.15 {
            continue;
        }
        rx = rx.max((p[1] - cx).abs());
        ry = ry.max((p[0] - cy).abs());
    }
    rx = rx.max(3.0);
    ry = ry.max(2.0);
    Some(EyeGeom { cx, cy, rx, ry })
}

/// Clamp the brightest pixels in an eye crop toward the 70th percentile.
pub fn suppress_glare(im: &mut BgrImage) {
    if im.data.len() < 12 {
        return;
    }
    let mut ys: Vec<u8> = im
        .data
        .chunks_exact(3)
        .map(|p| luma([p[0], p[1], p[2]]) as u8)
        .collect();
    if ys.is_empty() {
        return;
    }
    ys.sort_unstable();
    let p70 = ys[ys.len() * 7 / 10] as f32;
    let cap = (p70 + 18.0).min(220.0);
    for px in im.data.chunks_exact_mut(3) {
        let l = luma([px[0], px[1], px[2]]);
        if l <= cap {
            continue;
        }
        let s = (cap / l.max(1.0)).clamp(0.35, 1.0);
        for c in px.iter_mut() {
            *c = (*c as f32 * s).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Dark oval rims + a specular blob on each eye (offline bench / tests).
pub fn paint_synthetic_glasses(frame: &mut BgrImage, lms: &[[f32; 3]]) {
    for idx in [&RIGHT_EYE, &LEFT_EYE] {
        let Some(e) = eye_geom(lms, idx) else {
            continue;
        };
        let rx = e.rx * 1.25;
        let ry = e.ry * 1.45;
        let x1 = (e.cx - rx - 3.0).floor().max(0.0) as i32;
        let y1 = (e.cy - ry - 3.0).floor().max(0.0) as i32;
        let x2 = (e.cx + rx + 3.0).ceil().min(frame.width as f32 - 1.0) as i32;
        let y2 = (e.cy + ry + 3.0).ceil().min(frame.height as f32 - 1.0) as i32;
        for y in y1..=y2 {
            for x in x1..=x2 {
                let dx = (x as f32 - e.cx) / rx.max(1.0);
                let dy = (y as f32 - e.cy) / ry.max(1.0);
                let r2 = dx * dx + dy * dy;
                if r2 > 1.05 || r2 < 0.62 {
                    continue;
                }
                frame.set(x, y, [28, 24, 22]);
            }
        }
        let bx = e.cx + e.rx * 0.18;
        let by = e.cy - e.ry * 0.12;
        for y in (by - 2.5) as i32..=(by + 2.5) as i32 {
            for x in (bx - 3.5) as i32..=(bx + 3.5) as i32 {
                let dx = x as f32 - bx;
                let dy = y as f32 - by;
                if dx * dx / 12.0 + dy * dy / 6.0 <= 1.0 {
                    frame.set(x, y, [255, 252, 248]);
                }
            }
        }
    }
}

/// Vertical lid gap / eye width from 2D landmarks (row, col).
pub fn ear_2d(lms: &[[f32; 3]]) -> Option<f32> {
    if lms.len() < 48 {
        return None;
    }
    let right = lid_gap(lms, 36, 39, [37, 38, 41, 40]);
    let left = lid_gap(lms, 42, 45, [43, 44, 47, 46]);
    match (right, left) {
        (Some(a), Some(b)) => Some(0.5 * (a + b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        _ => None,
    }
}

fn lid_gap(lms: &[[f32; 3]], outer: usize, inner: usize, lids: [usize; 4]) -> Option<f32> {
    let o = lms.get(outer)?;
    let i = lms.get(inner)?;
    let w = (i[1] - o[1]).hypot(i[0] - o[0]).max(1e-6);
    let p0 = lms.get(lids[0])?;
    let p1 = lms.get(lids[1])?;
    let p2 = lms.get(lids[2])?;
    let p3 = lms.get(lids[3])?;
    let gap = ((p0[0] + p1[0]) * 0.5 - (p2[0] + p3[0]) * 0.5).abs();
    Some(gap / w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma_px(c: [u8; 3]) -> f32 {
        luma(c)
    }

    #[test]
    fn suppress_glare_lowers_peak() {
        let mut im = BgrImage {
            width: 8,
            height: 8,
            data: vec![80u8, 80, 80].repeat(64),
        };
        im.set(3, 3, [255, 255, 255]);
        im.set(4, 3, [255, 250, 250]);
        let before = luma_px(im.get(3, 3));
        suppress_glare(&mut im);
        let after = luma_px(im.get(3, 3));
        assert!(after < before - 10.0, "{before} -> {after}");
    }

    #[test]
    fn hold_gaze_keeps_prev_on_low_conf() {
        let prev = [[1.0, 10.0, 20.0, 0.5], [1.0, 11.0, 21.0, 0.6]];
        let next = [[1.0, 99.0, 99.0, 0.05], [1.0, 12.0, 22.0, 0.4]];
        let got = hold_gaze(prev, next);
        assert!((got[0][1] - 10.0).abs() < 1e-4);
        assert!((got[1][1] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn glare_frac_counts_hot_pixels() {
        let mut im = BgrImage {
            width: 4,
            height: 4,
            data: vec![40u8, 40, 40].repeat(16),
        };
        im.set(1, 1, [255, 255, 255]);
        assert!(glare_frac(&im) > 0.0);
        assert!(glare_frac(&im) < 0.2);
    }
}
