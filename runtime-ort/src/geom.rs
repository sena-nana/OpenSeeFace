//! Geometry helpers matching `tracker.py`.

pub fn clamp_to_im(x: f32, y: f32, w: f32, h: f32) -> (i32, i32) {
    let x = x.clamp(0.0, (w - 1.0).max(0.0));
    let y = y.clamp(0.0, (h - 1.0).max(0.0));
    (x as i32, y as i32 + 1)
}

pub fn rotate(origin: (f32, f32), point: (f32, f32), a: f32) -> (f32, f32) {
    let a = -a;
    let (ox, oy) = origin;
    let (px, py) = point;
    let (c, s) = (a.cos(), a.sin());
    (
        ox + c * (px - ox) - s * (py - oy),
        oy + s * (px - ox) + c * (py - oy),
    )
}

pub fn angle(p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    dy.atan2(dx).rem_euclid(2.0 * std::f32::consts::PI)
}

pub fn compensate(p1: [f32; 2], p2: [f32; 2]) -> ((f32, f32), f32) {
    let a = angle(p1, p2);
    (rotate((p1[0], p1[1]), (p2[0], p2[1]), a), a)
}

pub fn intersects(r1: [f32; 4], r2: [f32; 4], amount: f32) -> bool {
    let area1 = r1[2] * r1[3];
    let area2 = r2[2] * r2[3];
    let r1_x2 = r1[0] + r1[2];
    let r1_y2 = r1[1] + r1[3];
    let r2_x2 = r2[0] + r2[2];
    let r2_y2 = r2[1] + r2[3];
    let left = r1[0].max(r2[0]);
    let right = r1_x2.min(r2_x2);
    let top = r1[1].max(r2[1]);
    let bottom = r1_y2.min(r2_y2);
    let mut inter = 0.0;
    let mut total = area1 + area2;
    if left < right && top < bottom {
        inter = (right - left) * (bottom - top);
        total -= inter;
    }
    total > 0.0 && inter / total >= amount
}

pub fn group_rects(rects: &[[f32; 4]]) -> Vec<usize> {
    let n = rects.len();
    let mut group = vec![0usize; n];
    for i in 0..n {
        group[i] = i;
    }
    for i in 0..n {
        for j in 0..n {
            if i != j && intersects(rects[i], rects[j], 0.3) {
                let gi = group[i];
                let gj = group[j];
                let (lo, hi) = if gi < gj { (gi, gj) } else { (gj, gi) };
                for g in group.iter_mut() {
                    if *g == hi {
                        *g = lo;
                    }
                }
            }
        }
    }
    group
}

pub fn logit(p: f32, factor: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln() / factor
}

pub fn xywh_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax2, ay2) = (a[0] + a[2], a[1] + a[3]);
    let (bx2, by2) = (b[0] + b[2], b[1] + b[3]);
    let l = a[0].max(b[0]);
    let t = a[1].max(b[1]);
    let r = ax2.min(bx2);
    let bot = ay2.min(by2);
    if r <= l || bot <= t {
        return 0.0;
    }
    let inter = (r - l) * (bot - t);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

pub fn matrix_to_quaternion(m: &nalgebra::Matrix3<f32>) -> [f32; 4] {
    let (m00, m01, m02) = (m[(0, 0)], m[(0, 1)], m[(0, 2)]);
    let (m10, m11, m12) = (m[(1, 0)], m[(1, 1)], m[(1, 2)]);
    let (m20, m21, m22) = (m[(2, 0)], m[(2, 1)], m[(2, 2)]);
    let (t, q) = if m22 < 0.0 {
        if m00 > m11 {
            let t = 1.0 + m00 - m11 - m22;
            (t, [t, m01 + m10, m20 + m02, m12 - m21])
        } else {
            let t = 1.0 - m00 + m11 - m22;
            (t, [m01 + m10, t, m12 + m21, m20 - m02])
        }
    } else if m00 < -m11 {
        let t = 1.0 - m00 - m11 + m22;
        (t, [m20 + m02, m12 + m21, t, m01 - m10])
    } else {
        let t = 1.0 + m00 + m11 + m22;
        (t, [m12 - m21, m20 - m02, m01 - m10, t])
    };
    let s = 0.5 / t.sqrt();
    [q[0] * s, q[1] * s, q[2] * s, q[3] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xywh_iou_identical_is_one() {
        let b = [10.0, 20.0, 40.0, 50.0];
        assert!((xywh_iou(b, b) - 1.0).abs() < 1e-5);
        assert_eq!(xywh_iou(b, [100.0, 100.0, 10.0, 10.0]), 0.0);
    }
}
