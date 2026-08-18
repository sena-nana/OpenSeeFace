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

/// 2D similarity `dst ≈ s R src + t` (Umeyama, with scale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub scale: f32,
    pub theta: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Similarity {
    pub fn apply(self, p: [f32; 2]) -> [f32; 2] {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        [
            self.scale * (c * p[0] - s * p[1]) + self.tx,
            self.scale * (s * p[0] + c * p[1]) + self.ty,
        ]
    }

    pub fn aabb(self, pts: impl IntoIterator<Item = [f32; 2]>) -> Option<[f32; 4]> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut n = 0usize;
        for p in pts {
            let q = self.apply(p);
            min_x = min_x.min(q[0]);
            min_y = min_y.min(q[1]);
            max_x = max_x.max(q[0]);
            max_y = max_y.max(q[1]);
            n += 1;
        }
        (n > 0).then_some([
            min_x,
            min_y,
            (max_x - min_x).max(1.0),
            (max_y - min_y).max(1.0),
        ])
    }
}

/// Umeyama similarity from `src` to `dst`. Needs at least two point pairs.
pub fn similarity_umeyama(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Option<Similarity> {
    let n = src.len().min(dst.len());
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mut src_mean = [0.0f64; 2];
    let mut dst_mean = [0.0f64; 2];
    for i in 0..n {
        src_mean[0] += src[i][0] as f64;
        src_mean[1] += src[i][1] as f64;
        dst_mean[0] += dst[i][0] as f64;
        dst_mean[1] += dst[i][1] as f64;
    }
    src_mean[0] /= nf;
    src_mean[1] /= nf;
    dst_mean[0] /= nf;
    dst_mean[1] /= nf;

    let mut a00 = 0.0;
    let mut a01 = 0.0;
    let mut a10 = 0.0;
    let mut a11 = 0.0;
    let mut src_var = 0.0;
    for i in 0..n {
        let sx = src[i][0] as f64 - src_mean[0];
        let sy = src[i][1] as f64 - src_mean[1];
        let dx = dst[i][0] as f64 - dst_mean[0];
        let dy = dst[i][1] as f64 - dst_mean[1];
        a00 += dx * sx;
        a01 += dx * sy;
        a10 += dy * sx;
        a11 += dy * sy;
        src_var += sx * sx + sy * sy;
    }
    a00 /= nf;
    a01 /= nf;
    a10 /= nf;
    a11 /= nf;
    src_var /= nf;
    if src_var < 1e-12 {
        return None;
    }

    let a = nalgebra::Matrix2::new(a00, a01, a10, a11);
    let svd = nalgebra::SVD::new(a, true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let sv = svd.singular_values;
    let d0 = 1.0f64;
    let mut d1 = 1.0f64;
    if a.determinant() < 0.0 {
        d1 = -1.0;
    }
    let det_u = u.determinant();
    let det_v = v_t.transpose().determinant();
    if (a.rank(1e-12) == 1) && det_u * det_v < 0.0 {
        d1 = -1.0;
    }
    let d = nalgebra::Matrix2::new(d0, 0.0, 0.0, d1);
    let r = u * d * v_t;
    let scale = (sv[0] * d0 + sv[1] * d1) / src_var;
    if !scale.is_finite() || scale <= 1e-8 {
        return None;
    }
    let rsx = r[(0, 0)] * src_mean[0] + r[(0, 1)] * src_mean[1];
    let rsy = r[(1, 0)] * src_mean[0] + r[(1, 1)] * src_mean[1];
    let tx = dst_mean[0] - scale * rsx;
    let ty = dst_mean[1] - scale * rsy;
    let m00 = scale * r[(0, 0)];
    let m10 = scale * r[(1, 0)];
    let theta = m10.atan2(m00);
    Some(Similarity {
        scale: scale as f32,
        theta: theta as f32,
        tx: tx as f32,
        ty: ty as f32,
    })
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
    fn umeyama_identity() {
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let s = similarity_umeyama(&src, &src).unwrap();
        assert!((s.scale - 1.0).abs() < 1e-4, "{s:?}");
        assert!(s.theta.abs() < 1e-4, "{s:?}");
        assert!(s.tx.abs() < 1e-4 && s.ty.abs() < 1e-4, "{s:?}");
    }

    #[test]
    fn umeyama_scale_translate() {
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let dst: Vec<[f32; 2]> = src
            .iter()
            .map(|p| [2.0 * p[0] + 3.0, 2.0 * p[1] + 4.0])
            .collect();
        let s = similarity_umeyama(&src, &dst).unwrap();
        assert!((s.scale - 2.0).abs() < 1e-3, "{s:?}");
        assert!(s.theta.abs() < 1e-3, "{s:?}");
        assert!(
            (s.tx - 3.0).abs() < 1e-3 && (s.ty - 4.0).abs() < 1e-3,
            "{s:?}"
        );
        let q = s.apply([1.0, 1.0]);
        assert!((q[0] - 5.0).abs() < 1e-3 && (q[1] - 6.0).abs() < 1e-3);
    }

    #[test]
    fn umeyama_rotation_90() {
        let src = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let dst: Vec<[f32; 2]> = src.iter().map(|p| [-p[1], p[0]]).collect();
        let s = similarity_umeyama(&src, &dst).unwrap();
        assert!((s.scale - 1.0).abs() < 1e-3, "{s:?}");
        assert!(
            (s.theta - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "{s:?}"
        );
        assert!(s.tx.abs() < 1e-3 && s.ty.abs() < 1e-3, "{s:?}");
    }

    #[test]
    fn xywh_iou_identical_is_one() {
        let b = [10.0, 20.0, 40.0, 50.0];
        assert!((xywh_iou(b, b) - 1.0).abs() < 1e-5);
        assert_eq!(xywh_iou(b, [100.0, 100.0, 10.0, 10.0]), 0.0);
    }
}
