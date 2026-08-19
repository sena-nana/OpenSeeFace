//! CPU SIMD dispatch for preprocess and enhance pixel loops.
//!
//! x86_64 uses runtime `is_x86_feature_detected!` so generic Windows binaries stay
//! baseline-SSE2. aarch64 always has NEON.

use half::f16;

mod scalar;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

pub(crate) const HIST: usize = 256;

#[inline]
pub(crate) fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

pub fn backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            "avx2+f16c"
        } else if is_x86_feature_detected!("ssse3") {
            "ssse3"
        } else {
            "sse2"
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "x86")))]
    {
        "scalar"
    }
}

fn run_nchw(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    dst_w: u32,
    dst_h: u32,
    src: [usize; 3],
    scale: [f32; 3],
    bias: [f32; 3],
    data: &mut [f16],
    lanes: u32,
    chunk: impl Fn(
        &[u8],
        u32,
        u32,
        u32,
        f32,
        u32,
        u32,
        f32,
        u32,
        u32,
        u32,
        [usize; 3],
        [f32; 3],
        [f32; 3],
        usize,
        &mut [f16],
    ),
) {
    let n = (dst_w * dst_h) as usize;
    let fx = cw as f32 / dst_w as f32;
    let fy = ch as f32 / dst_h as f32;
    let last_x = cw - 1;
    let last_y = ch - 1;
    for y in 0..dst_h {
        let (iy0, iy1, wy) = scalar::sample_y(y, fy, last_y);
        let gy0 = y0 + iy0;
        let gy1 = y0 + iy1;
        let mut x = 0u32;
        while x + lanes <= dst_w {
            chunk(
                src_bgr, src_w, x0, last_x, fx, gy0, gy1, wy, x, y, dst_w, src, scale, bias, n,
                data,
            );
            x += lanes;
        }
        while x < dst_w {
            scalar::nchw_pixel(
                src_bgr,
                src_w,
                x0,
                last_x,
                fx,
                gy0,
                gy1,
                wy,
                x,
                src,
                scale,
                bias,
                n,
                (y * dst_w + x) as usize,
                data,
            );
            x += 1;
        }
    }
}

fn run_resize(
    src: &[u8],
    width: u32,
    x1: u32,
    y1: u32,
    cw: u32,
    ch: u32,
    dw: u32,
    dh: u32,
    dst: &mut [u8],
    lanes: u32,
    chunk: impl Fn(&[u8], u32, u32, u32, f32, u32, u32, f32, u32, &mut [u8], usize),
) {
    let fx = cw as f32 / dw as f32;
    let fy = ch as f32 / dh as f32;
    let last_x = cw - 1;
    let last_y = ch - 1;
    for y in 0..dh {
        let (iy0, iy1, wy) = scalar::sample_y(y, fy, last_y);
        let gy0 = y1 + iy0;
        let gy1 = y1 + iy1;
        let mut x = 0u32;
        while x + lanes <= dw {
            chunk(
                src,
                width,
                x1,
                last_x,
                fx,
                gy0,
                gy1,
                wy,
                x,
                dst,
                ((y * dw + x) * 3) as usize,
            );
            x += lanes;
        }
        while x < dw {
            scalar::resize_pixel(
                src,
                width,
                x1,
                last_x,
                fx,
                gy0,
                gy1,
                wy,
                x,
                dst,
                ((y * dw + x) * 3) as usize,
            );
            x += 1;
        }
    }
}

pub fn bilinear_nchw(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    dst_w: u32,
    dst_h: u32,
    src: [usize; 3],
    scale: [f32; 3],
    bias: [f32; 3],
    data: &mut [f16],
) {
    #[cfg(target_arch = "aarch64")]
    run_nchw(
        src_bgr,
        src_w,
        x0,
        y0,
        cw,
        ch,
        dst_w,
        dst_h,
        src,
        scale,
        bias,
        data,
        neon::NCHW_LANES,
        |a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p| unsafe {
            neon::nchw_chunk4(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
        },
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            x86::bilinear_nchw_avx2(
                src_bgr, src_w, x0, y0, cw, ch, dst_w, dst_h, src, scale, bias, data,
            );
        } else {
            x86::bilinear_nchw_sse2(
                src_bgr, src_w, x0, y0, cw, ch, dst_w, dst_h, src, scale, bias, data,
            );
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "x86")))]
    scalar::bilinear_nchw(
        src_bgr, src_w, x0, y0, cw, ch, dst_w, dst_h, src, scale, bias, data,
    );
}

pub fn apply_affine_roi(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    src: [usize; 3],
    scale: [f32; 3],
    bias: [f32; 3],
    data: &mut [f16],
) {
    #[cfg(target_arch = "aarch64")]
    neon::apply_affine_roi(src_bgr, src_w, x0, y0, cw, ch, src, scale, bias, data);
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("ssse3") {
            x86::apply_affine_roi_ssse3(src_bgr, src_w, x0, y0, cw, ch, src, scale, bias, data);
        } else {
            scalar::apply_affine_roi(src_bgr, src_w, x0, y0, cw, ch, src, scale, bias, data);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "x86")))]
    scalar::apply_affine_roi(src_bgr, src_w, x0, y0, cw, ch, src, scale, bias, data);
}

pub fn resize_bilinear(
    src: &[u8],
    width: u32,
    x1: u32,
    y1: u32,
    cw: u32,
    ch: u32,
    dw: u32,
    dh: u32,
    dst: &mut [u8],
) {
    #[cfg(target_arch = "aarch64")]
    run_resize(
        src,
        width,
        x1,
        y1,
        cw,
        ch,
        dw,
        dh,
        dst,
        neon::NCHW_LANES,
        |a, b, c, d, e, f, g, h, i, j, k| unsafe {
            neon::resize_chunk4(a, b, c, d, e, f, g, h, i, j, k)
        },
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    run_resize(
        src,
        width,
        x1,
        y1,
        cw,
        ch,
        dw,
        dh,
        dst,
        x86::NCHW_LANES_SSE,
        x86::resize_chunk4,
    );
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "x86")))]
    scalar::resize_bilinear(src, width, x1, y1, cw, ch, dw, dh, dst);
}

pub fn rgb_to_bgr_in_place(data: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    neon::rgb_to_bgr_in_place(data);
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("ssse3") {
            x86::rgb_to_bgr_ssse3(data);
        } else {
            scalar::rgb_to_bgr_in_place(data);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "x86")))]
    scalar::rgb_to_bgr_in_place(data);
}

pub fn box3_bgr(data: &mut [u8], width: u32, height: u32) {
    scalar::box3_bgr(data, width, height);
}

pub fn gray_world(bgr: &mut [u8]) {
    scalar::gray_world(bgr);
}

pub fn blend_u8(dst: &mut [u8], src: &[u8], a: f32) {
    scalar::blend_u8(dst, src, a);
}

pub fn clahe_remap(
    bgr: &mut [u8],
    width: u32,
    height: u32,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    luts: &[[u8; HIST]],
) {
    scalar::clahe_remap(bgr, width, height, tx, ty, tw, th, luts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::ColorNorm;
    use half::f16;

    fn fill(w: u32, h: u32) -> Vec<u8> {
        (0..w as usize * h as usize * 3)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect()
    }

    fn max_abs_f16(a: &[f16], b: &[f16]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0f32, f32::max)
    }

    fn run(
        f: impl Fn(&[u8], u32, u32, u32, u32, u32, u32, u32, [usize; 3], [f32; 3], [f32; 3], &mut [f16]),
        src: &[u8],
        sw: u32,
        sh: u32,
        dw: u32,
        dh: u32,
        n: &ColorNorm,
    ) -> Vec<f16> {
        let mut d = vec![f16::ZERO; 3 * dw as usize * dh as usize];
        f(
            src, sw, 0, 0, sw, sh, dw, dh, n.src, n.scale, n.bias, &mut d,
        );
        d
    }

    #[test]
    fn bilinear_dispatch_matches_scalar() {
        let n = ColorNorm::IMAGENET;
        let src = fill(64, 48);
        for (dw, dh) in [(16u32, 16u32), (17, 16), (16, 17), (8, 8)] {
            let a = run(bilinear_nchw, &src, 64, 48, dw, dh, &n);
            let b = run(scalar::bilinear_nchw, &src, 64, 48, dw, dh, &n);
            let err = max_abs_f16(&a, &b);
            assert!(err < 2e-3, "{dw}x{dh} err={err}");
        }
    }

    #[test]
    fn affine_roi_dispatch_matches_scalar() {
        let n = ColorNorm::IMAGENET;
        let src = fill(40, 30);
        let mut a = vec![f16::ZERO; 3 * 21 * 16];
        let mut b = vec![f16::ZERO; 3 * 21 * 16];
        apply_affine_roi(&src, 40, 5, 7, 21, 16, n.src, n.scale, n.bias, &mut a);
        scalar::apply_affine_roi(&src, 40, 5, 7, 21, 16, n.src, n.scale, n.bias, &mut b);
        assert!(max_abs_f16(&a, &b) < 2e-3);
    }

    #[test]
    fn resize_dispatch_matches_scalar() {
        let src = fill(40, 30);
        let mut a = vec![0u8; 17 * 16 * 3];
        let mut b = vec![0u8; 17 * 16 * 3];
        resize_bilinear(&src, 40, 2, 3, 30, 24, 17, 16, &mut a);
        scalar::resize_bilinear(&src, 40, 2, 3, 30, 24, 17, 16, &mut b);
        let err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs())
            .max()
            .unwrap();
        assert!(err <= 1, "max abs {err}");
    }

    #[test]
    fn rgb_to_bgr_dispatch_matches_scalar() {
        let mut a = fill(19, 5);
        let mut b = a.clone();
        rgb_to_bgr_in_place(&mut a);
        scalar::rgb_to_bgr_in_place(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn box3_constant_is_identity() {
        let mut im = vec![77u8; 24 * 18 * 3];
        box3_bgr(&mut im, 24, 18);
        assert!(im.iter().all(|&v| v == 77));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn bilinear_sse2_and_avx2_match_scalar() {
        let n = ColorNorm::RETINA;
        let src = fill(48, 40);
        let sse = run(x86::bilinear_nchw_sse2, &src, 48, 40, 17, 16, &n);
        let sc = run(scalar::bilinear_nchw, &src, 48, 40, 17, 16, &n);
        assert!(max_abs_f16(&sse, &sc) < 2e-3, "sse2");
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            let avx = run(x86::bilinear_nchw_avx2, &src, 48, 40, 17, 16, &n);
            assert!(max_abs_f16(&avx, &sc) < 2e-3, "avx2");
        }
    }
}
