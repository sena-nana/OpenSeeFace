//! aarch64 NEON kernels (macOS, Linux, Windows ARM64).

use half::f16;
use std::arch::aarch64::*;

use super::scalar;

pub const NCHW_LANES: u32 = 4;
pub const BGR_LANES: u32 = 8;

#[inline]
unsafe fn load4(src: &[u8], idx: [usize; 4]) -> float32x4_t {
    vld1q_f32(
        [
            src[idx[0]] as f32,
            src[idx[1]] as f32,
            src[idx[2]] as f32,
            src[idx[3]] as f32,
        ]
        .as_ptr(),
    )
}

#[inline]
unsafe fn store4_f16(data: &mut [f16], base: usize, v: float32x4_t) {
    let mut t = [0f32; 4];
    vst1q_f32(t.as_mut_ptr(), v);
    for i in 0..4 {
        data[base + i] = f16::from_f32(t[i]);
    }
}

unsafe fn weights4(
    x: u32,
    last_x: u32,
    fx: f32,
    wy: f32,
) -> ([u32; 4], [u32; 4], [float32x4_t; 4]) {
    let mut ix0 = [0u32; 4];
    let mut ix1 = [0u32; 4];
    let mut wx = [0f32; 4];
    for i in 0..4 {
        let (a, b, w) = scalar::sample_x(x + i as u32, fx, last_x);
        ix0[i] = a;
        ix1[i] = b;
        wx[i] = w;
    }
    let vwx = vld1q_f32(wx.as_ptr());
    let one = vdupq_n_f32(1.0);
    let omwx = vsubq_f32(one, vwx);
    let vwy = vdupq_n_f32(wy);
    let omwy = vsubq_f32(one, vwy);
    (
        ix0,
        ix1,
        [
            vmulq_f32(omwx, omwy),
            vmulq_f32(vwx, omwy),
            vmulq_f32(omwx, vwy),
            vmulq_f32(vwx, vwy),
        ],
    )
}

unsafe fn lerp4(p: [float32x4_t; 4], w: [float32x4_t; 4]) -> float32x4_t {
    let mut acc = vmulq_f32(p[0], w[0]);
    acc = vmlaq_f32(acc, p[1], w[1]);
    acc = vmlaq_f32(acc, p[2], w[2]);
    vmlaq_f32(acc, p[3], w[3])
}

pub unsafe fn nchw_chunk4(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    last_x: u32,
    fx: f32,
    gy0: u32,
    gy1: u32,
    wy: f32,
    x: u32,
    y: u32,
    dst_w: u32,
    src: [usize; 3],
    scale: [f32; 3],
    bias: [f32; 3],
    n: usize,
    data: &mut [f16],
) {
    let (ix0, ix1, w) = weights4(x, last_x, fx, wy);
    let o = (y * dst_w + x) as usize;
    for c in 0..3 {
        let s = src[c];
        let mut idx = [[0usize; 4]; 4];
        for i in 0..4 {
            let gx0 = (x0 + ix0[i]) as usize;
            let gx1 = (x0 + ix1[i]) as usize;
            idx[0][i] = (gy0 as usize * src_w as usize + gx0) * 3 + s;
            idx[1][i] = (gy0 as usize * src_w as usize + gx1) * 3 + s;
            idx[2][i] = (gy1 as usize * src_w as usize + gx0) * 3 + s;
            idx[3][i] = (gy1 as usize * src_w as usize + gx1) * 3 + s;
        }
        let v = lerp4(
            [
                load4(src_bgr, idx[0]),
                load4(src_bgr, idx[1]),
                load4(src_bgr, idx[2]),
                load4(src_bgr, idx[3]),
            ],
            w,
        );
        store4_f16(
            data,
            c * n + o,
            vmlaq_n_f32(vdupq_n_f32(bias[c]), v, scale[c]),
        );
    }
}

pub unsafe fn resize_chunk4(
    src: &[u8],
    width: u32,
    x1: u32,
    last_x: u32,
    fx: f32,
    gy0: u32,
    gy1: u32,
    wy: f32,
    x: u32,
    dst: &mut [u8],
    o: usize,
) {
    let (ix0, ix1, w) = weights4(x, last_x, fx, wy);
    for c in 0..3 {
        let mut idx = [[0usize; 4]; 4];
        for i in 0..4 {
            idx[0][i] = (gy0 as usize * width as usize + x1 as usize + ix0[i] as usize) * 3 + c;
            idx[1][i] = (gy0 as usize * width as usize + x1 as usize + ix1[i] as usize) * 3 + c;
            idx[2][i] = (gy1 as usize * width as usize + x1 as usize + ix0[i] as usize) * 3 + c;
            idx[3][i] = (gy1 as usize * width as usize + x1 as usize + ix1[i] as usize) * 3 + c;
        }
        let v = lerp4(
            [
                load4(src, idx[0]),
                load4(src, idx[1]),
                load4(src, idx[2]),
                load4(src, idx[3]),
            ],
            w,
        );
        let mut t = [0f32; 4];
        vst1q_f32(t.as_mut_ptr(), vrndnq_f32(v));
        for i in 0..4 {
            dst[o + i * 3 + c] = t[i].clamp(0.0, 255.0) as u8;
        }
    }
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
    let n = (cw * ch) as usize;
    for y in 0..ch {
        let mut x = 0u32;
        while x + BGR_LANES <= cw {
            let p = (((y0 + y) * src_w + (x0 + x)) * 3) as usize;
            let o = (y * cw + x) as usize;
            unsafe {
                let pix = vld3_u8(src_bgr.as_ptr().add(p));
                let chans = [pix.0, pix.1, pix.2];
                for c in 0..3 {
                    let v16 = vmovl_u8(chans[src[c]]);
                    let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16)));
                    let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v16)));
                    store4_f16(
                        data,
                        c * n + o,
                        vmlaq_n_f32(vdupq_n_f32(bias[c]), lo, scale[c]),
                    );
                    store4_f16(
                        data,
                        c * n + o + 4,
                        vmlaq_n_f32(vdupq_n_f32(bias[c]), hi, scale[c]),
                    );
                }
            }
            x += BGR_LANES;
        }
        while x < cw {
            let p = (((y0 + y) * src_w + (x0 + x)) * 3) as usize;
            let o = (y * cw + x) as usize;
            data[o] = scalar::affine_f16(src_bgr[p + src[0]] as f32, scale[0], bias[0]);
            data[n + o] = scalar::affine_f16(src_bgr[p + src[1]] as f32, scale[1], bias[1]);
            data[2 * n + o] = scalar::affine_f16(src_bgr[p + src[2]] as f32, scale[2], bias[2]);
            x += 1;
        }
    }
}

pub fn rgb_to_bgr_in_place(data: &mut [u8]) {
    let mut i = 0usize;
    while i + 24 <= data.len() {
        unsafe {
            let pix = vld3_u8(data.as_ptr().add(i));
            vst3_u8(data.as_mut_ptr().add(i), uint8x8x3_t(pix.2, pix.1, pix.0));
        }
        i += 24;
    }
    if i < data.len() {
        scalar::rgb_to_bgr_in_place(&mut data[i..]);
    }
}
