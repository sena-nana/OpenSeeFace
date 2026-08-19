//! x86_64 SSE2 / SSSE3 / AVX2+F16C kernels (Windows, Linux, Intel Mac).

use half::f16;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar;

pub const NCHW_LANES_AVX2: u32 = 8;
pub const NCHW_LANES_SSE: u32 = 4;

#[inline]
unsafe fn load4(src: &[u8], idx: [usize; 4]) -> __m128 {
    _mm_setr_ps(
        src[idx[0]] as f32,
        src[idx[1]] as f32,
        src[idx[2]] as f32,
        src[idx[3]] as f32,
    )
}

#[inline]
unsafe fn store4_f16(data: &mut [f16], base: usize, v: __m128) {
    let mut t = [0f32; 4];
    _mm_storeu_ps(t.as_mut_ptr(), v);
    for i in 0..4 {
        data[base + i] = f16::from_f32(t[i]);
    }
}

unsafe fn weights4(x: u32, last_x: u32, fx: f32, wy: f32) -> ([u32; 4], [u32; 4], [__m128; 4]) {
    let mut ix0 = [0u32; 4];
    let mut ix1 = [0u32; 4];
    let mut wx = [0f32; 4];
    for i in 0..4 {
        let (a, b, w) = scalar::sample_x(x + i as u32, fx, last_x);
        ix0[i] = a;
        ix1[i] = b;
        wx[i] = w;
    }
    let vwx = _mm_loadu_ps(wx.as_ptr());
    let one = _mm_set1_ps(1.0);
    let omwx = _mm_sub_ps(one, vwx);
    let vwy = _mm_set1_ps(wy);
    let omwy = _mm_sub_ps(one, vwy);
    (
        ix0,
        ix1,
        [
            _mm_mul_ps(omwx, omwy),
            _mm_mul_ps(vwx, omwy),
            _mm_mul_ps(omwx, vwy),
            _mm_mul_ps(vwx, vwy),
        ],
    )
}

unsafe fn lerp4(p: [__m128; 4], w: [__m128; 4]) -> __m128 {
    let mut acc = _mm_mul_ps(p[0], w[0]);
    acc = _mm_add_ps(acc, _mm_mul_ps(p[1], w[1]));
    acc = _mm_add_ps(acc, _mm_mul_ps(p[2], w[2]));
    _mm_add_ps(acc, _mm_mul_ps(p[3], w[3]))
}

fn gather4(
    src: &[u8],
    src_w: u32,
    x0: u32,
    ix0: [u32; 4],
    ix1: [u32; 4],
    gy0: u32,
    gy1: u32,
    s: usize,
) -> [[usize; 4]; 4] {
    let mut idx = [[0usize; 4]; 4];
    for i in 0..4 {
        let gx0 = (x0 + ix0[i]) as usize;
        let gx1 = (x0 + ix1[i]) as usize;
        idx[0][i] = (gy0 as usize * src_w as usize + gx0) * 3 + s;
        idx[1][i] = (gy0 as usize * src_w as usize + gx1) * 3 + s;
        idx[2][i] = (gy1 as usize * src_w as usize + gx0) * 3 + s;
        idx[3][i] = (gy1 as usize * src_w as usize + gx1) * 3 + s;
    }
    idx
}

pub fn nchw_chunk4(
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
    unsafe {
        let (ix0, ix1, w) = weights4(x, last_x, fx, wy);
        let o = (y * dst_w + x) as usize;
        for c in 0..3 {
            let idx = gather4(src_bgr, src_w, x0, ix0, ix1, gy0, gy1, src[c]);
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
                _mm_add_ps(_mm_mul_ps(v, _mm_set1_ps(scale[c])), _mm_set1_ps(bias[c])),
            );
        }
    }
}

#[target_feature(enable = "avx2", enable = "f16c")]
pub unsafe fn nchw_chunk8(
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
    let mut ix0 = [0u32; 8];
    let mut ix1 = [0u32; 8];
    let mut wx = [0f32; 8];
    for i in 0..8 {
        let (a, b, w) = scalar::sample_x(x + i as u32, fx, last_x);
        ix0[i] = a;
        ix1[i] = b;
        wx[i] = w;
    }
    let vwx = _mm256_loadu_ps(wx.as_ptr());
    let one = _mm256_set1_ps(1.0);
    let omwx = _mm256_sub_ps(one, vwx);
    let vwy = _mm256_set1_ps(wy);
    let omwy = _mm256_sub_ps(one, vwy);
    let w00 = _mm256_mul_ps(omwx, omwy);
    let w10 = _mm256_mul_ps(vwx, omwy);
    let w01 = _mm256_mul_ps(omwx, vwy);
    let w11 = _mm256_mul_ps(vwx, vwy);
    let o = (y * dst_w + x) as usize;
    for c in 0..3 {
        let s = src[c];
        let mut p00 = [0f32; 8];
        let mut p10 = [0f32; 8];
        let mut p01 = [0f32; 8];
        let mut p11 = [0f32; 8];
        for i in 0..8 {
            let gx0 = (x0 + ix0[i]) as usize;
            let gx1 = (x0 + ix1[i]) as usize;
            p00[i] = src_bgr[(gy0 as usize * src_w as usize + gx0) * 3 + s] as f32;
            p10[i] = src_bgr[(gy0 as usize * src_w as usize + gx1) * 3 + s] as f32;
            p01[i] = src_bgr[(gy1 as usize * src_w as usize + gx0) * 3 + s] as f32;
            p11[i] = src_bgr[(gy1 as usize * src_w as usize + gx1) * 3 + s] as f32;
        }
        let mut acc = _mm256_mul_ps(_mm256_loadu_ps(p00.as_ptr()), w00);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(p10.as_ptr()), w10));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(p01.as_ptr()), w01));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(p11.as_ptr()), w11));
        let out = _mm256_add_ps(
            _mm256_mul_ps(acc, _mm256_set1_ps(scale[c])),
            _mm256_set1_ps(bias[c]),
        );
        _mm_storeu_si128(
            data.as_mut_ptr().add(c * n + o) as *mut __m128i,
            _mm256_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT),
        );
    }
}

pub fn resize_chunk4(
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
    unsafe {
        let (ix0, ix1, w) = weights4(x, last_x, fx, wy);
        for c in 0..3 {
            let idx = gather4(src, width, x1, ix0, ix1, gy0, gy1, c);
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
            _mm_storeu_ps(t.as_mut_ptr(), v);
            for i in 0..4 {
                dst[o + i * 3 + c] = t[i].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// 4 BGR pixels (12 bytes); caller must ensure `ptr` has 16 readable bytes.
#[target_feature(enable = "ssse3")]
unsafe fn deinterleave_bgr4(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let v = _mm_loadu_si128(ptr as *const __m128i);
    (
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        ),
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        ),
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        ),
    )
}

#[inline]
unsafe fn u8x4_to_f32(v: __m128i) -> __m128 {
    let zero = _mm_setzero_si128();
    _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero), zero))
}

pub fn apply_affine_roi_ssse3(
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
        while x + 4 <= cw {
            let p = (((y0 + y) * src_w + (x0 + x)) * 3) as usize;
            let o = (y * cw + x) as usize;
            if p + 16 <= src_bgr.len() {
                unsafe {
                    let planes = deinterleave_bgr4(src_bgr.as_ptr().add(p));
                    let chans = [planes.0, planes.1, planes.2];
                    for c in 0..3 {
                        let v = u8x4_to_f32(chans[src[c]]);
                        store4_f16(
                            data,
                            c * n + o,
                            _mm_add_ps(_mm_mul_ps(v, _mm_set1_ps(scale[c])), _mm_set1_ps(bias[c])),
                        );
                    }
                }
            } else {
                for k in 0..4 {
                    let px = (((y0 + y) * src_w + (x0 + x + k)) * 3) as usize;
                    let oo = o + k as usize;
                    data[oo] = scalar::affine_f16(src_bgr[px + src[0]] as f32, scale[0], bias[0]);
                    data[n + oo] =
                        scalar::affine_f16(src_bgr[px + src[1]] as f32, scale[1], bias[1]);
                    data[2 * n + oo] =
                        scalar::affine_f16(src_bgr[px + src[2]] as f32, scale[2], bias[2]);
                }
            }
            x += 4;
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

pub fn rgb_to_bgr_ssse3(data: &mut [u8]) {
    let mut i = 0usize;
    while i + 16 <= data.len() {
        unsafe {
            let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
            let swapped = _mm_shuffle_epi8(
                v,
                _mm_setr_epi8(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15),
            );
            let mut tmp = [0u8; 16];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, swapped);
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), data.as_mut_ptr().add(i), 12);
        }
        i += 12;
    }
    if i < data.len() {
        scalar::rgb_to_bgr_in_place(&mut data[i..]);
    }
}

pub fn bilinear_nchw_sse2(
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
    super::run_nchw(
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
        NCHW_LANES_SSE,
        nchw_chunk4,
    );
}

pub fn bilinear_nchw_avx2(
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
    super::run_nchw(
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
        NCHW_LANES_AVX2,
        |a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p| unsafe {
            nchw_chunk8(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
        },
    );
}
