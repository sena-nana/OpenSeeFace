//! Shared FMA scalar kernels (remainder, fallback, and SIMD test oracle).

use half::f16;

#[inline]
pub fn affine_f16(v: f32, scale: f32, bias: f32) -> f16 {
    f16::from_f32(v * scale + bias)
}

#[inline]
pub fn sample_x(x: u32, fx: f32, last_x: u32) -> (u32, u32, f32) {
    let sx = (x as f32 + 0.5) * fx - 0.5;
    let ix0 = sx.floor().clamp(0.0, last_x as f32) as u32;
    (
        ix0,
        (ix0 + 1).min(last_x),
        (sx - ix0 as f32).clamp(0.0, 1.0),
    )
}

#[inline]
pub fn sample_y(y: u32, fy: f32, last_y: u32) -> (u32, u32, f32) {
    let sy = (y as f32 + 0.5) * fy - 0.5;
    let iy0 = sy.floor().clamp(0.0, last_y as f32) as u32;
    (
        iy0,
        (iy0 + 1).min(last_y),
        (sy - iy0 as f32).clamp(0.0, 1.0),
    )
}

#[inline]
pub fn nchw_pixel(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    last_x: u32,
    fx: f32,
    gy0: u32,
    gy1: u32,
    wy: f32,
    x: u32,
    src: [usize; 3],
    scale: [f32; 3],
    bias: [f32; 3],
    n: usize,
    o: usize,
    data: &mut [f16],
) {
    let (ix0, ix1, wx) = sample_x(x, fx, last_x);
    let i00 = ((gy0 * src_w + x0 + ix0) * 3) as usize;
    let i10 = ((gy0 * src_w + x0 + ix1) * 3) as usize;
    let i01 = ((gy1 * src_w + x0 + ix0) * 3) as usize;
    let i11 = ((gy1 * src_w + x0 + ix1) * 3) as usize;
    let omwx = 1.0 - wx;
    let omwy = 1.0 - wy;
    for c in 0..3 {
        let s = src[c];
        let v = src_bgr[i00 + s] as f32 * omwx * omwy
            + src_bgr[i10 + s] as f32 * wx * omwy
            + src_bgr[i01 + s] as f32 * omwx * wy
            + src_bgr[i11 + s] as f32 * wx * wy;
        data[c * n + o] = affine_f16(v, scale[c], bias[c]);
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
    let n = (dst_w * dst_h) as usize;
    let fx = cw as f32 / dst_w as f32;
    let fy = ch as f32 / dst_h as f32;
    let last_x = cw - 1;
    let last_y = ch - 1;
    for y in 0..dst_h {
        let (iy0, iy1, wy) = sample_y(y, fy, last_y);
        let gy0 = y0 + iy0;
        let gy1 = y0 + iy1;
        for x in 0..dst_w {
            nchw_pixel(
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
        for x in 0..cw {
            let p = (((y0 + y) * src_w + (x0 + x)) * 3) as usize;
            let o = (y * cw + x) as usize;
            data[o] = affine_f16(src_bgr[p + src[0]] as f32, scale[0], bias[0]);
            data[n + o] = affine_f16(src_bgr[p + src[1]] as f32, scale[1], bias[1]);
            data[2 * n + o] = affine_f16(src_bgr[p + src[2]] as f32, scale[2], bias[2]);
        }
    }
}

#[inline]
pub fn resize_pixel(
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
    let (ix0, ix1, wx) = sample_x(x, fx, last_x);
    let i00 = ((gy0 * width + x1 + ix0) * 3) as usize;
    let i10 = ((gy0 * width + x1 + ix1) * 3) as usize;
    let i01 = ((gy1 * width + x1 + ix0) * 3) as usize;
    let i11 = ((gy1 * width + x1 + ix1) * 3) as usize;
    let omwx = 1.0 - wx;
    let omwy = 1.0 - wy;
    for c in 0..3 {
        let v = src[i00 + c] as f32 * omwx * omwy
            + src[i10 + c] as f32 * wx * omwy
            + src[i01 + c] as f32 * omwx * wy
            + src[i11 + c] as f32 * wx * wy;
        dst[o + c] = v.round().clamp(0.0, 255.0) as u8;
    }
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
    let fx = cw as f32 / dw as f32;
    let fy = ch as f32 / dh as f32;
    let last_x = cw - 1;
    let last_y = ch - 1;
    for y in 0..dh {
        let (iy0, iy1, wy) = sample_y(y, fy, last_y);
        let gy0 = y1 + iy0;
        let gy1 = y1 + iy1;
        for x in 0..dw {
            resize_pixel(
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
        }
    }
}

pub fn rgb_to_bgr_in_place(data: &mut [u8]) {
    for px in data.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}
