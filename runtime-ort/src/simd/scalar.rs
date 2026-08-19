//! Shared FMA scalar kernels (remainder, fallback, and SIMD test oracle).

use half::f16;

use super::{clamp_u8, HIST};

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

/// 3×3 box with replicated borders, integer `/9` via separable sums.
pub fn box3_bgr(data: &mut [u8], width: u32, height: u32) {
    if width < 2 || height < 2 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let mut tmp = vec![0u16; w * h * 3];
    for y in 0..h {
        let row = y * w * 3;
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            for c in 0..3 {
                tmp[row + x * 3 + c] = data[row + xm * 3 + c] as u16
                    + data[row + x * 3 + c] as u16
                    + data[row + xp * 3 + c] as u16;
            }
        }
    }
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                let s = tmp[(ym * w + x) * 3 + c] as u32
                    + tmp[i] as u32
                    + tmp[(yp * w + x) * 3 + c] as u32;
                data[i] = (s / 9) as u8;
            }
        }
    }
}

pub fn gray_world(bgr: &mut [u8]) {
    let n = bgr.len() / 3;
    if n == 0 {
        return;
    }
    let mut sb = 0u64;
    let mut sg = 0u64;
    let mut sr = 0u64;
    for p in bgr.chunks_exact(3) {
        sb += p[0] as u64;
        sg += p[1] as u64;
        sr += p[2] as u64;
    }
    let inv = 1.0 / n as f32;
    let mb = sb as f32 * inv;
    let mg = sg as f32 * inv;
    let mr = sr as f32 * inv;
    let gray = (mb + mg + mr) / 3.0;
    let gain = |m: f32| gray / m.max(1e-3);
    let gb = gain(mb);
    let gg = gain(mg);
    let gr = gain(mr);
    for p in bgr.chunks_exact_mut(3) {
        p[0] = clamp_u8(p[0] as f32 * gb);
        p[1] = clamp_u8(p[1] as f32 * gg);
        p[2] = clamp_u8(p[2] as f32 * gr);
    }
}

pub fn blend_u8(dst: &mut [u8], src: &[u8], a: f32) {
    let b = 1.0 - a;
    for (o, s) in dst.iter_mut().zip(src.iter()) {
        *o = clamp_u8(*o as f32 * a + *s as f32 * b);
    }
}

#[inline]
fn yuv_from_bgr(b: f32, g: f32, r: f32) -> (f32, f32, f32) {
    let y = 0.114 * b + 0.587 * g + 0.299 * r;
    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    (y, cb, cr)
}

#[inline]
fn bgr_from_yuv(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    (
        y + 1.772 * (cb - 128.0),
        y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0),
        y + 1.402 * (cr - 128.0),
    )
}

fn map_y(
    luts: &[[u8; HIST]],
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    x: u32,
    y: u32,
    bin: usize,
) -> f32 {
    let fx = x as f32 / tw as f32 - 0.5;
    let fy = y as f32 / th as f32 - 0.5;
    let tx1 = fx.floor() as i32;
    let ty1 = fy.floor() as i32;
    let wx = fx - tx1 as f32;
    let wy = fy - ty1 as f32;
    let clamp_t = |t: i32, n: u32| t.clamp(0, n as i32 - 1) as u32;
    let xa = clamp_t(tx1, tx);
    let xb = clamp_t(tx1 + 1, tx);
    let ya = clamp_t(ty1, ty);
    let yb = clamp_t(ty1 + 1, ty);
    let lut = |txx: u32, tyy: u32| luts[(tyy * tx + txx) as usize][bin] as f32;
    lut(xa, ya) * (1.0 - wy) * (1.0 - wx)
        + lut(xb, ya) * (1.0 - wy) * wx
        + lut(xa, yb) * wy * (1.0 - wx)
        + lut(xb, yb) * wy * wx
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
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let (yv, cb, cr) = yuv_from_bgr(bgr[i] as f32, bgr[i + 1] as f32, bgr[i + 2] as f32);
            let bin = yv.round().clamp(0.0, 255.0) as usize;
            let (b2, g2, r2) = bgr_from_yuv(map_y(luts, tx, ty, tw, th, x, y, bin), cb, cr);
            bgr[i] = clamp_u8(b2);
            bgr[i + 1] = clamp_u8(g2);
            bgr[i + 2] = clamp_u8(r2);
        }
    }
}
