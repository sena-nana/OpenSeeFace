//! CPU fused bilinear resize + BGR/RGB remap + baked mean/std → f16 NCHW.

use std::path::Path;

use anyhow::Result;
use half::f16;

use crate::decode::TensorF16;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const RETINA_MEAN: [f32; 3] = [104.0, 117.0, 123.0];

const fn bake_lut(scale: [f32; 3], bias: [f32; 3]) -> [[f16; 256]; 3] {
    let mut lut = [[f16::from_f32_const(0.0); 256]; 3];
    let mut c = 0;
    while c < 3 {
        let mut v = 0;
        while v < 256 {
            lut[c][v] = f16::from_f32_const((v as f32) * scale[c] + bias[c]);
            v += 1;
        }
        c += 1;
    }
    lut
}

/// Baked BGR-source → NCHW dest normalization.
///
/// ImageNet dest planes are RGB (`src = [2,1,0]`) with `(x/255 - mean) / std`.
/// RetinaFace dest planes stay BGR and subtract `(104, 117, 123)`.
#[derive(Clone, Copy, Debug)]
pub struct ColorNorm {
    pub scale: [f32; 3],
    pub bias: [f32; 3],
    /// BGR channel index for destination planes 0..2.
    pub src: [usize; 3],
    pub lut: [[f16; 256]; 3],
}

impl ColorNorm {
    pub const IMAGENET: Self = {
        let scale = [
            1.0 / (IMAGENET_STD[0] * 255.0),
            1.0 / (IMAGENET_STD[1] * 255.0),
            1.0 / (IMAGENET_STD[2] * 255.0),
        ];
        let bias = [
            -(IMAGENET_MEAN[0] / IMAGENET_STD[0]),
            -(IMAGENET_MEAN[1] / IMAGENET_STD[1]),
            -(IMAGENET_MEAN[2] / IMAGENET_STD[2]),
        ];
        Self {
            scale,
            bias,
            src: [2, 1, 0],
            lut: bake_lut(scale, bias),
        }
    };

    pub const RETINA: Self = {
        let scale = [1.0, 1.0, 1.0];
        let bias = [-RETINA_MEAN[0], -RETINA_MEAN[1], -RETINA_MEAN[2]];
        Self {
            scale,
            bias,
            src: [0, 1, 2],
            lut: bake_lut(scale, bias),
        }
    };
}

#[derive(Clone)]
pub struct BgrImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl BgrImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    pub fn zeros(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width as usize * height as usize * 3],
        }
    }

    pub fn get(&self, x: i32, y: i32) -> [u8; 3] {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return [0, 0, 0];
        }
        let i = ((y as u32 * self.width + x as u32) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn set(&mut self, x: i32, y: i32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let i = ((y as u32 * self.width + x as u32) * 3) as usize;
        self.data[i] = c[0];
        self.data[i + 1] = c[1];
        self.data[i + 2] = c[2];
    }

    pub fn resize_buffer(&mut self, width: u32, height: u32) {
        let n = width as usize * height as usize * 3;
        if self.data.len() != n {
            self.data.resize(n, 0);
        }
        self.width = width;
        self.height = height;
    }

    pub fn flip_h(&self) -> Self {
        let mut out = Self::zeros(self.width, self.height);
        flip_h_rows(
            &self.data,
            &mut out.data,
            self.width as usize,
            self.height as usize,
        );
        out
    }

    pub fn flip_h_in_place(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut row = vec![0u8; w * 3];
        for y in 0..h {
            let s = y * w * 3;
            row.copy_from_slice(&self.data[s..s + w * 3]);
            for x in 0..w {
                let d = s + (w - 1 - x) * 3;
                self.data[d..d + 3].copy_from_slice(&row[x * 3..x * 3 + 3]);
            }
        }
    }

    pub fn sample(&self, x: f32, y: f32) -> [u8; 3] {
        if self.width == 0 || self.height == 0 {
            return [0, 0, 0];
        }
        let x0 = x.floor().clamp(0.0, (self.width - 1) as f32) as i32;
        let y0 = y.floor().clamp(0.0, (self.height - 1) as f32) as i32;
        let x1 = (x0 + 1).min(self.width as i32 - 1);
        let y1 = (y0 + 1).min(self.height as i32 - 1);
        let wx = (x - x0 as f32).clamp(0.0, 1.0);
        let wy = (y - y0 as f32).clamp(0.0, 1.0);
        let p00 = self.get(x0, y0);
        let p10 = self.get(x1, y0);
        let p01 = self.get(x0, y1);
        let p11 = self.get(x1, y1);
        let mut o = [0u8; 3];
        for c in 0..3 {
            let v = p00[c] as f32 * (1.0 - wx) * (1.0 - wy)
                + p10[c] as f32 * wx * (1.0 - wy)
                + p01[c] as f32 * (1.0 - wx) * wy
                + p11[c] as f32 * wx * wy;
            o[c] = v.round().clamp(0.0, 255.0) as u8;
        }
        o
    }

    /// OpenCV `getRotationMatrix2D` + `warpAffine` (positive degrees CCW).
    pub fn rotate_about(&self, angle_rad: f32, center: (f32, f32)) -> Self {
        let a = angle_rad;
        let (cos, sin) = (a.cos(), a.sin());
        let (cx, cy) = center;
        let mut out = Self::zeros(self.width, self.height);
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let sx = cos * dx + sin * dy + cx;
                let sy = -sin * dx + cos * dy + cy;
                out.set(x, y, self.sample(sx, sy));
            }
        }
        out
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let img = image::open(path).or_else(|_| image::load_from_memory(&std::fs::read(path)?))?;
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for p in rgb.pixels() {
            data.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

fn flip_h_rows(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    for y in 0..h {
        let row = y * w * 3;
        for x in 0..w {
            let s = row + x * 3;
            let d = row + (w - 1 - x) * 3;
            dst[d..d + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
}

fn apply_lut_roi(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    norm: &ColorNorm,
    data: &mut [f16],
) {
    let n = (cw * ch) as usize;
    let src = norm.src;
    let lut = &norm.lut;
    for y in 0..ch {
        for x in 0..cw {
            let p = (((y0 + y) * src_w + (x0 + x)) * 3) as usize;
            let o = (y * cw + x) as usize;
            data[o] = lut[0][src_bgr[p + src[0]] as usize];
            data[n + o] = lut[1][src_bgr[p + src[1]] as usize];
            data[2 * n + o] = lut[2][src_bgr[p + src[2]] as usize];
        }
    }
}

fn bilinear_nchw(
    src_bgr: &[u8],
    src_w: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    dst_w: u32,
    dst_h: u32,
    norm: &ColorNorm,
    data: &mut [f16],
) {
    let n = (dst_w * dst_h) as usize;
    let src = norm.src;
    let lut = &norm.lut;
    let fx = cw as f32 / dst_w as f32;
    let fy = ch as f32 / dst_h as f32;
    let last_x = cw - 1;
    let last_y = ch - 1;
    for y in 0..dst_h {
        let sy = (y as f32 + 0.5) * fy - 0.5;
        let iy0 = sy.floor().clamp(0.0, last_y as f32) as u32;
        let iy1 = (iy0 + 1).min(last_y);
        let wy = (sy - iy0 as f32).clamp(0.0, 1.0);
        let gy0 = y0 + iy0;
        let gy1 = y0 + iy1;
        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * fx - 0.5;
            let ix0 = sx.floor().clamp(0.0, last_x as f32) as u32;
            let ix1 = (ix0 + 1).min(last_x);
            let wx = (sx - ix0 as f32).clamp(0.0, 1.0);
            let gx0 = x0 + ix0;
            let gx1 = x0 + ix1;
            let i00 = ((gy0 * src_w + gx0) * 3) as usize;
            let i10 = ((gy0 * src_w + gx1) * 3) as usize;
            let i01 = ((gy1 * src_w + gx0) * 3) as usize;
            let i11 = ((gy1 * src_w + gx1) * 3) as usize;
            let o = (y * dst_w + x) as usize;
            for c in 0..3 {
                let s = src[c];
                let v = src_bgr[i00 + s] as f32 * (1.0 - wx) * (1.0 - wy)
                    + src_bgr[i10 + s] as f32 * wx * (1.0 - wy)
                    + src_bgr[i01 + s] as f32 * (1.0 - wx) * wy
                    + src_bgr[i11 + s] as f32 * wx * wy;
                let u = v.round().clamp(0.0, 255.0) as u8 as usize;
                data[c * n + o] = lut[c][u];
            }
        }
    }
}

fn clamp_roi(bgr: &BgrImage, x1: i32, y1: i32, x2: i32, y2: i32) -> (u32, u32, u32, u32) {
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = (x2.max(0) as u32).min(bgr.width);
    let y2 = (y2.max(0) as u32).min(bgr.height);
    (x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1))
}

/// Fused bilinear resize + BGR/RGB remap + baked mean/std → f16 NCHW, writing `dst`.
pub(crate) fn nchw_into(
    bgr: &BgrImage,
    width: u32,
    height: u32,
    norm: &ColorNorm,
    dst: &mut [f16],
) {
    nchw_roi_into(
        bgr,
        0,
        0,
        bgr.width as i32,
        bgr.height as i32,
        width,
        height,
        norm,
        dst,
    );
}

/// Like [`nchw_into`], sampling an axis-aligned ROI instead of copying a BGR crop first.
pub(crate) fn nchw_roi_into(
    bgr: &BgrImage,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    width: u32,
    height: u32,
    norm: &ColorNorm,
    dst: &mut [f16],
) {
    let n = (width as usize).saturating_mul(height as usize);
    if n == 0 || dst.len() < 3 * n {
        dst.fill(f16::ZERO);
        return;
    }
    let data = &mut dst[..3 * n];
    let (rx, ry, cw, ch) = clamp_roi(bgr, x1, y1, x2, y2);
    if cw == 0 || ch == 0 {
        data.fill(f16::ZERO);
        return;
    }
    if width == cw && height == ch {
        apply_lut_roi(&bgr.data, bgr.width, rx, ry, cw, ch, norm, data);
        return;
    }
    bilinear_nchw(
        &bgr.data, bgr.width, rx, ry, cw, ch, width, height, norm, data,
    );
}

/// Fused bilinear resize + BGR/RGB remap + baked mean/std → f16 NCHW.
pub fn nchw(bgr: &BgrImage, width: u32, height: u32, norm: &ColorNorm) -> TensorF16 {
    let n = (width * height) as usize;
    let mut data = vec![f16::ZERO; 3 * n];
    nchw_into(bgr, width, height, norm, &mut data);
    TensorF16 {
        shape: vec![1, 3, height as i64, width as i64],
        data,
    }
}

/// Half-pixel bilinear resize in BGR. Identity when the size already matches.
pub fn resize_bgr(bgr: &BgrImage, dw: u32, dh: u32) -> BgrImage {
    if dw == 0 || dh == 0 {
        return BgrImage {
            width: 0,
            height: 0,
            data: Vec::new(),
        };
    }
    if dw == bgr.width && dh == bgr.height {
        return bgr.clone();
    }
    let mut data = vec![0u8; dw as usize * dh as usize * 3];
    resize_roi_into(
        &bgr.data,
        bgr.width,
        bgr.height,
        0,
        0,
        bgr.width as i32,
        bgr.height as i32,
        dw,
        dh,
        &mut data,
    );
    BgrImage {
        width: dw,
        height: dh,
        data,
    }
}

/// Crop `[x1,y1,x2,y2)` then half-pixel bilinear-resize into `dst` (`dw*dh*3` bytes).
pub(crate) fn resize_roi_into(
    src: &[u8],
    width: u32,
    height: u32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    dw: u32,
    dh: u32,
    dst: &mut [u8],
) {
    let need = dw as usize * dh as usize * 3;
    if dw == 0 || dh == 0 || dst.len() < need {
        return;
    }
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = (x2.max(0) as u32).min(width);
    let y2 = (y2.max(0) as u32).min(height);
    let cw = x2.saturating_sub(x1);
    let ch = y2.saturating_sub(y1);
    if cw == 0 || ch == 0 || width == 0 || height == 0 {
        dst[..need].fill(0);
        return;
    }
    if cw == dw && ch == dh {
        for row in 0..dh {
            let s = (((y1 + row) * width + x1) * 3) as usize;
            let d = (row * dw * 3) as usize;
            let n = (dw * 3) as usize;
            dst[d..d + n].copy_from_slice(&src[s..s + n]);
        }
        return;
    }
    let fx = cw as f32 / dw as f32;
    let fy = ch as f32 / dh as f32;
    let y_max = (ch - 1) as f32;
    let x_max = (cw - 1) as f32;
    for y in 0..dh {
        let sy = (y as f32 + 0.5) * fy - 0.5;
        let y0 = sy.floor().clamp(0.0, y_max) as u32;
        let y1s = (y0 + 1).min(ch - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..dw {
            let sx = (x as f32 + 0.5) * fx - 0.5;
            let x0 = sx.floor().clamp(0.0, x_max) as u32;
            let x1s = (x0 + 1).min(cw - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i00 = (((y1 + y0) * width + x1 + x0) * 3) as usize;
            let i10 = (((y1 + y0) * width + x1 + x1s) * 3) as usize;
            let i01 = (((y1 + y1s) * width + x1 + x0) * 3) as usize;
            let i11 = (((y1 + y1s) * width + x1 + x1s) * 3) as usize;
            let o = ((y * dw + x) * 3) as usize;
            for c in 0..3 {
                let v = src[i00 + c] as f32 * (1.0 - wx) * (1.0 - wy)
                    + src[i10 + c] as f32 * wx * (1.0 - wy)
                    + src[i01 + c] as f32 * (1.0 - wx) * wy
                    + src[i11 + c] as f32 * wx * wy;
                dst[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Padded detection crop, matching `benchmarks/scenarios.py` `face_crop`.
pub fn face_crop(frame: &BgrImage, d: &[f32; 5], margin: f32) -> (BgrImage, (i32, i32), f32) {
    let (x, y, w, h) = (d[0], d[1], d[2], d[3]);
    let x1 = (x - w * margin).max(0.0) as i32;
    let y1 = (y - h * margin).max(0.0) as i32;
    let x2 = (x + w * (1.0 + margin)).min(frame.width as f32) as i32;
    let y2 = (y + h * (1.0 + margin)).min(frame.height as f32) as i32;
    if x2 - x1 < 8 || y2 - y1 < 8 {
        return (frame.clone(), (0, 0), frame.height as f32);
    }
    let inner_h = (y + h).min(y2 as f32) - y.max(y1 as f32);
    (crop_img(frame, x1, y1, x2, y2), (x1, y1), inner_h.max(8.0))
}

pub fn synth_canvas(width: u32, height: u32) -> BgrImage {
    let n = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    let data = (0..n).map(|i| 42u8.wrapping_add((i % 7) as u8)).collect();
    BgrImage {
        width,
        height,
        data,
    }
}

pub fn paste_bgr(dst: &mut BgrImage, src: &BgrImage, x: i32, y: i32) {
    let x1 = x.max(0) as u32;
    let y1 = y.max(0) as u32;
    let x2 = ((x + src.width as i32).max(0) as u32).min(dst.width);
    let y2 = ((y + src.height as i32).max(0) as u32).min(dst.height);
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let sx0 = x1 as i32 - x;
    let sy0 = y1 as i32 - y;
    for row in 0..(y2 - y1) {
        let ds = (((y1 + row) * dst.width + x1) * 3) as usize;
        let ss = (((sy0 as u32 + row) * src.width + sx0 as u32) * 3) as usize;
        let n = ((x2 - x1) * 3) as usize;
        dst.data[ds..ds + n].copy_from_slice(&src.data[ss..ss + n]);
    }
}

pub fn imagenet_nchw(bgr: &BgrImage, size: u32) -> TensorF16 {
    nchw(bgr, size, size, &ColorNorm::IMAGENET)
}

pub(crate) fn imagenet_nchw_into(bgr: &BgrImage, size: u32, dst: &mut [f16]) {
    nchw_into(bgr, size, size, &ColorNorm::IMAGENET, dst);
}

pub fn imagenet_nchw_roi_into(
    bgr: &BgrImage,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    size: u32,
    dst: &mut [f16],
) {
    nchw_roi_into(bgr, x1, y1, x2, y2, size, size, &ColorNorm::IMAGENET, dst);
}

pub fn retina_nchw(bgr: &BgrImage) -> TensorF16 {
    nchw(bgr, 640, 640, &ColorNorm::RETINA)
}

pub(crate) fn retina_nchw_into(bgr: &BgrImage, dst: &mut [f16]) {
    nchw_into(bgr, 640, 640, &ColorNorm::RETINA, dst);
}

/// In-place RGB→BGR swizzle of packed 8-bit pixels.
pub(crate) fn rgb_to_bgr_in_place(data: &mut [u8]) {
    for px in data.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

/// Padded face crop in image coordinates, matching the Python tracker.
pub fn crop_box_pad(
    frame: &BgrImage,
    d: &[f32; 5],
    pad_x: f32,
    pad_y: f32,
) -> (i32, i32, i32, i32) {
    let (x, y, w, h) = (d[0], d[1], d[2], d[3]);
    let x1 = x - (w * pad_x) as i32 as f32;
    let y1 = y - (h * pad_y) as i32 as f32;
    let x2 = x + w + (w * pad_x) as i32 as f32;
    let y2 = y + h + (h * pad_y) as i32 as f32;
    let clamp = |px: f32, py: f32| {
        let px = px.clamp(0.0, frame.width as f32 - 1.0) as i32;
        let py = py.clamp(0.0, frame.height as f32 - 1.0) as i32 + 1;
        (px, py)
    };
    let (x1, y1) = clamp(x1, y1);
    let (x2, y2) = clamp(x2, y2);
    (x1, y1, x2, y2)
}

pub fn crop_box(frame: &BgrImage, d: &[f32; 5]) -> (i32, i32, i32, i32) {
    crop_box_pad(frame, d, 0.1, 0.125)
}

pub(crate) fn crop_slice(
    data: &[u8],
    width: u32,
    height: u32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
) -> BgrImage {
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = (x2.max(0) as u32).min(width);
    let y2 = (y2.max(0) as u32).min(height);
    let w = x2.saturating_sub(x1);
    let h = y2.saturating_sub(y1);
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in y1..y2 {
        if y >= height {
            break;
        }
        let s = ((y * width + x1) * 3) as usize;
        let n = (w * 3) as usize;
        if s + n <= data.len() {
            out.extend_from_slice(&data[s..s + n]);
        }
    }
    BgrImage {
        width: w,
        height: h,
        data: out,
    }
}

pub fn crop_img(im: &BgrImage, x1: i32, y1: i32, x2: i32, y2: i32) -> BgrImage {
    crop_slice(&im.data, im.width, im.height, x1, y1, x2, y2)
}

pub fn iou(a: &[f32; 5], b: &[f32; 4]) -> f32 {
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
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}
