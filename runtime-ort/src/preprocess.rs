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

fn apply_lut(src_bgr: &[u8], n: usize, norm: &ColorNorm, data: &mut [f16]) {
    let src = norm.src;
    let lut = &norm.lut;
    for i in 0..n {
        let p = i * 3;
        data[i] = lut[0][src_bgr[p + src[0]] as usize];
        data[n + i] = lut[1][src_bgr[p + src[1]] as usize];
        data[2 * n + i] = lut[2][src_bgr[p + src[2]] as usize];
    }
}

/// Fused bilinear resize + BGR/RGB remap + baked mean/std → f16 NCHW.
pub fn nchw(bgr: &BgrImage, width: u32, height: u32, norm: &ColorNorm) -> TensorF16 {
    let n = (width * height) as usize;
    let mut data = vec![f16::ZERO; 3 * n];
    let src = norm.src;
    let lut = &norm.lut;
    if width == bgr.width && height == bgr.height {
        apply_lut(&bgr.data, n, norm, &mut data);
        return TensorF16 {
            shape: vec![1, 3, height as i64, width as i64],
            data,
        };
    }
    let fx = bgr.width as f32 / width as f32;
    let fy = bgr.height as f32 / height as f32;
    for y in 0..height {
        let sy = (y as f32 + 0.5) * fy - 0.5;
        let y0 = sy.floor().clamp(0.0, (bgr.height - 1) as f32) as u32;
        let y1 = (y0 + 1).min(bgr.height - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..width {
            let sx = (x as f32 + 0.5) * fx - 0.5;
            let x0 = sx.floor().clamp(0.0, (bgr.width - 1) as f32) as u32;
            let x1 = (x0 + 1).min(bgr.width - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i00 = ((y0 * bgr.width + x0) * 3) as usize;
            let i10 = ((y0 * bgr.width + x1) * 3) as usize;
            let i01 = ((y1 * bgr.width + x0) * 3) as usize;
            let i11 = ((y1 * bgr.width + x1) * 3) as usize;
            let o = (y * width + x) as usize;
            for c in 0..3 {
                let s = src[c];
                let v = bgr.data[i00 + s] as f32 * (1.0 - wx) * (1.0 - wy)
                    + bgr.data[i10 + s] as f32 * wx * (1.0 - wy)
                    + bgr.data[i01 + s] as f32 * (1.0 - wx) * wy
                    + bgr.data[i11 + s] as f32 * wx * wy;
                let u = v.round().clamp(0.0, 255.0) as u8 as usize;
                data[c * n + o] = lut[c][u];
            }
        }
    }
    TensorF16 {
        shape: vec![1, 3, height as i64, width as i64],
        data,
    }
}

pub fn imagenet_nchw(bgr: &BgrImage, size: u32) -> TensorF16 {
    nchw(bgr, size, size, &ColorNorm::IMAGENET)
}

pub fn retina_nchw(bgr: &BgrImage) -> TensorF16 {
    nchw(bgr, 640, 640, &ColorNorm::RETINA)
}

/// Padded face crop in image coordinates, matching the Python tracker.
pub fn crop_box(frame: &BgrImage, d: &[f32; 5]) -> (i32, i32, i32, i32) {
    let (x, y, w, h) = (d[0], d[1], d[2], d[3]);
    let x1 = x - (w * 0.1) as i32 as f32;
    let y1 = y - (h * 0.125) as i32 as f32;
    let x2 = x + w + (w * 0.1) as i32 as f32;
    let y2 = y + h + (h * 0.125) as i32 as f32;
    let clamp = |px: f32, py: f32| {
        let px = px.clamp(0.0, frame.width as f32 - 1.0) as i32;
        let py = py.clamp(0.0, frame.height as f32 - 1.0) as i32 + 1;
        (px, py)
    };
    let (x1, y1) = clamp(x1, y1);
    let (x2, y2) = clamp(x2, y2);
    (x1, y1, x2, y2)
}

pub fn crop_img(im: &BgrImage, x1: i32, y1: i32, x2: i32, y2: i32) -> BgrImage {
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = (x2.max(0) as u32).min(im.width);
    let y2 = (y2.max(0) as u32).min(im.height);
    let w = x2.saturating_sub(x1);
    let h = y2.saturating_sub(y1);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in y1..y2 {
        let s = ((y * im.width + x1) * 3) as usize;
        data.extend_from_slice(&im.data[s..s + (w * 3) as usize]);
    }
    BgrImage {
        width: w,
        height: h,
        data,
    }
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
