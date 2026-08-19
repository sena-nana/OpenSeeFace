//! CPU illumination preprocess: gray-world WB, AHE, and CLAHE on BT.601 Y.

use crate::preprocess::BgrImage;

const HIST: usize = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum HeMode {
    Off,
    Ahe,
    #[default]
    Clahe,
}

impl HeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ahe => "ahe",
            Self::Clahe => "clahe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneKind {
    WellLit,
    Dark,
    Overexp,
    Backlight,
    Noisy,
    Cast,
}

/// `--enhance` uses [`EnhanceCfg::auto`]: CLAHE on dark, AHE on backlight,
/// denoise+CLAHE on noise, WB on cast, identity on well-lit.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct EnhanceCfg {
    pub wb: bool,
    pub he: HeMode,
    pub clip_limit: f32,
    pub tiles: u32,
    pub blend: f32,
    pub auto: bool,
    pub denoise: bool,
}

impl Default for EnhanceCfg {
    fn default() -> Self {
        Self::clahe()
    }
}

impl EnhanceCfg {
    pub fn off() -> Self {
        Self {
            wb: false,
            he: HeMode::Off,
            clip_limit: 2.0,
            tiles: 8,
            blend: 1.0,
            auto: false,
            denoise: false,
        }
    }

    pub fn clahe() -> Self {
        Self {
            he: HeMode::Clahe,
            clip_limit: 4.0,
            blend: 0.7,
            ..Self::off()
        }
    }

    pub fn auto() -> Self {
        Self {
            auto: true,
            ..Self::clahe()
        }
    }

    pub fn is_off(self) -> bool {
        !self.auto && !self.wb && self.he == HeMode::Off && !self.denoise
    }

    fn ahe(self) -> bool {
        matches!(self.he, HeMode::Ahe) || (self.he == HeMode::Clahe && self.clip_limit <= 0.0)
    }

    pub fn resolve(self, img: &BgrImage) -> Self {
        if self.auto {
            policy(classify(&analyze(img)))
        } else {
            self
        }
    }
}

#[inline]
pub fn luma_bt601(b: u8, g: u8, r: u8) -> f32 {
    0.114 * b as f32 + 0.587 * g as f32 + 0.299 * r as f32
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

pub fn enhance_bgr(src: &BgrImage, cfg: &EnhanceCfg) -> BgrImage {
    let mut out = src.clone();
    enhance_bgr_in_place(&mut out, cfg);
    out
}

pub fn enhance_bgr_in_place(img: &mut BgrImage, cfg: &EnhanceCfg) {
    let cfg = cfg.resolve(img);
    if cfg.is_off() || img.width == 0 || img.height == 0 {
        return;
    }
    let orig = if cfg.blend < 1.0 {
        img.data.clone()
    } else {
        Vec::new()
    };
    if cfg.denoise {
        box3_bgr(&mut img.data, img.width, img.height);
    }
    if cfg.wb {
        gray_world(&mut img.data);
    }
    if cfg.he != HeMode::Off {
        clahe_y(&mut img.data, img.width, img.height, &cfg);
    }
    if cfg.blend <= 0.0 {
        img.data.copy_from_slice(&orig);
        return;
    }
    if cfg.blend < 1.0 {
        crate::simd::blend_u8(&mut img.data, &orig, cfg.blend.clamp(0.0, 1.0));
    }
}

struct SceneStats {
    center_y: f32,
    surround_y: f32,
    shadow_frac: f32,
    highlight_frac: f32,
    noise: f32,
    chroma: f32,
}

fn analyze(img: &BgrImage) -> SceneStats {
    let w = img.width.max(1);
    let h = img.height.max(1);
    let step = ((w.min(h) / 160).max(1)) as usize;
    let side = ((h.min(w) as f32) * 0.22).max(8.0) as u32;
    let x0 = w.saturating_sub(side) / 2;
    let y0 = h.saturating_sub(side) / 2;
    let x1 = (x0 + side).min(w);
    let y1 = (y0 + side).min(h);
    let bx = (w / 7).max(1);
    let by = (h / 7).max(1);

    let mut c_sum = 0.0f64;
    let mut c_n = 0u32;
    let mut s_sum = 0.0f64;
    let mut s_n = 0u32;
    let mut shadows = 0u32;
    let mut highs = 0u32;
    let mut noise = 0.0f64;
    let mut noise_n = 0u32;
    let mut sb = 0.0f64;
    let mut sg = 0.0f64;
    let mut sr = 0.0f64;
    let mut chroma_n = 0u32;

    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            let i = ((y * w + x) * 3) as usize;
            let b = img.data[i];
            let g = img.data[i + 1];
            let r = img.data[i + 2];
            let yv = luma_bt601(b, g, r);
            let in_c = x >= x0 && x < x1 && y >= y0 && y < y1;
            let in_surr = x < bx || x >= w - bx || y < by || y >= h - by;
            if in_c {
                c_sum += yv as f64;
                c_n += 1;
                if yv < 45.0 {
                    shadows += 1;
                }
                if yv > 225.0 {
                    highs += 1;
                }
                if yv > 28.0 {
                    sb += b as f64;
                    sg += g as f64;
                    sr += r as f64;
                    chroma_n += 1;
                }
            }
            if in_surr {
                s_sum += yv as f64;
                s_n += 1;
            }
            if in_c && x + (step as u32) < w && y + (step as u32) < h {
                let ix = ((y * w + x + step as u32) * 3) as usize;
                let iy = (((y + step as u32) * w + x) * 3) as usize;
                let yx = luma_bt601(img.data[ix], img.data[ix + 1], img.data[ix + 2]);
                let yy = luma_bt601(img.data[iy], img.data[iy + 1], img.data[iy + 2]);
                noise += (yv - yx).abs() as f64 + (yv - yy).abs() as f64;
                noise_n += 2;
            }
        }
    }
    let center_y = if c_n > 0 {
        (c_sum / c_n as f64) as f32
    } else {
        0.0
    };
    let surround_y = if s_n > 0 {
        (s_sum / s_n as f64) as f32
    } else {
        0.0
    };
    let inv = if c_n > 0 { 1.0 / c_n as f32 } else { 0.0 };
    let cin = if chroma_n > 0 {
        1.0 / chroma_n as f32
    } else {
        0.0
    };
    let mb = (sb as f32) * cin;
    let mg = (sg as f32) * cin;
    let mr = (sr as f32) * cin;
    let gray = (mb + mg + mr) / 3.0 + 1e-3;
    SceneStats {
        center_y,
        surround_y,
        shadow_frac: shadows as f32 * inv,
        highlight_frac: highs as f32 * inv,
        noise: if noise_n > 0 {
            (noise / noise_n as f64) as f32
        } else {
            0.0
        },
        chroma: (mb - mg).abs().max((mg - mr).abs()).max((mb - mr).abs()) / gray,
    }
}

fn classify(s: &SceneStats) -> SceneKind {
    if s.surround_y > s.center_y + 18.0 && s.surround_y > 100.0 && s.center_y < 130.0 {
        SceneKind::Backlight
    } else if s.highlight_frac > 0.16 || s.center_y > 185.0 {
        SceneKind::Overexp
    } else if s.noise > 18.0 && s.center_y > 75.0 {
        SceneKind::Noisy
    } else if s.center_y < 72.0 || s.shadow_frac > 0.5 {
        SceneKind::Dark
    } else if s.chroma > 0.35 && s.center_y > 75.0 {
        SceneKind::Cast
    } else {
        SceneKind::WellLit
    }
}

fn policy(kind: SceneKind) -> EnhanceCfg {
    match kind {
        SceneKind::WellLit => EnhanceCfg::off(),
        SceneKind::Dark => EnhanceCfg::clahe(),
        SceneKind::Backlight | SceneKind::Overexp => EnhanceCfg {
            he: HeMode::Ahe,
            blend: 1.0,
            ..EnhanceCfg::off()
        },
        SceneKind::Noisy => EnhanceCfg {
            denoise: true,
            he: HeMode::Clahe,
            clip_limit: 2.0,
            blend: 0.55,
            ..EnhanceCfg::off()
        },
        SceneKind::Cast => EnhanceCfg {
            wb: true,
            ..EnhanceCfg::off()
        },
    }
}

fn box3_bgr(data: &mut [u8], width: u32, height: u32) {
    crate::simd::box3_bgr(data, width, height);
}

fn gray_world(bgr: &mut [u8]) {
    crate::simd::gray_world(bgr);
}

pub(crate) fn tile_grid(width: u32, height: u32, tiles: u32) -> (u32, u32, u32, u32) {
    let tx = tiles.min(width).max(1);
    let ty = tiles.min(height).max(1);
    (tx, ty, (width / tx).max(1), (height / ty).max(1))
}

fn tile_of(x: u32, y: u32, tw: u32, th: u32, tx: u32, ty: u32) -> (u32, u32) {
    ((x / tw).min(tx - 1), (y / th).min(ty - 1))
}

fn tile_rect(
    txi: u32,
    tyi: u32,
    tw: u32,
    th: u32,
    tx: u32,
    ty: u32,
    width: u32,
    height: u32,
) -> (u32, u32, u32, u32) {
    let x1 = txi * tw;
    let y1 = tyi * th;
    let x2 = if txi + 1 == tx { width } else { (txi + 1) * tw };
    let y2 = if tyi + 1 == ty {
        height
    } else {
        (tyi + 1) * th
    };
    (x1, y1, x2, y2)
}

fn clip_hist(hist: &mut [u32; HIST], clip_limit: u32) {
    let mut clipped = 0u32;
    for h in hist.iter_mut() {
        if *h > clip_limit {
            clipped += *h - clip_limit;
            *h = clip_limit;
        }
    }
    let batch = clipped / HIST as u32;
    let mut residual = clipped - batch * HIST as u32;
    for h in hist.iter_mut() {
        *h += batch;
    }
    let mut i = 0usize;
    while residual > 0 {
        hist[i] += 1;
        residual -= 1;
        i = (i + 1) % HIST;
    }
}

fn lut_from_hist(hist: &[u32; HIST], tile_n: u32) -> [u8; HIST] {
    let scale = 255.0 / tile_n.max(1) as f32;
    let mut sum = 0u32;
    let mut lut = [0u8; HIST];
    for (i, &h) in hist.iter().enumerate() {
        sum += h;
        lut[i] = clamp_u8(sum as f32 * scale);
    }
    lut
}

/// Build per-tile 256-entry LUTs (OpenCV-style clip + CDF).
pub(crate) fn clahe_luts(
    bgr: &[u8],
    width: u32,
    height: u32,
    cfg: &EnhanceCfg,
) -> (u32, u32, u32, u32, Vec<[u8; HIST]>) {
    let (tx, ty, tw, th) = tile_grid(width, height, cfg.tiles);
    let n_tiles = (tx * ty) as usize;
    let mut hists = vec![[0u32; HIST]; n_tiles];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let yv = luma_bt601(bgr[i], bgr[i + 1], bgr[i + 2]);
            let bin = yv.round().clamp(0.0, 255.0) as usize;
            let (txi, tyi) = tile_of(x, y, tw, th, tx, ty);
            hists[(tyi * tx + txi) as usize][bin] += 1;
        }
    }
    let ahe = cfg.ahe();
    let mut luts = vec![[0u8; HIST]; n_tiles];
    for tyi in 0..ty {
        for txi in 0..tx {
            let (x1, y1, x2, y2) = tile_rect(txi, tyi, tw, th, tx, ty, width, height);
            let tile_n = (x2 - x1) * (y2 - y1);
            let idx = (tyi * tx + txi) as usize;
            if !ahe {
                let clip = ((cfg.clip_limit * tile_n as f32) / HIST as f32)
                    .floor()
                    .max(1.0) as u32;
                clip_hist(&mut hists[idx], clip);
            }
            luts[idx] = lut_from_hist(&hists[idx], tile_n);
        }
    }
    (tx, ty, tw, th, luts)
}

fn clahe_y(bgr: &mut [u8], width: u32, height: u32, cfg: &EnhanceCfg) {
    let (tx, ty, tw, th, luts) = clahe_luts(bgr, width, height, cfg);
    crate::simd::clahe_remap(bgr, width, height, tx, ty, tw, th, &luts);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, b: u8, g: u8, r: u8) -> BgrImage {
        BgrImage {
            width: w,
            height: h,
            data: vec![b, g, r].repeat((w * h) as usize),
        }
    }

    fn mean_y(img: &BgrImage) -> f32 {
        let n = (img.width * img.height) as f32;
        img.data
            .chunks_exact(3)
            .map(|p| luma_bt601(p[0], p[1], p[2]))
            .sum::<f32>()
            / n
    }

    fn std_y(img: &BgrImage) -> f32 {
        let n = (img.width * img.height) as f32;
        let m = mean_y(img);
        let var = img
            .data
            .chunks_exact(3)
            .map(|p| {
                let d = luma_bt601(p[0], p[1], p[2]) - m;
                d * d
            })
            .sum::<f32>()
            / n;
        var.sqrt()
    }

    #[test]
    fn gray_world_leaves_neutral_gray() {
        let src = solid(8, 8, 120, 120, 120);
        let cfg = EnhanceCfg {
            wb: true,
            ..EnhanceCfg::off()
        };
        let got = enhance_bgr(&src, &cfg);
        assert_eq!(got.data, src.data);
    }

    #[test]
    fn gray_world_corrects_red_cast() {
        let src = solid(16, 16, 80, 80, 160);
        let cfg = EnhanceCfg {
            wb: true,
            ..EnhanceCfg::off()
        };
        let got = enhance_bgr(&src, &cfg);
        let (mut mb, mut mg, mut mr) = (0.0, 0.0, 0.0);
        let n = (got.width * got.height) as f32;
        for p in got.data.chunks_exact(3) {
            mb += p[0] as f32;
            mg += p[1] as f32;
            mr += p[2] as f32;
        }
        mb /= n;
        mg /= n;
        mr /= n;
        assert!((mb - mg).abs() < 2.0, "{mb} {mg} {mr}");
        assert!((mg - mr).abs() < 2.0, "{mb} {mg} {mr}");
    }

    #[test]
    fn off_is_identity() {
        let src = solid(4, 4, 10, 20, 30);
        let got = enhance_bgr(&src, &EnhanceCfg::off());
        assert_eq!(got.data, src.data);
    }

    fn dark_ramp(w: u32, h: u32) -> BgrImage {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = (20 + (x + y) * 40 / (w + h).max(1)).min(80) as u8;
                data.extend_from_slice(&[v, v, v]);
            }
        }
        BgrImage {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn clahe_raises_luma_std_on_dark_ramp() {
        let src = dark_ramp(64, 64);
        let cfg = EnhanceCfg {
            he: HeMode::Clahe,
            clip_limit: 2.0,
            blend: 1.0,
            ..EnhanceCfg::off()
        };
        let got = enhance_bgr(&src, &cfg);
        assert!(
            std_y(&got) > std_y(&src) + 5.0,
            "std {} -> {}",
            std_y(&src),
            std_y(&got)
        );
    }

    fn high_contrast_block(w: u32, h: u32) -> BgrImage {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / 4) + (y / 4)) % 2 == 0 {
                    10
                } else {
                    240
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        BgrImage {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn clahe_clips_more_than_ahe() {
        let src = high_contrast_block(64, 64);
        let ahe = enhance_bgr(
            &src,
            &EnhanceCfg {
                he: HeMode::Ahe,
                blend: 1.0,
                ..EnhanceCfg::off()
            },
        );
        let clahe = enhance_bgr(
            &src,
            &EnhanceCfg {
                he: HeMode::Clahe,
                clip_limit: 2.0,
                blend: 1.0,
                ..EnhanceCfg::off()
            },
        );
        let peak = |img: &BgrImage| {
            img.data
                .chunks_exact(3)
                .zip(src.data.chunks_exact(3))
                .map(|(a, b)| (luma_bt601(a[0], a[1], a[2]) - luma_bt601(b[0], b[1], b[2])).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            peak(&clahe) <= peak(&ahe) + 1.0,
            "clahe {} ahe {}",
            peak(&clahe),
            peak(&ahe)
        );
    }

    #[test]
    fn clip_limit_zero_matches_ahe() {
        let src = dark_ramp(32, 32);
        let a = enhance_bgr(
            &src,
            &EnhanceCfg {
                he: HeMode::Ahe,
                tiles: 4,
                blend: 1.0,
                ..EnhanceCfg::off()
            },
        );
        let b = enhance_bgr(
            &src,
            &EnhanceCfg {
                he: HeMode::Clahe,
                clip_limit: 0.0,
                tiles: 4,
                blend: 1.0,
                ..EnhanceCfg::off()
            },
        );
        assert_eq!(a.data, b.data);
    }

    fn fill(w: u32, h: u32, b: u8, g: u8, r: u8) -> BgrImage {
        BgrImage {
            width: w,
            height: h,
            data: vec![b, g, r].repeat((w * h) as usize),
        }
    }

    fn paint_rect(img: &mut BgrImage, x0: u32, y0: u32, x1: u32, y1: u32, v: u8) {
        for y in y0..y1.min(img.height) {
            for x in x0..x1.min(img.width) {
                let i = ((y * img.width + x) * 3) as usize;
                img.data[i] = v;
                img.data[i + 1] = v;
                img.data[i + 2] = v;
            }
        }
    }

    #[test]
    fn classifies_dark_overexp_backlight_noise() {
        assert_eq!(
            classify(&analyze(&fill(80, 80, 30, 30, 30))),
            SceneKind::Dark
        );
        assert_eq!(
            classify(&analyze(&fill(80, 80, 240, 240, 240))),
            SceneKind::Overexp
        );

        let mut back = fill(80, 80, 220, 220, 220);
        paint_rect(&mut back, 20, 20, 60, 60, 40);
        assert_eq!(classify(&analyze(&back)), SceneKind::Backlight);

        let mut noisy = fill(80, 80, 128, 128, 128);
        let mut s = 0x9e37_79b9u32;
        for p in &mut noisy.data {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s as i32 as f32 / 2.147e9) * 48.0;
            *p = (*p as f32 + n).round().clamp(0.0, 255.0) as u8;
        }
        assert_eq!(classify(&analyze(&noisy)), SceneKind::Noisy);

        assert_eq!(
            classify(&analyze(&fill(80, 80, 80, 80, 170))),
            SceneKind::Cast
        );
        assert_eq!(
            classify(&analyze(&fill(80, 80, 140, 140, 140))),
            SceneKind::WellLit
        );
    }

    #[test]
    fn auto_skips_well_lit() {
        let src = fill(32, 32, 140, 140, 140);
        let got = enhance_bgr(&src, &EnhanceCfg::auto());
        assert_eq!(got.data, src.data);
    }
}
