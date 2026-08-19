//! Gaze / pupil from `mnv3_gaze32_split_opt.onnx`.

use anyhow::Result;
use half::f16;

use crate::geom::{clamp_to_im, compensate, logit, rotate};
use crate::preprocess::{crop_img, imagenet_nchw_into, BgrImage};
use crate::session::OrtModel;

fn extract_face(frame: &BgrImage, lms: &[[f32; 3]]) -> (BgrImage, Vec<[f32; 2]>, [f32; 2]) {
    let mut xy: Vec<[f32; 2]> = lms.iter().map(|p| [p[1], p[0]]).collect();
    let mut x1 = f32::MAX;
    let mut y1 = f32::MAX;
    let mut x2 = f32::MIN;
    let mut y2 = f32::MIN;
    for p in &xy {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    let radius_x = 1.2 * (x2 - x1) / 2.0;
    let radius_y = 1.2 * (y2 - y1) / 2.0;
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let (x1, y1) = clamp_to_im(
        cx - radius_x,
        cy - radius_y,
        frame.height as f32,
        frame.width as f32,
    );
    let (x2, y2) = clamp_to_im(
        cx + radius_x + 1.0,
        cy + radius_y + 1.0,
        frame.height as f32,
        frame.width as f32,
    );
    let offset = [x1 as f32, y1 as f32];
    for p in xy.iter_mut() {
        p[0] -= offset[0];
        p[1] -= offset[1];
    }
    let crop = crop_img(frame, x1, y1, x2, y2);
    (crop, xy, offset)
}

struct EyePrep {
    x1: f32,
    y1: f32,
    w: u32,
    h: u32,
    scale: [f32; 2],
    reference: [f32; 2],
    angle: f32,
    flip: bool,
    glare: bool,
}

fn prepare_eye(face: &BgrImage, corners: [[f32; 2]; 2], flip: bool) -> Option<EyePrep> {
    let c1 = corners[0];
    let (c2, a) = compensate(c1, corners[1]);
    let center = [(c1[0] + c2.0) / 2.0, (c1[1] + c2.1) / 2.0];
    let radius = ((c1[0] - c2.0).hypot(c1[1] - c2.1) / 2.0).max(1.0);
    let w = face.width as f32;
    let h = face.height as f32;
    let probe_rx = radius * 1.4;
    let probe_ry = radius * 1.2;
    let glare = crate::glasses::region_glare_frac(
        face,
        center[0] - probe_rx,
        center[1] - probe_ry,
        center[0] + probe_rx,
        center[1] + probe_ry,
    ) > crate::glasses::GLARE_FRAC_THRESH;
    let (px, py) = if glare { (1.65, 1.45) } else { (1.4, 1.2) };
    let rx = radius * px;
    let ry = radius * py;
    let (x1, y1) = clamp_to_im(center[0] - rx, center[1] - ry, w, h);
    let (x2, y2) = clamp_to_im(center[0] + rx, center[1] + ry, w, h);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    Some(EyePrep {
        x1: x1 as f32,
        y1: y1 as f32,
        w: (x2 - x1) as u32,
        h: (y2 - y1) as u32,
        scale: [(x2 - x1) as f32 / 32.0, (y2 - y1) as f32 / 32.0],
        reference: c1,
        angle: a,
        flip,
        glare,
    })
}

/// Inverse of `rotate_about` over the eye box only, then ImageNet NCHW at 32×32.
fn eye_nchw_into(face: &BgrImage, p: &EyePrep, dst: &mut [f16]) {
    let (cw, ch) = (p.w, p.h);
    if cw < 2 || ch < 2 {
        dst.fill(f16::ZERO);
        return;
    }
    let (cos, sin) = (p.angle.cos(), p.angle.sin());
    let (cx, cy) = (p.reference[0], p.reference[1]);
    let (ix1, iy1) = (p.x1 as i32, p.y1 as i32);
    let mut crop = BgrImage::zeros(cw, ch);
    for y in 0..ch as i32 {
        for x in 0..cw as i32 {
            let dx = (ix1 + x) as f32 - cx;
            let dy = (iy1 + y) as f32 - cy;
            let pix = face.sample(cos * dx + sin * dy + cx, -sin * dx + cos * dy + cy);
            let i = ((y as u32 * cw + x as u32) * 3) as usize;
            crop.data[i..i + 3].copy_from_slice(&pix);
        }
    }
    if p.flip {
        crop.flip_h_in_place();
    }
    if p.glare || crate::glasses::glare_frac(&crop) > crate::glasses::GLARE_FRAC_THRESH {
        crate::glasses::suppress_glare(&mut crop);
    }
    imagenet_nchw_into(&crop, 32, dst);
}

/// Returns `[open, row, col, conf]` per eye (right, left), matching Python `eye_state`.
pub fn get_eye_state(
    gaze: &mut OrtModel,
    frame: &BgrImage,
    lms: &[[f32; 3]],
    no_gaze: bool,
) -> Result<[[f32; 4]; 2]> {
    let dummy = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    if no_gaze || lms.len() < 46 {
        return Ok(dummy);
    }
    let (face, xy, offset) = extract_face(frame, lms);
    if xy.len() < 46 {
        return Ok(dummy);
    }
    let right = prepare_eye(&face, [xy[36], xy[39]], false);
    let left = prepare_eye(&face, [xy[42], xy[45]], true);
    let (Some(right), Some(left)) = (right, left) else {
        return Ok(dummy);
    };
    let plane = 3 * 32 * 32;
    Ok(gaze
        .run_prep(
            &[2, 3, 32, 32],
            |buf| {
                let n = plane.min(buf.len() / 2);
                eye_nchw_into(&face, &right, &mut buf[..n]);
                eye_nchw_into(&face, &left, &mut buf[n..n + n]);
            },
            |outs| Ok(decode_eyes(outs[0], &right, &left, offset, dummy)),
        )
        .unwrap_or(dummy))
}

fn decode_eyes(
    raw: &[f16],
    right: &EyePrep,
    left: &EyePrep,
    offset: [f32; 2],
    dummy: [[f32; 4]; 2],
) -> [[f32; 4]; 2] {
    let data: Vec<f32> = raw.iter().map(|x| x.to_f32()).collect();
    let cells = 8 * 8;
    let per_eye = 3 * cells;
    let mut state = dummy;
    let preps = [right, left];
    for i in 0..2 {
        let base = i * per_eye;
        if base + per_eye > data.len() {
            let n = data.len() / 2;
            let ebase = i * n;
            if ebase + 3 * cells > data.len() {
                continue;
            }
        }
        let hm = &data[base..base + cells];
        let (m, conf) = hm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let x = m / 8;
        let y = m % 8;
        let off_x = 32.0 * logit(data[base + cells + m], 8.0);
        let off_y = 32.0 * logit(data[base + 2 * cells + m], 8.0);
        let mut eye_x = 32.0 * x as f32 / 8.0 + off_x;
        let mut eye_y = 32.0 * y as f32 / 8.0 + off_y;
        let p = preps[i];
        if i == 0 {
            eye_x = p.x1 + p.scale[0] * eye_x;
        } else {
            eye_x = p.x1 + p.scale[0] * (32.0 - eye_x);
        }
        eye_y = p.y1 + p.scale[1] * eye_y;
        let (eye_x, eye_y) = rotate((p.reference[0], p.reference[1]), (eye_x, eye_y), -p.angle);
        let eye_x = eye_x + offset[0];
        let eye_y = eye_y + offset[1];
        state[i] = [1.0, eye_y, eye_x, *conf];
        if state[i].iter().any(|v| v.is_nan()) {
            state[i] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    state
}
