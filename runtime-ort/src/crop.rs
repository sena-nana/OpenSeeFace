//! Next-frame crop: axis-aligned box from an eyes+nose similarity fit.
//!
//! The landmark CNN is trained on axis-aligned padded boxes, so in-plane
//! rotation is not applied to the crop. Scale and translation come from
//! Umeyama on points 36/39/42/45/30; size is a running max of the detector
//! box, the landmark hull, and the scaled FACE_3D template (grows, never
//! shrinks with jaw/mouth jitter), capped at [`GROW_CAP`] times the hull.
//! The hull covers the 66 model landmarks only — the gaze pupils appended to
//! `lms` are not face landmarks, and an outlier pupil must not blow up the
//! crop. A fit whose template box no longer covers the hull (foreshortened
//! eyes under strong yaw) falls back to the hull box. A parallel brows+nose
//! fit is used when the eye corners are pulled onto a glasses rim; weak refs
//! hold the previous box.

use crate::geom::{similarity_umeyama, xywh_iou, Similarity};
use crate::pnp::FACE_3D;

const MIN_CONF: f32 = 0.35;
const MIN_PTS: usize = 2;
const ALPHA_S: f32 = 0.25;
const ALPHA_T: f32 = 0.8;
const IOU_RESET: f32 = 0.3;
const JUMP_FRAC: f32 = 0.2;
/// Caps the grow-only size rule so one outlier frame cannot latch a giant crop.
const GROW_CAP: f32 = 1.5;
/// Minimum template-box coverage of the landmark hull for a fit to be
/// trusted. Healthy frames measure 0.72-0.94; the failing strong-yaw frame 0.42.
const COVERAGE_FRAC: f32 = 0.55;
const INTERIOR: std::ops::Range<usize> = 17..48;
const EYES_NOSE: [usize; 5] = [36, 39, 42, 45, 30];
/// Brows 17–26 + nose 27–35 (no lids / corners).
const BROWS_NOSE: std::ops::Range<usize> = 17..36;
/// Switch to brows+nose when implied eye corners diverge by this fraction of IOD.
/// 0.12 fires on a clean 3D face (different subsets); 0.55 still catches a ~48px rim pull.
const SHIFT_FRAC: f32 = 0.55;
const EYE_CORNERS: [usize; 4] = [36, 39, 42, 45];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CropSmoothState {
    ready: bool,
    scale: f32,
    tx: f32,
    ty: f32,
    last: Option<[f32; 4]>,
}

impl CropSmoothState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn last_box(&self) -> Option<[f32; 4]> {
        self.last
    }

    pub fn seed_size(&mut self, b: [f32; 4]) {
        if self.last.is_none() {
            self.last = Some(b);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CropTrack {
    state: CropSmoothState,
}

impl CropTrack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn seed_size(&mut self, b: [f32; 4]) {
        self.state.seed_size(b);
    }

    pub fn next_box(&mut self, pts: &[[f32; 3]], conf: f32) -> Option<[f32; 5]> {
        stable_landmark_bbox(pts, Some(&mut self.state)).map(|mut b| {
            b[4] = conf;
            b
        })
    }
}

fn template_xy(i: usize) -> Option<[f32; 2]> {
    FACE_3D.get(i).map(|p| [p[0], -p[1]])
}

fn push_pair(pts: &[[f32; 3]], i: usize, src: &mut Vec<[f32; 2]>, dst: &mut Vec<[f32; 2]>) {
    let Some(p) = pts.get(i) else { return };
    if p[2] < MIN_CONF {
        return;
    }
    let Some(t) = template_xy(i) else { return };
    src.push(t);
    dst.push([p[1], p[0]]);
}

fn collect_indices(pts: &[[f32; 3]], idx: &[usize]) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for &i in idx {
        push_pair(pts, i, &mut src, &mut dst);
    }
    (src, dst)
}

fn collect_range(
    pts: &[[f32; 3]],
    range: std::ops::Range<usize>,
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in range {
        push_pair(pts, i, &mut src, &mut dst);
    }
    (src, dst)
}

fn collect_fallback(pts: &[[f32; 3]]) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let interior = collect_range(pts, INTERIOR);
    if interior.0.len() >= MIN_PTS {
        return interior;
    }
    collect_range(pts, 0..pts.len().min(66))
}

fn fits_disagree(eyes: &Similarity, brows: &Similarity) -> bool {
    // Compare implied eye-corner locations, not raw (tx, scale): brows vs eyes
    // subsets have different centroids/scales even on a clean 3D face.
    let mut max_d = 0.0f32;
    let mut n = 0u32;
    for &i in &EYE_CORNERS {
        let Some(t) = template_xy(i) else { continue };
        let a = eyes.apply(t);
        let b = brows.apply(t);
        max_d = max_d.max((a[0] - b[0]).hypot(a[1] - b[1]));
        n += 1;
    }
    if n == 0 {
        return false;
    }
    let iod = match (template_xy(36), template_xy(45)) {
        (Some(l), Some(r)) => {
            let a = eyes.apply(l);
            let b = eyes.apply(r);
            (a[0] - b[0]).hypot(a[1] - b[1])
        }
        _ => eyes.scale.abs() * 2.0,
    };
    max_d > SHIFT_FRAC * iod.max(1.0)
}

fn aabb_pts(pts: &[[f32; 3]], range: std::ops::Range<usize>) -> Option<[f32; 4]> {
    let mut min_r = f32::MAX;
    let mut max_r = f32::MIN;
    let mut min_c = f32::MAX;
    let mut max_c = f32::MIN;
    let mut n = 0usize;
    for i in range {
        let Some(p) = pts.get(i) else { break };
        if p[2] < MIN_CONF {
            continue;
        }
        min_r = min_r.min(p[0]);
        max_r = max_r.max(p[0]);
        min_c = min_c.min(p[1]);
        max_c = max_c.max(p[1]);
        n += 1;
    }
    (n > 0).then_some([
        min_c,
        min_r,
        (max_c - min_c).max(1.0),
        (max_r - min_r).max(1.0),
    ])
}

fn place_box(tmpl: [f32; 4], last: Option<[f32; 4]>, hull: Option<[f32; 4]>) -> [f32; 4] {
    let cx = tmpl[0] + tmpl[2] * 0.5;
    let cy = tmpl[1] + tmpl[3] * 0.5;
    let mut w = tmpl[2].max(1.0);
    let mut h = tmpl[3].max(1.0);
    if let Some(p) = last {
        w = w.max(p[2]);
        h = h.max(p[3]);
    }
    if let Some(p) = hull {
        w = w.max(p[2]).min((p[2] * GROW_CAP).max(1.0));
        h = h.max(p[3]).min((p[3] * GROW_CAP).max(1.0));
    }
    [cx - w * 0.5, cy - h * 0.5, w, h]
}

fn pair_means(src: &[[f32; 2]], dst: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let n = src.len().min(dst.len()).max(1) as f32;
    let mut sm = [0.0; 2];
    let mut dm = [0.0; 2];
    for i in 0..src.len().min(dst.len()) {
        sm[0] += src[i][0];
        sm[1] += src[i][1];
        dm[0] += dst[i][0];
        dm[1] += dst[i][1];
    }
    ([sm[0] / n, sm[1] / n], [dm[0] / n, dm[1] / n])
}

fn axis_pose(scale: f32, src: &[[f32; 2]], dst: &[[f32; 2]]) -> Similarity {
    let s = scale.abs().max(1e-6);
    let (sm, dm) = pair_means(src, dst);
    Similarity {
        scale: s,
        theta: 0.0,
        tx: dm[0] - s * sm[0],
        ty: dm[1] - s * sm[1],
    }
}

fn pose_from_pairs(
    src: &[[f32; 2]],
    dst: &[[f32; 2]],
    prev: Option<Similarity>,
) -> Option<Similarity> {
    if src.len() < MIN_PTS {
        return None;
    }
    if let Some(sim) = similarity_umeyama(src, dst) {
        return Some(axis_pose(sim.scale, src, dst));
    }
    prev.map(|p| axis_pose(p.scale, src, dst))
}

fn fit_sim(pts: &[[f32; 3]], prev: Option<Similarity>) -> Option<Similarity> {
    let eyes = collect_indices(pts, &EYES_NOSE);
    let brows = collect_range(pts, BROWS_NOSE);
    let eyes_fit = pose_from_pairs(&eyes.0, &eyes.1, prev);
    let brows_fit = pose_from_pairs(&brows.0, &brows.1, prev);
    match (eyes_fit, brows_fit) {
        (Some(e), Some(b)) if fits_disagree(&e, &b) => Some(b),
        (Some(e), _) => Some(e),
        (None, Some(b)) => Some(b),
        (None, None) => {
            let (src, dst) = collect_fallback(pts);
            if src.len() >= MIN_PTS {
                if let Some(sim) = similarity_umeyama(&src, &dst) {
                    return Some(axis_pose(sim.scale, &src, &dst));
                }
            }
            if let Some(p) = prev {
                if !src.is_empty() {
                    return Some(axis_pose(p.scale, &src, &dst));
                }
                return Some(Similarity {
                    scale: p.scale,
                    theta: 0.0,
                    tx: p.tx,
                    ty: p.ty,
                });
            }
            None
        }
    }
}

fn mix(state: &mut CropSmoothState, raw: Similarity) -> Similarity {
    if !state.ready {
        state.ready = true;
        state.scale = raw.scale;
        state.tx = raw.tx;
        state.ty = raw.ty;
        return raw;
    }
    let jump = (raw.tx - state.tx).hypot(raw.ty - state.ty);
    let span = state
        .last
        .map(|b| b[2].max(b[3]))
        .unwrap_or(raw.scale.abs() * 2.0)
        .max(1.0);
    let follow = jump > JUMP_FRAC * span;
    let mixed = Similarity {
        scale: (1.0 - ALPHA_S) * state.scale + ALPHA_S * raw.scale,
        theta: 0.0,
        tx: if follow {
            raw.tx
        } else {
            (1.0 - ALPHA_T) * state.tx + ALPHA_T * raw.tx
        },
        ty: if follow {
            raw.ty
        } else {
            (1.0 - ALPHA_T) * state.ty + ALPHA_T * raw.ty
        },
    };
    let Some(raw_box) = raw.aabb((0..66).filter_map(template_xy)) else {
        return raw;
    };
    let Some(mix_box) = mixed.aabb((0..66).filter_map(template_xy)) else {
        return raw;
    };
    if xywh_iou(raw_box, mix_box) < IOU_RESET {
        state.scale = raw.scale;
        state.tx = raw.tx;
        state.ty = raw.ty;
        return raw;
    }
    state.scale = mixed.scale;
    state.tx = mixed.tx;
    state.ty = mixed.ty;
    mixed
}

fn hull_box(
    hull: Option<[f32; 4]>,
    pts: &[[f32; 3]],
    last: Option<[f32; 4]>,
    state: Option<&mut CropSmoothState>,
) -> Option<[f32; 5]> {
    // The full-landmark hull matches what the CNN is trained on; the interior
    // subset excludes mouth and chin, so it only serves when contour refs are
    // weak; with no confident refs at all the previous box is held.
    let box4 = hull
        .or_else(|| aabb_pts(pts, INTERIOR))
        .or(last)?;
    if let Some(st) = state {
        st.last = Some(box4);
    }
    Some([box4[0], box4[1], box4[2], box4[3], 1.0])
}

pub fn stable_landmark_bbox(
    pts: &[[f32; 3]],
    mut state: Option<&mut CropSmoothState>,
) -> Option<[f32; 5]> {
    let (prev, last) = match state.as_deref() {
        Some(s) if s.ready => (
            Some(Similarity {
                scale: s.scale,
                theta: 0.0,
                tx: s.tx,
                ty: s.ty,
            }),
            s.last,
        ),
        Some(s) => (None, s.last),
        None => (None, None),
    };
    let hull = aabb_pts(pts, 0..pts.len().min(66));
    let Some(raw) = fit_sim(pts, prev) else {
        return hull_box(hull, pts, last, state);
    };
    let sim = if let Some(st) = state.as_deref_mut() {
        mix(st, raw)
    } else {
        raw
    };
    let tmpl = sim.aabb((0..66).filter_map(template_xy))?;
    if hull.is_some_and(|hu| tmpl[2] < COVERAGE_FRAC * hu[2] || tmpl[3] < COVERAGE_FRAC * hu[3]) {
        // Foreshortened fit points (strong yaw) make the least-squares scale
        // collapse; the mapped template no longer covers the face.
        return hull_box(hull, pts, last, state);
    }
    let box4 = place_box(tmpl, last, hull);
    if let Some(st) = state {
        st.last = Some(box4);
    }
    Some([box4[0], box4[1], box4[2], box4[3], 1.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::landmark_bbox;

    fn synth_pts(scale: f32, tx: f32, ty: f32) -> Vec<[f32; 3]> {
        (0..66)
            .map(|i| {
                let t = template_xy(i).unwrap_or([0.0, 0.0]);
                let col = scale * t[0] + tx;
                let row = scale * t[1] + ty;
                [row, col, 1.0]
            })
            .collect()
    }

    #[test]
    fn stable_box_matches_hull_on_template_fit() {
        let pts = synth_pts(120.0, 400.0, 300.0);
        let hull = landmark_bbox(&pts).unwrap();
        let stable = stable_landmark_bbox(&pts, None).unwrap();
        for i in 0..4 {
            assert!(
                (hull[i] - stable[i]).abs() < 2.0,
                "i={i} hull={hull:?} stable={stable:?}"
            );
        }
        assert!(stable[2] + 1e-3 >= hull[2]);
    }

    #[test]
    fn stable_box_is_image_xywh() {
        let pts = synth_pts(80.0, 200.0, 150.0);
        let b = stable_landmark_bbox(&pts, None).unwrap();
        let mm = landmark_bbox(&pts).unwrap();
        let cx_b = b[0] + b[2] * 0.5;
        let cx_m = mm[0] + mm[2] * 0.5;
        assert!((cx_b - cx_m).abs() < 2.0, "x {} vs {}", cx_b, cx_m);
        let cy_b = b[1] + b[3] * 0.5;
        let cy_m = mm[1] + mm[3] * 0.5;
        assert!((cy_b - cy_m).abs() < 2.0, "y {} vs {}", cy_b, cy_m);
    }

    #[test]
    fn rotation_keeps_center_and_does_not_shrink() {
        let base = synth_pts(100.0, 320.0, 240.0);
        let mut st = CropSmoothState::default();
        let b0 = stable_landmark_bbox(&base, Some(&mut st)).unwrap();
        let (c, s) = (0.3f32.cos(), 0.3f32.sin());
        let mut rot = base.clone();
        for p in &mut rot {
            let x = p[1] - 320.0;
            let y = p[0] - 240.0;
            p[1] = 320.0 + c * x - s * y;
            p[0] = 240.0 + s * x + c * y;
        }
        let b1 = stable_landmark_bbox(&rot, Some(&mut st)).unwrap();
        let c0x = b0[0] + b0[2] * 0.5;
        let c0y = b0[1] + b0[3] * 0.5;
        let c1x = b1[0] + b1[2] * 0.5;
        let c1y = b1[1] + b1[3] * 0.5;
        assert!((c1x - c0x).hypot(c1y - c0y) < 20.0);
        assert!(b1[2] + 1e-3 >= b0[2]);
        assert!(b1[3] + 1e-3 >= b0[3]);
    }

    #[test]
    fn first_frame_size_not_smaller_than_hull() {
        let pts = synth_pts(100.0, 320.0, 240.0);
        let mut st = CropSmoothState::default();
        st.seed_size([300.0, 200.0, 40.0, 40.0]);
        let b = stable_landmark_bbox(&pts, Some(&mut st)).unwrap();
        let mm = landmark_bbox(&pts).unwrap();
        assert!(b[3] + 1e-3 >= mm[3], "h {} vs hull {}", b[3], mm[3]);
    }

    #[test]
    fn ema_damps_jitter() {
        let a = synth_pts(100.0, 320.0, 240.0);
        let mut st = CropSmoothState::default();
        let b0 = stable_landmark_bbox(&a, Some(&mut st)).unwrap();
        let b = synth_pts(100.0, 328.0, 244.0);
        let b1 = stable_landmark_bbox(&b, Some(&mut st)).unwrap();
        let jump = (b1[0] - b0[0]).abs();
        assert!(jump < 8.0 * 0.95, "smoothed dx={jump}");
        assert!(st.ready);
    }

    #[test]
    fn weak_eyes_keep_a_box() {
        let mut pts = synth_pts(90.0, 100.0, 80.0);
        for &i in EYES_NOSE.iter() {
            pts[i][2] = 0.1;
        }
        assert!(stable_landmark_bbox(&pts, None).is_some());
    }

    #[test]
    fn empty_refs_reuse_last_box() {
        let pts = synth_pts(90.0, 100.0, 80.0);
        let mut st = CropSmoothState::default();
        let first = stable_landmark_bbox(&pts, Some(&mut st)).unwrap();
        let dead = vec![[0.0, 0.0, 0.0]; 66];
        let again = stable_landmark_bbox(&dead, Some(&mut st)).unwrap();
        assert!((first[0] - again[0]).abs() < 1e-3);
        assert!((first[2] - again[2]).abs() < 1e-3);
    }

    #[test]
    fn ema_follows_large_jumps() {
        let a = synth_pts(100.0, 320.0, 240.0);
        let mut st = CropSmoothState::default();
        let _ = stable_landmark_bbox(&a, Some(&mut st)).unwrap();
        let b = synth_pts(100.0, 520.0, 240.0);
        let b1 = stable_landmark_bbox(&b, Some(&mut st)).unwrap();
        assert!(
            (b1[0] + b1[2] * 0.5 - 520.0).abs() < 40.0,
            "must follow a large translation, got {:?}",
            b1
        );
    }

    #[test]
    fn shifted_eye_corners_use_brows_nose() {
        let mut pts = synth_pts(100.0, 320.0, 240.0);
        for &i in &[36usize, 39, 42, 45] {
            pts[i][1] += 48.0;
        }
        let eyes = collect_indices(&pts, &EYES_NOSE);
        let brows = collect_range(&pts, BROWS_NOSE);
        let e = pose_from_pairs(&eyes.0, &eyes.1, None).unwrap();
        let b = pose_from_pairs(&brows.0, &brows.1, None).unwrap();
        assert!(fits_disagree(&e, &b), "eyes vs brows should disagree");
        let box4 = stable_landmark_bbox(&pts, None).unwrap();
        let cx = box4[0] + box4[2] * 0.5;
        assert!(
            (cx - 320.0).abs() < 12.0,
            "brows+nose should keep cx near 320, got {cx}"
        );
    }

    #[test]
    fn clean_face_keeps_eyes_nose() {
        let pts = synth_pts(100.0, 320.0, 240.0);
        let eyes = collect_indices(&pts, &EYES_NOSE);
        let brows = collect_range(&pts, BROWS_NOSE);
        let e = pose_from_pairs(&eyes.0, &eyes.1, None).unwrap();
        let b = pose_from_pairs(&brows.0, &brows.1, None).unwrap();
        assert!(!fits_disagree(&e, &b), "clean face fits should agree");
        let box4 = stable_landmark_bbox(&pts, None).unwrap();
        let cx = box4[0] + box4[2] * 0.5;
        assert!((cx - 320.0).abs() < 2.0, "cx={cx}");
    }

    /// Frame 1 of TrackingTest pose_20260916/20260916_162851 (three-quarter
    /// view, left eye foreshortened): pre-fix, `stable_landmark_bbox` turned
    /// these landmarks into an 836x614 crop (986x656 by the next frame) and
    /// the tracker dropped the face for ~11 of every 12 frames.
    const POSE_FRAME: [[f32; 3]; 66] = [
        [415.5, 605.2, 0.946], [453.5, 610.5, 1.026], [488.6, 619.8, 0.8145], [519.0, 631.5, 0.747],
        [551.2, 648.9, 0.8547], [575.3, 682.2, 0.8099], [591.9, 719.9, 0.865], [604.1, 761.6, 1.032],
        [613.7, 798.0, 0.8829], [607.0, 815.9, 0.9078], [597.9, 810.0, 0.9189], [588.3, 797.1, 0.8072],
        [564.7, 783.4, 0.6673], [533.7, 776.4, 0.7415], [505.3, 774.4, 0.6639], [472.1, 775.8, 0.6703],
        [431.7, 770.3, 0.759], [369.9, 705.3, 0.7804], [353.0, 725.2, 0.8029], [350.3, 745.7, 0.8414],
        [353.0, 765.9, 0.829], [359.8, 783.0, 0.8332], [368.1, 807.0, 0.9183], [365.6, 813.2, 0.9869],
        [365.0, 819.2, 0.7853], [366.9, 824.4, 0.9137], [379.7, 829.4, 0.8079], [396.5, 803.7, 0.9218],
        [415.1, 814.8, 0.8224], [433.0, 828.2, 0.9627], [452.6, 832.4, 0.9109], [474.5, 799.6, 0.8754],
        [475.2, 809.0, 1.006], [477.5, 819.1, 1.009], [476.5, 824.1, 0.974], [476.4, 824.5, 0.9637],
        [397.7, 726.0, 0.9464], [392.9, 739.2, 0.7869], [392.8, 761.4, 0.9961], [399.6, 767.6, 0.9049],
        [401.4, 760.8, 1.001], [401.7, 738.9, 0.8556], [404.3, 810.6, 0.8921], [401.0, 817.0, 0.9634],
        [402.8, 828.2, 0.7817], [407.4, 830.2, 0.681], [409.4, 828.5, 0.7639], [408.1, 817.9, 0.9759],
        [511.3, 799.0, 0.8829], [503.8, 819.8, 0.876], [505.0, 826.5, 0.9141], [505.8, 832.2, 1.004],
        [515.6, 835.8, 1.001], [533.0, 835.1, 1.017], [536.7, 832.0, 0.8131], [539.4, 825.9, 0.8902],
        [538.4, 809.8, 0.8485], [531.9, 794.3, 0.9864], [525.2, 777.1, 1.009], [517.4, 805.3, 0.858],
        [517.9, 826.1, 0.9314], [519.3, 830.3, 0.6521], [529.0, 834.0, 0.9511], [521.9, 828.8, 0.7534],
        [523.7, 823.3, 0.8733], [522.1, 802.8, 0.9016],
    ];

    #[test]
    fn yaw_degenerate_fit_stays_bounded() {
        let mut st = CropSmoothState::default();
        st.seed_size([605.0, 350.0, 231.0, 264.0]);
        let b = stable_landmark_bbox(&POSE_FRAME, Some(&mut st)).unwrap();
        let hull = landmark_bbox(&POSE_FRAME).unwrap();
        assert!(
            b[2] <= hull[2] * GROW_CAP + 2.0 && b[3] <= hull[3] * GROW_CAP + 2.0,
            "box {b:?} must stay within GROW_CAP of hull {hull:?}"
        );
        // The crop must still cover the eye region, not collapse to a sliver.
        assert!(b[2] > hull[2] * 0.6 && b[3] > hull[3] * 0.6, "box {b:?} vs hull {hull:?}");
        // And a second call (the frame actually fed to the CNN) stays bounded too.
        let b2 = stable_landmark_bbox(&POSE_FRAME, Some(&mut st)).unwrap();
        assert!(
            b2[2] <= hull[2] * GROW_CAP + 2.0 && b2[3] <= hull[3] * GROW_CAP + 2.0,
            "second box {b2:?} must stay within GROW_CAP of hull {hull:?}"
        );
    }

    #[test]
    fn grow_only_rule_is_capped_by_hull() {
        let pts = synth_pts(100.0, 320.0, 240.0);
        let mut st = CropSmoothState::default();
        st.last = Some([100.0, 50.0, 900.0, 700.0]);
        let b = stable_landmark_bbox(&pts, Some(&mut st)).unwrap();
        let hull = landmark_bbox(&pts).unwrap();
        assert!(
            b[2] <= hull[2] * GROW_CAP + 2.0 && b[3] <= hull[3] * GROW_CAP + 2.0,
            "box {b:?} must be capped by hull {hull:?}"
        );
    }
}
