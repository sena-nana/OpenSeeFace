//! Canonical face output: derived params once, protocol adapters only convert.

use std::time::Instant;

use crate::ext::{cats, winning, ExtState, SIL, VISEME_COUNT};
use crate::features::{
    face_norm_basis, FeatureVec, FEAT_BROW_UD_L, FEAT_BROW_UD_R, FEAT_CORNER_UD_L,
    FEAT_CORNER_UD_R, FEAT_JAW_OPEN, FEAT_MOUTH_FUNNEL, FEAT_MOUTH_OPEN, FEAT_MOUTH_PUCKER,
    FEAT_MOUTH_WIDE,
};
use crate::udp::FacePacket;

#[derive(Clone, Copy, Debug, Default)]
pub struct PsGeom {
    pub upper_l: f32,
    pub upper_r: f32,
    pub lower_l: f32,
    pub lower_r: f32,
    pub jaw_right: f32,
    pub jaw_left: f32,
    pub jaw_forward: f32,
}

#[derive(Clone, Debug)]
pub struct FaceOutput<'a> {
    pub packet: &'a FacePacket,
    pub mouth: [f32; 6],
    pub look: (f32, f32),
    pub blink: [f32; 2],
    pub brow_ud: f32,
    pub expression: String,
    pub expression_weight: f32,
    pub visemes: [f32; VISEME_COUNT],
    pub audio: f32,
    pub ext_mouth: Option<[f32; 6]>,
    pub ps_geom: PsGeom,
}

#[derive(Clone, Debug, Default)]
struct SimpleExpr {
    last_mouth: f32,
    last_brows: f32,
    label: String,
}

impl SimpleExpr {
    fn update(&mut self, f: &FeatureVec) -> &str {
        let mouth = (f[FEAT_CORNER_UD_L] + f[FEAT_CORNER_UD_R]) * 0.5;
        let brows = (f[FEAT_BROW_UD_L] + f[FEAT_BROW_UD_R]) * 0.5;
        self.last_mouth = self.last_mouth * 0.6 + mouth * 0.4;
        self.last_brows = self.last_brows * 0.6 + brows * 0.4;
        self.label = if self.last_mouth < -0.2 {
            "fun"
        } else if self.last_brows > 0.2 {
            "surprise"
        } else if self.last_brows < -0.25 && self.last_mouth > -0.3 {
            "angry"
        } else {
            "neutral"
        }
        .into();
        &self.label
    }
}

#[derive(Default)]
pub struct OutputDriver {
    mouth: [f32; 6],
    blink: [f32; 2],
    look: (f32, f32),
    brow_ud: f32,
    simple: SimpleExpr,
    expr_label: String,
    expr_weight: f32,
    ps_rest: [f32; 6],
    ps_geom_init: bool,
    audio: f32,
}

impl OutputDriver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update<'a>(
        &mut self,
        pkt: &'a FacePacket,
        ext: Option<&ExtState>,
    ) -> Option<FaceOutput<'a>> {
        if !pkt.success {
            return None;
        }
        let now = Instant::now();
        if let Some((name, w)) = ext.and_then(|e| e.expression_live(now)) {
            self.expr_label = name.into();
            self.expr_weight = w;
        } else {
            self.expr_label = self.simple.update(&pkt.features).into();
            self.expr_weight = 1.0;
        }
        self.audio = ext.map(|e| e.audio).unwrap_or(self.audio);

        let ext_vis = ext.and_then(|e| e.visemes_live(now));
        let ext_mouth = ext_vis.and_then(|v| {
            let (idx, w) = winning(&v);
            (idx != SIL).then(|| {
                let w = (w * 1.5).clamp(0.0, 1.0);
                cats(idx).map(|c| c * w)
            })
        });

        let raw = mouth_states(pkt.features[FEAT_MOUTH_OPEN], pkt.features[FEAT_MOUTH_WIDE]);
        for i in 0..6 {
            self.mouth[i] = lerp(self.mouth[i], raw[i], 0.4);
        }
        self.blink[0] = lerp(self.blink[0], eye_to_blink(pkt.eye_blink[0]), 0.25);
        self.blink[1] = lerp(self.blink[1], eye_to_blink(pkt.eye_blink[1]), 0.25);
        let (lr, ud) = look_from_points(&pkt.pts_3d);
        self.look.0 = lerp(self.look.0, lr, 0.4);
        self.look.1 = lerp(self.look.1, ud, 0.4);
        self.brow_ud = lerp(
            self.brow_ud,
            (pkt.features[FEAT_BROW_UD_L] + pkt.features[FEAT_BROW_UD_R]) * 0.5,
            0.35,
        );

        Some(FaceOutput {
            packet: pkt,
            mouth: self.mouth,
            look: self.look,
            blink: self.blink,
            brow_ud: self.brow_ud,
            expression: self.expr_label.clone(),
            expression_weight: self.expr_weight,
            visemes: ext_vis.unwrap_or_else(|| from_mouth_states(&self.mouth)),
            audio: self.audio,
            ext_mouth,
            ps_geom: self.update_ps_geom(pkt),
        })
    }

    fn update_ps_geom(&mut self, pkt: &FacePacket) -> PsGeom {
        let f = &pkt.features;
        let (nx, ny) = face_norm_basis(&pkt.pts_3d);
        let rest = f[FEAT_MOUTH_OPEN] < 0.15
            && f[FEAT_JAW_OPEN] < 0.15
            && f[FEAT_MOUTH_WIDE].abs() < 0.2
            && f[FEAT_MOUTH_PUCKER] < 0.15
            && f[FEAT_MOUTH_FUNNEL] < 0.15;
        let p = &pkt.pts_3d;
        let d_ul = self.ps_delta(0, (p[51][1] - p[30][1]) / ny, rest);
        let d_ur = self.ps_delta(1, (p[49][1] - p[30][1]) / ny, rest);
        let d_ll = self.ps_delta(2, (p[54][1] - p[30][1]) / ny, rest);
        let d_lr = self.ps_delta(3, (p[56][1] - p[30][1]) / ny, rest);
        let d_chin = self.ps_delta(4, (p[8][0] - (p[27][0] + p[30][0]) * 0.5) / nx, rest);
        let d_fwd = self.ps_delta(5, (p[30][2] - p[8][2]) / ny, rest);
        let ready = self.ps_geom_init;
        self.ps_geom_init = true;
        let mut g = PsGeom::default();
        if ready && f[FEAT_MOUTH_PUCKER] < 0.25 && f[FEAT_MOUTH_FUNNEL] < 0.25 {
            let jaw = f[FEAT_JAW_OPEN].max(0.0) * 0.55;
            g.upper_l = (-d_ul * 1.8).clamp(0.0, 1.0);
            g.upper_r = (-d_ur * 1.8).clamp(0.0, 1.0);
            g.lower_l = (d_ll * 1.8 - jaw).clamp(0.0, 1.0);
            g.lower_r = (d_lr * 1.8 - jaw).clamp(0.0, 1.0);
        }
        if ready {
            g.jaw_right = ps_pos(d_chin, 0.08, 2.2);
            g.jaw_left = ps_pos(-d_chin, 0.08, 2.2);
            g.jaw_forward = ps_pos(d_fwd, 0.12, 1.6);
        }
        g
    }

    fn ps_delta(&mut self, i: usize, v: f32, adapt: bool) -> f32 {
        if !self.ps_geom_init {
            self.ps_rest[i] = v;
        } else if adapt {
            self.ps_rest[i] = lerp(self.ps_rest[i], v, 0.04);
        }
        v - self.ps_rest[i]
    }
}

pub fn from_mouth_states(m: &[f32; 6]) -> [f32; VISEME_COUNT] {
    let mut v = [0.0f32; VISEME_COUNT];
    v[10] = m[0];
    v[12] = m[1];
    v[14] = m[2];
    v[11] = m[3];
    v[13] = m[4];
    if m.iter().take(5).all(|x| *x < 1e-3) {
        v[SIL] = 1.0;
    }
    v
}

pub fn mouth_states(open: f32, wide: f32) -> [f32; 6] {
    mouth_states_stab(open, wide, 0.2, 0.3)
}

fn mouth_states_stab(open: f32, wide: f32, stabilizer: f32, stabilizer_wide: f32) -> [f32; 6] {
    let mut s = [0.0f32; 6];
    s[5] = (open / 0.55 - 0.1).clamp(0.0, 1.0);
    if open < stabilizer && wide.abs() < stabilizer {
        return s;
    }
    if wide > stabilizer && open < stabilizer_wide {
        return s;
    }
    if open > 0.5 {
        s[4] = open;
    } else if open >= 0.0 {
        s[0] = open;
    }
    if wide >= 0.0 && open > stabilizer * 0.5 {
        if wide > 0.5 {
            s[1] = wide;
        } else {
            s[3] = wide;
        }
    } else if wide < (stabilizer_wide * 1.5).clamp(0.0, 0.8) && open > stabilizer {
        s[2] = -wide;
    }
    let mut total = 0.0;
    let mut max = 0.0;
    for v in s.iter().take(5) {
        total += *v;
        if *v > max {
            max = *v;
        }
    }
    max = (max * 3.0).clamp(0.0, 1.0);
    if total < 1e-4 {
        return s;
    }
    for v in s.iter_mut().take(5) {
        *v = max * (*v / total);
    }
    s
}

fn look_from_points(pts: &[[f32; 3]; 70]) -> (f32, f32) {
    let a = look_one(pts, 66, 37, 38, 41, 40);
    let b = look_one(pts, 67, 43, 44, 47, 46);
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

fn look_one(
    pts: &[[f32; 3]; 70],
    gaze: usize,
    tr: usize,
    tl: usize,
    br: usize,
    bl: usize,
) -> (f32, f32) {
    let (brx, blx) = (
        (pts[tr][0] + pts[br][0]) * 0.5,
        (pts[tl][0] + pts[bl][0]) * 0.5,
    );
    let hc = (brx + blx) * 0.5;
    let hr = ((hc - brx).abs() + (blx - hc).abs()).max(2e-5) * 0.5;
    let (bt, bb) = (
        (pts[tr][1] + pts[tl][1]) * 0.5,
        (pts[br][1] + pts[bl][1]) * 0.5,
    );
    let vc = (bt + bb) * 0.5;
    let vr = ((vc - bt).abs() + (bb - vc).abs()).max(2e-5) * 0.5;
    (
        ((pts[gaze][0] - hc) / hr).clamp(-1.0, 1.0),
        ((pts[gaze][1] - vc) / vr).clamp(-1.0, 1.0),
    )
}

fn eye_to_blink(open: f32) -> f32 {
    if open > 0.55 {
        0.0
    } else if open < 0.2 {
        1.0
    } else {
        1.0 - (open - 0.2) / 0.35
    }
}

fn ps_pos(d: f32, dead: f32, scale: f32) -> f32 {
    if d > dead {
        ((d - dead) * scale).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mouth_open_maps_to_o_or_a() {
        let o = mouth_states(0.6, 0.0);
        assert!(o[4] > 0.9, "O {}", o[4]);
        let a = mouth_states(0.3, 0.0);
        assert!(a[0] > 0.8, "A {}", a[0]);
    }

    #[test]
    fn mouth_wide_maps_to_i_or_u() {
        let i = mouth_states(0.3, 0.7);
        assert!(i[1] > i[0]);
        let u = mouth_states(0.3, -0.5);
        assert!(u[2] > 0.5);
    }
}
