//! Map an OpenSee [`FacePacket`] to VRM bones + blendshapes for VMC.

use std::collections::BTreeMap;
use std::time::Instant;

use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};

use crate::ext::{
    cats, expr_blend, from_mouth_states, winning, ExtState, SimpleExpr, SIL, VISEME_COUNT,
};
use crate::features::{
    FEAT_CHEEK_PUFF, FEAT_JAW_OPEN, FEAT_MOUTH_FUNNEL, FEAT_MOUTH_OFFSET_X, FEAT_MOUTH_PRESS_LIP_OPEN,
    FEAT_MOUTH_PUCKER,
};
use crate::udp::FacePacket;

const FEAT_EYE_L: usize = 0;
const FEAT_EYE_R: usize = 1;
const FEAT_BROW_STEEP_L: usize = 2;
const FEAT_BROW_UD_L: usize = 3;
const FEAT_BROW_STEEP_R: usize = 5;
const FEAT_BROW_UD_R: usize = 6;
const FEAT_CORNER_UD_L: usize = 8;
const FEAT_CORNER_IO_L: usize = 9;
const FEAT_CORNER_UD_R: usize = 10;
const FEAT_CORNER_IO_R: usize = 11;
const FEAT_MOUTH_OPEN: usize = 12;
const FEAT_MOUTH_WIDE: usize = 13;

const EPS: f32 = 1e-4;
const VF: f32 = 0.6;

#[derive(Clone, Debug)]
pub struct VrmCfg {
    pub perfect_sync: bool,
    pub mirror: bool,
    pub translation_scale: f32,
}

impl Default for VrmCfg {
    fn default() -> Self {
        Self {
            perfect_sync: true,
            mirror: false,
            translation_scale: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BonePose {
    pub name: &'static str,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct VrmFrame {
    pub time: f64,
    pub root_pos: [f32; 3],
    pub root_rot: [f32; 4],
    pub bones: Vec<BonePose>,
    pub blends: Vec<(String, f32)>,
    pub visemes: [f32; VISEME_COUNT],
    pub expression: String,
    pub expression_weight: f32,
    pub audio: f32,
}

impl VrmFrame {
    pub fn blend(&self, name: &str) -> Option<f32> {
        self.blends.iter().find(|(n, _)| n == name).map(|(_, w)| *w)
    }
}

pub struct VrmDriver {
    cfg: VrmCfg,
    calibrated: bool,
    d_r: UnitQuaternion<f32>,
    d_t: Vector3<f32>,
    mouth: [f32; 6],
    blink_l: f32,
    blink_r: f32,
    look_lr: f32,
    look_ud: f32,
    brow_ud: f32,
    simple: SimpleExpr,
    expr_label: String,
    expr_weight: f32,
    ps_rest: [f32; 11],
    ps_geom_init: bool,
    audio: f32,
}

impl VrmDriver {
    pub fn new(cfg: VrmCfg) -> Self {
        Self {
            cfg,
            calibrated: false,
            d_r: UnitQuaternion::identity(),
            d_t: Vector3::zeros(),
            mouth: [0.0; 6],
            blink_l: 0.0,
            blink_r: 0.0,
            look_lr: 0.0,
            look_ud: 0.0,
            brow_ud: 0.0,
            simple: SimpleExpr::default(),
            expr_label: "neutral".into(),
            expr_weight: 0.0,
            ps_rest: [0.0; 11],
            ps_geom_init: false,
            audio: 0.0,
        }
    }

    pub fn update(&mut self, pkt: &FacePacket) -> Option<VrmFrame> {
        self.update_with(pkt, None)
    }

    pub fn update_with(&mut self, pkt: &FacePacket, ext: Option<&ExtState>) -> Option<VrmFrame> {
        if !pkt.success {
            return None;
        }
        let now = Instant::now();
        if let Some((name, w)) = ext.and_then(|e| e.expression_live(now)) {
            self.expr_label = name.into();
            self.expr_weight = w;
        } else {
            self.expr_label = self.simple.update(pkt).into();
            self.expr_weight = 1.0;
        }
        self.audio = ext.map(|e| e.audio).unwrap_or(self.audio);

        let mut q = convert_quat(pkt.quaternion);
        let mut t = convert_translation(pkt.translation);
        if self.cfg.mirror {
            q = mirror_quat(q);
            t = mirror_vec(t);
        }
        if !self.calibrated {
            self.d_r = q.inverse();
            self.d_t = t;
            self.calibrated = true;
        }
        let pos = (t - self.d_t) * self.cfg.translation_scale;
        let head = q * self.d_r;
        let mut blends = BTreeMap::new();

        let ext_vis = ext.and_then(|e| e.visemes_live(now));
        let mut camera_mouth = true;
        let mut jaw_w = 0.0;
        if let Some(v) = ext_vis {
            let (idx, w) = winning(&v);
            if idx != SIL {
                camera_mouth = false;
                let w = (w * 1.5).clamp(0.0, 1.0);
                let c = cats(idx);
                let names = ["A", "I", "U", "E", "O"];
                for i in 0..5 {
                    set_blend(&mut blends, names[i], c[i] * w * VF);
                }
                jaw_w = c[5] * w * VF;
            }
        }

        let raw = mouth_states(pkt.features[FEAT_MOUTH_OPEN], pkt.features[FEAT_MOUTH_WIDE]);
        for i in 0..6 {
            self.mouth[i] = lerp(self.mouth[i], raw[i], 0.4);
        }
        if camera_mouth && !self.cfg.perfect_sync {
            let names = ["A", "I", "U", "E", "O"];
            for i in 0..5 {
                set_blend(&mut blends, names[i], self.mouth[i] * VF);
            }
            jaw_w = self.mouth[5] * VF;
        } else if camera_mouth {
            jaw_w = self.mouth[5] * VF;
        }

        self.blink_r = lerp(self.blink_r, eye_to_blink(pkt.eye_blink[0]), 0.25);
        self.blink_l = lerp(self.blink_l, eye_to_blink(pkt.eye_blink[1]), 0.25);
        if !self.cfg.perfect_sync {
            set_blend(&mut blends, "Blink_R", self.blink_r);
            set_blend(&mut blends, "Blink_L", self.blink_l);
        }

        let (lr, ud) = look_from_points(&pkt.pts_3d);
        self.look_lr = lerp(self.look_lr, lr, 0.4);
        self.look_ud = lerp(self.look_ud, ud, 0.4);
        let look_lr = if self.cfg.mirror {
            self.look_lr
        } else {
            -self.look_lr
        };
        if !self.cfg.perfect_sync {
            if self.look_ud > 0.0 {
                set_blend(&mut blends, "LookUp", (5.0 * self.look_ud).min(1.0));
            } else {
                set_blend(&mut blends, "LookDown", (-5.0 * self.look_ud).min(1.0));
            }
            if look_lr > 0.0 {
                set_blend(&mut blends, "LookLeft", (10.0 * look_lr).min(1.0));
            } else {
                set_blend(&mut blends, "LookRight", (-10.0 * look_lr).min(1.0));
            }
        }

        let brow = (pkt.features[FEAT_BROW_UD_L] + pkt.features[FEAT_BROW_UD_R]) * 0.5;
        self.brow_ud = lerp(self.brow_ud, brow, 0.35);
        if self.brow_ud > 0.0 {
            set_blend(&mut blends, "Brows up", self.brow_ud);
        } else {
            set_blend(&mut blends, "Brows down", -self.brow_ud);
        }

        if self.cfg.perfect_sync {
            self.apply_ps(pkt, &mut blends, camera_mouth);
        }
        if let Some(name) = expr_blend(&self.expr_label) {
            set_blend(&mut blends, name, self.expr_weight);
        }

        let eye = gaze_quat(self.look_ud, look_lr);
        let jaw = angle_axis(20.0 * jaw_w, Vector3::x());
        let neck = UnitQuaternion::identity().slerp(&head, 0.4);
        let visemes = ext_vis.unwrap_or_else(|| from_mouth_states(&self.mouth));
        let blends: Vec<(String, f32)> = blends.into_iter().filter(|(_, w)| *w > EPS).collect();

        Some(VrmFrame {
            time: pkt.time,
            root_pos: [pos.x, pos.y, pos.z],
            root_rot: quat_xyzw(head),
            bones: vec![
                bone("Head", head),
                bone("Neck", neck),
                bone("LeftEye", eye),
                bone("RightEye", eye),
                bone("Jaw", jaw),
            ],
            blends,
            visemes,
            expression: self.expr_label.clone(),
            expression_weight: self.expr_weight,
            audio: self.audio,
        })
    }

    fn apply_ps(&mut self, pkt: &FacePacket, blends: &mut BTreeMap<String, f32>, mouth: bool) {
        let f = &pkt.features;
        set_blend(
            blends,
            "BrowInnerUp",
            (f[FEAT_BROW_UD_L] + f[FEAT_BROW_UD_R]).max(0.0) * 0.4,
        );
        ps_blink(blends, "Left", f[FEAT_BROW_UD_L], f[FEAT_EYE_L]);
        ps_blink(blends, "Right", f[FEAT_BROW_UD_R], f[FEAT_EYE_R]);
        set_blend(
            blends,
            "BrowOuterUpLeft",
            if f[FEAT_BROW_STEEP_L] < 0.2 {
                -f[FEAT_BROW_STEEP_L]
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "BrowOuterUpRight",
            if f[FEAT_BROW_STEEP_R] < 0.2 {
                -f[FEAT_BROW_STEEP_R]
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "EyeWideLeft",
            if f[FEAT_EYE_L] > 0.5 {
                f[FEAT_EYE_L] * 0.7
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "EyeWideRight",
            if f[FEAT_EYE_R] > 0.5 {
                f[FEAT_EYE_R] * 0.7
            } else {
                0.0
            },
        );
        let dead = 0.1f32;
        let lr = if self.cfg.mirror {
            self.look_lr
        } else {
            -self.look_lr
        };
        let ud = self.look_ud;
        let up = if ud > 0.0 { ps_look(ud, dead) } else { 0.0 };
        let down = if ud < 0.0 { ps_look(ud, dead) } else { 0.0 };
        let left = if lr > 0.0 { ps_look(lr, dead) } else { 0.0 };
        let right = if lr < 0.0 { ps_look(lr, dead) } else { 0.0 };
        set_blend(blends, "EyeLookUpLeft", up);
        set_blend(blends, "EyeLookUpRight", up);
        set_blend(blends, "EyeLookDownLeft", down);
        set_blend(blends, "EyeLookDownRight", down);
        let m = self.cfg.mirror;
        set_blend(
            blends,
            if m {
                "EyeLookOutRight"
            } else {
                "EyeLookOutLeft"
            },
            left,
        );
        set_blend(
            blends,
            if m { "EyeLookInLeft" } else { "EyeLookInRight" },
            left,
        );
        set_blend(
            blends,
            if m { "EyeLookInRight" } else { "EyeLookInLeft" },
            right,
        );
        set_blend(
            blends,
            if m {
                "EyeLookOutLeft"
            } else {
                "EyeLookOutRight"
            },
            right,
        );
        if mouth {
            self.apply_ps_mouth(pkt, blends);
        }
    }

    fn apply_ps_mouth(&mut self, pkt: &FacePacket, blends: &mut BTreeMap<String, f32>) {
        let f = &pkt.features;
        set_blend(blends, "MouthPucker", f[FEAT_MOUTH_PUCKER]);
        set_blend(blends, "JawOpen", f[FEAT_JAW_OPEN]);
        set_blend(blends, "MouthClose", (-f[FEAT_MOUTH_OPEN]).max(0.0));
        set_blend(blends, "MouthFunnel", f[FEAT_MOUTH_FUNNEL]);
        set_blend(blends, "CheekPuff", f[FEAT_CHEEK_PUFF]);
        set_blend(
            blends,
            "MouthLeft",
            if f[FEAT_MOUTH_OFFSET_X] < -0.3 {
                -f[FEAT_MOUTH_OFFSET_X] * 0.5
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "MouthRight",
            if f[FEAT_MOUTH_OFFSET_X] > 0.3 {
                f[FEAT_MOUTH_OFFSET_X] * 0.5
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "MouthSmileLeft",
            if f[FEAT_CORNER_UD_L] > 0.3 {
                f[FEAT_CORNER_UD_L] * 0.5
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "MouthSmileRight",
            if f[FEAT_CORNER_UD_R] > 0.3 {
                f[FEAT_CORNER_UD_R] * 0.5
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "MouthFrownLeft",
            if f[FEAT_CORNER_UD_L] < -0.3 {
                -f[FEAT_CORNER_UD_L]
            } else {
                0.0
            },
        );
        set_blend(
            blends,
            "MouthFrownRight",
            if f[FEAT_CORNER_UD_R] < -0.3 {
                -f[FEAT_CORNER_UD_R]
            } else {
                0.0
            },
        );
        let open_gate = 1.0 - (f[FEAT_MOUTH_OPEN] / 0.45).clamp(0.0, 1.0);
        let wide = f[FEAT_MOUTH_WIDE].max(0.0);
        set_blend(
            blends,
            "MouthStretchLeft",
            (wide + f[FEAT_CORNER_IO_L].max(0.0)) * 0.6 * open_gate,
        );
        set_blend(
            blends,
            "MouthStretchRight",
            (wide + f[FEAT_CORNER_IO_R].max(0.0)) * 0.6 * open_gate,
        );
        let press = (-f[FEAT_MOUTH_PRESS_LIP_OPEN]).max(0.0);
        set_blend(blends, "MouthPressLeft", press);
        set_blend(blends, "MouthPressRight", press);

        let p = unity_pts(&pkt.pts_3d);
        let ny = ((p[27][1] - p[28][1]).abs()
            + (p[28][1] - p[29][1]).abs()
            + (p[29][1] - p[30][1]).abs())
            / 3.0;
        let ny = ny.max(1e-5);
        let nx = ((p[0][0] - p[16][0]).abs() + (p[1][0] - p[15][0]).abs()) / 2.0;
        let nx = nx.max(1e-5);
        let rest = f[FEAT_MOUTH_OPEN] < 0.15
            && f[FEAT_JAW_OPEN] < 0.15
            && f[FEAT_MOUTH_WIDE].abs() < 0.2
            && f[FEAT_MOUTH_PUCKER] < 0.15
            && f[FEAT_MOUTH_FUNNEL] < 0.15;
        let d_ul = self.ps_delta(0, (p[51][1] - p[30][1]) / ny, rest);
        let d_ur = self.ps_delta(1, (p[49][1] - p[30][1]) / ny, rest);
        let d_ll = self.ps_delta(2, (p[54][1] - p[30][1]) / ny, rest);
        let d_lr = self.ps_delta(3, (p[56][1] - p[30][1]) / ny, rest);
        let d_chin = self.ps_delta(4, (p[8][0] - (p[27][0] + p[30][0]) * 0.5) / nx, rest);
        let d_fwd = self.ps_delta(5, (p[8][2] - p[30][2]) / ny, rest);
        let ready = self.ps_geom_init;
        self.ps_geom_init = true;
        let mut upper_l = 0.0;
        let mut upper_r = 0.0;
        let mut lower_l = 0.0;
        let mut lower_r = 0.0;
        if ready && f[FEAT_MOUTH_PUCKER] < 0.25 && f[FEAT_MOUTH_FUNNEL] < 0.25 {
            upper_l = (-d_ul * 1.8).clamp(0.0, 1.0);
            upper_r = (-d_ur * 1.8).clamp(0.0, 1.0);
            lower_l = (d_ll * 1.8 - f[FEAT_JAW_OPEN].max(0.0) * 0.55).clamp(0.0, 1.0);
            lower_r = (d_lr * 1.8 - f[FEAT_JAW_OPEN].max(0.0) * 0.55).clamp(0.0, 1.0);
        }
        set_blend(blends, "MouthUpperUpLeft", upper_l);
        set_blend(blends, "MouthUpperUpRight", upper_r);
        set_blend(blends, "MouthLowerDownLeft", lower_l);
        set_blend(blends, "MouthLowerDownRight", lower_r);
        if ready {
            set_blend(blends, "JawRight", ps_pos(d_chin, 0.08, 2.2));
            set_blend(blends, "JawLeft", ps_pos(-d_chin, 0.08, 2.2));
            set_blend(blends, "JawForward", ps_pos(d_fwd, 0.12, 1.6));
        }
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

pub fn convert_quaternion(raw: [f32; 4]) -> [f32; 4] {
    [-raw[1], -raw[0], raw[2], raw[3]]
}

fn convert_quat(raw: [f32; 4]) -> UnitQuaternion<f32> {
    let q = convert_quaternion(raw);
    UnitQuaternion::new_normalize(Quaternion::new(q[3], q[0], q[1], q[2]))
}

#[cfg(test)]
pub fn convert_translation_vec(raw: [f32; 3]) -> [f32; 3] {
    let t = convert_translation(raw);
    [t.x, t.y, t.z]
}

fn convert_translation(raw: [f32; 3]) -> Vector3<f32> {
    Vector3::new(raw[1], raw[0], raw[2])
}

fn mirror_quat(q: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let q = q.quaternion();
    UnitQuaternion::new_normalize(Quaternion::new(-q.w, -q.i, q.j, q.k))
}

fn mirror_vec(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(-v.x, v.y, v.z)
}

fn quat_xyzw(q: UnitQuaternion<f32>) -> [f32; 4] {
    let q = q.into_inner();
    [q.i, q.j, q.k, q.w]
}

fn bone(name: &'static str, q: UnitQuaternion<f32>) -> BonePose {
    BonePose {
        name,
        pos: [0.0, 0.0, 0.0],
        rot: quat_xyzw(q),
    }
}

fn angle_axis(deg: f32, axis: Vector3<f32>) -> UnitQuaternion<f32> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), deg.to_radians())
}

fn gaze_quat(look_ud: f32, look_lr: f32) -> UnitQuaternion<f32> {
    angle_axis(-5.0 * look_ud, Vector3::x()) * angle_axis(-10.0 * look_lr, Vector3::y())
}

fn unity_pts(pts: &[[f32; 3]; 70]) -> [[f32; 3]; 70] {
    let mut out = [[0.0; 3]; 70];
    for (i, p) in pts.iter().enumerate() {
        out[i] = [p[0], p[1], -p[2]];
    }
    out
}

fn look_from_points(pts: &[[f32; 3]; 70]) -> (f32, f32) {
    let a = look_one(pts, 66, 37, 38, 41, 40);
    let b = look_one(pts, 67, 43, 44, 47, 46);
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

fn look_one(
    pts: &[[f32; 3]; 70],
    gaze: usize,
    top_right: usize,
    top_left: usize,
    bottom_right: usize,
    bottom_left: usize,
) -> (f32, f32) {
    let br = (pts[top_right][0] + pts[bottom_right][0]) * 0.5;
    let bl = (pts[top_left][0] + pts[bottom_left][0]) * 0.5;
    let hc = (br + bl) * 0.5;
    let hr = ((hc - br).abs() + (bl - hc).abs()).max(2e-5) * 0.5;
    let bt = (pts[top_right][1] + pts[top_left][1]) * 0.5;
    let bb = (pts[bottom_right][1] + pts[bottom_left][1]) * 0.5;
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

fn ps_blink(blends: &mut BTreeMap<String, f32>, side: &str, brow_ud: f32, eye: f32) {
    let squint = brow_ud < 0.2 && eye < 0.1 && eye > -0.6;
    set_blend(
        blends,
        &format!("BrowDown{side}"),
        if brow_ud < 0.2 { -brow_ud * 0.5 } else { 0.0 },
    );
    set_blend(
        blends,
        &format!("EyeSquint{side}"),
        if squint { -eye } else { 0.0 },
    );
    let cut = if brow_ud < 0.2 { -0.6 } else { -0.3 };
    set_blend(
        blends,
        &format!("EyeBlink{side}"),
        if squint {
            0.0
        } else if eye <= cut {
            -eye * 1.5
        } else {
            0.0
        },
    );
}

fn ps_look(v: f32, dead: f32) -> f32 {
    let a = v.abs();
    if a <= dead {
        0.0
    } else {
        ((a - dead) / (1.0 - dead)).clamp(0.0, 1.0)
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

fn set_blend(map: &mut BTreeMap<String, f32>, name: &str, w: f32) {
    let w = w.max(0.0);
    if w > EPS {
        map.insert(name.to_string(), w.min(1.0));
    } else {
        map.remove(name);
    }
}

#[cfg(test)]
pub(crate) fn sample_packet() -> FacePacket {
    FacePacket {
        time: 1.0,
        id: 0,
        width: 640.0,
        height: 360.0,
        eye_blink: [1.0, 1.0],
        success: true,
        pnp_error: 0.0,
        quaternion: [0.0, 0.0, 0.0, 1.0],
        euler: [0.0; 3],
        translation: [0.0; 3],
        lms: vec![[0.0; 3]; 68],
        pts_3d: [[0.0; 3]; 70],
        features: [0.0; crate::features::FEATURE_COUNT],
    }
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

    #[test]
    fn quaternion_matches_unity_ik() {
        let q = convert_quaternion([0.1, 0.2, 0.3, 0.9]);
        assert!((q[0] + 0.2).abs() < 1e-5);
        assert!((q[1] + 0.1).abs() < 1e-5);
        assert!((q[2] - 0.3).abs() < 1e-5);
        assert!((q[3] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn translation_matches_unity_ik() {
        assert_eq!(convert_translation_vec([1.0, 2.0, 3.0]), [2.0, 1.0, 3.0]);
    }

    #[test]
    fn driver_mouth_a_without_perfect_sync() {
        let mut d = VrmDriver::new(VrmCfg {
            perfect_sync: false,
            ..VrmCfg::default()
        });
        let mut pkt = sample_packet();
        pkt.features[FEAT_MOUTH_OPEN] = 0.3;
        let mut a = 0.0;
        for _ in 0..4 {
            a = d.update(&pkt).unwrap().blend("A").unwrap_or(0.0);
        }
        assert!(a > 0.3, "A {a}");
    }

    #[test]
    fn perfect_sync_mouth_pucker() {
        let mut d = VrmDriver::new(VrmCfg::default());
        let mut pkt = sample_packet();
        pkt.features[FEAT_MOUTH_PUCKER] = 0.7;
        let frame = d.update(&pkt).unwrap();
        assert!((frame.blend("MouthPucker").unwrap_or(0.0) - 0.7).abs() < 1e-5);
        assert!(frame.blend("A").is_none());
    }
}
