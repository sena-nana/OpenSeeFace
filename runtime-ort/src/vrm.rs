//! Map canonical [`FaceOutput`] to VRM bones + blendshape names for VMC.

use std::collections::BTreeMap;

use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};

use crate::ext::{expr_blend, VISEME_COUNT};
use crate::features::{
    FEAT_BROW_STEEP_L, FEAT_BROW_STEEP_R, FEAT_BROW_UD_L, FEAT_BROW_UD_R, FEAT_CHEEK_PUFF,
    FEAT_CORNER_IO_L, FEAT_CORNER_IO_R, FEAT_CORNER_UD_L, FEAT_CORNER_UD_R, FEAT_EYE_L, FEAT_EYE_R,
    FEAT_JAW_OPEN, FEAT_MOUTH_FUNNEL, FEAT_MOUTH_OFFSET_X, FEAT_MOUTH_OPEN,
    FEAT_MOUTH_PRESS_LIP_OPEN, FEAT_MOUTH_PUCKER, FEAT_MOUTH_WIDE,
};
use crate::output::FaceOutput;

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
}

impl VrmDriver {
    pub fn new(cfg: VrmCfg) -> Self {
        Self {
            cfg,
            calibrated: false,
            d_r: UnitQuaternion::identity(),
            d_t: Vector3::zeros(),
        }
    }

    pub fn map(&mut self, out: &FaceOutput<'_>) -> Option<VrmFrame> {
        let pkt = out.packet;
        if !pkt.success {
            return None;
        }

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

        let mouth = out.ext_mouth.unwrap_or(out.mouth);
        let jaw_w = mouth[5] * VF;
        if out.ext_mouth.is_some() || !self.cfg.perfect_sync {
            for (name, w) in ["A", "I", "U", "E", "O"].iter().zip(mouth) {
                set_blend(&mut blends, name, w * VF);
            }
        }

        let look_lr = if self.cfg.mirror {
            out.look.0
        } else {
            -out.look.0
        };
        let look_ud = out.look.1;
        if !self.cfg.perfect_sync {
            set_blend(&mut blends, "Blink_R", out.blink[0]);
            set_blend(&mut blends, "Blink_L", out.blink[1]);
            signed_look(&mut blends, "LookUp", "LookDown", look_ud, 5.0);
            signed_look(&mut blends, "LookLeft", "LookRight", look_lr, 10.0);
        }
        signed_look(&mut blends, "Brows up", "Brows down", out.brow_ud, 1.0);

        if self.cfg.perfect_sync {
            self.apply_ps(out, &mut blends, look_lr, look_ud);
        }
        if let Some(name) = expr_blend(&out.expression) {
            set_blend(&mut blends, name, out.expression_weight);
        }

        let eye = gaze_quat(look_ud, look_lr);
        let jaw = angle_axis(20.0 * jaw_w, Vector3::x());
        let neck = UnitQuaternion::identity().slerp(&head, 0.4);
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
            visemes: out.visemes,
            expression: out.expression.clone(),
            expression_weight: out.expression_weight,
            audio: out.audio,
        })
    }

    fn apply_ps(
        &self,
        out: &FaceOutput<'_>,
        blends: &mut BTreeMap<String, f32>,
        look_lr: f32,
        look_ud: f32,
    ) {
        let f = &out.packet.features;
        set_blend(
            blends,
            "BrowInnerUp",
            (f[FEAT_BROW_UD_L] + f[FEAT_BROW_UD_R]).max(0.0) * 0.4,
        );
        ps_blink(blends, true, f[FEAT_BROW_UD_L], f[FEAT_EYE_L]);
        ps_blink(blends, false, f[FEAT_BROW_UD_R], f[FEAT_EYE_R]);
        set_blend(
            blends,
            "BrowOuterUpLeft",
            below(f[FEAT_BROW_STEEP_L], 0.2, 1.0),
        );
        set_blend(
            blends,
            "BrowOuterUpRight",
            below(f[FEAT_BROW_STEEP_R], 0.2, 1.0),
        );
        set_blend(blends, "EyeWideLeft", above(f[FEAT_EYE_L], 0.5, 0.7));
        set_blend(blends, "EyeWideRight", above(f[FEAT_EYE_R], 0.5, 0.7));

        let dead = 0.1;
        let (up, down) = (pos_dir(look_ud, dead), neg_dir(look_ud, dead));
        let (left, right) = (pos_dir(look_lr, dead), neg_dir(look_lr, dead));
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
        if out.ext_mouth.is_none() {
            self.apply_ps_mouth(out, blends);
        }
    }

    fn apply_ps_mouth(&self, out: &FaceOutput<'_>, blends: &mut BTreeMap<String, f32>) {
        let f = &out.packet.features;
        set_blend(blends, "MouthPucker", f[FEAT_MOUTH_PUCKER]);
        set_blend(blends, "JawOpen", f[FEAT_JAW_OPEN]);
        set_blend(blends, "MouthClose", (-f[FEAT_MOUTH_OPEN]).max(0.0));
        set_blend(blends, "MouthFunnel", f[FEAT_MOUTH_FUNNEL]);
        set_blend(blends, "CheekPuff", f[FEAT_CHEEK_PUFF]);
        set_blend(
            blends,
            "MouthLeft",
            below(f[FEAT_MOUTH_OFFSET_X], -0.3, 0.5),
        );
        set_blend(
            blends,
            "MouthRight",
            above(f[FEAT_MOUTH_OFFSET_X], 0.3, 0.5),
        );
        set_blend(
            blends,
            "MouthSmileLeft",
            above(f[FEAT_CORNER_UD_L], 0.3, 0.5),
        );
        set_blend(
            blends,
            "MouthSmileRight",
            above(f[FEAT_CORNER_UD_R], 0.3, 0.5),
        );
        set_blend(
            blends,
            "MouthFrownLeft",
            below(f[FEAT_CORNER_UD_L], -0.3, 1.0),
        );
        set_blend(
            blends,
            "MouthFrownRight",
            below(f[FEAT_CORNER_UD_R], -0.3, 1.0),
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
        let g = &out.ps_geom;
        set_blend(blends, "MouthUpperUpLeft", g.upper_l);
        set_blend(blends, "MouthUpperUpRight", g.upper_r);
        set_blend(blends, "MouthLowerDownLeft", g.lower_l);
        set_blend(blends, "MouthLowerDownRight", g.lower_r);
        set_blend(blends, "JawRight", g.jaw_right);
        set_blend(blends, "JawLeft", g.jaw_left);
        set_blend(blends, "JawForward", g.jaw_forward);
    }
}

pub fn convert_quaternion(raw: [f32; 4]) -> [f32; 4] {
    [-raw[1], -raw[0], raw[2], raw[3]]
}

fn convert_quat(raw: [f32; 4]) -> UnitQuaternion<f32> {
    let q = convert_quaternion(raw);
    UnitQuaternion::new_normalize(Quaternion::new(q[3], q[0], q[1], q[2]))
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

fn above(v: f32, t: f32, s: f32) -> f32 {
    if v > t {
        v * s
    } else {
        0.0
    }
}

fn below(v: f32, t: f32, s: f32) -> f32 {
    if v < t {
        -v * s
    } else {
        0.0
    }
}

fn ps_look(v: f32, dead: f32) -> f32 {
    let a = v.abs();
    if a <= dead {
        0.0
    } else {
        ((a - dead) / (1.0 - dead)).clamp(0.0, 1.0)
    }
}

fn pos_dir(v: f32, dead: f32) -> f32 {
    if v > 0.0 {
        ps_look(v, dead)
    } else {
        0.0
    }
}

fn neg_dir(v: f32, dead: f32) -> f32 {
    if v < 0.0 {
        ps_look(v, dead)
    } else {
        0.0
    }
}

fn signed_look(blends: &mut BTreeMap<String, f32>, pos: &str, neg: &str, v: f32, scale: f32) {
    if v > 0.0 {
        set_blend(blends, pos, (scale * v).min(1.0));
    } else {
        set_blend(blends, neg, (-scale * v).min(1.0));
    }
}

fn ps_blink(blends: &mut BTreeMap<String, f32>, left: bool, brow_ud: f32, eye: f32) {
    let (brow, squint_n, blink_n) = if left {
        ("BrowDownLeft", "EyeSquintLeft", "EyeBlinkLeft")
    } else {
        ("BrowDownRight", "EyeSquintRight", "EyeBlinkRight")
    };
    let squint = brow_ud < 0.2 && eye < 0.1 && eye > -0.6;
    set_blend(blends, brow, below(brow_ud, 0.2, 0.5));
    set_blend(blends, squint_n, if squint { -eye } else { 0.0 });
    let cut = if brow_ud < 0.2 { -0.6 } else { -0.3 };
    set_blend(
        blends,
        blink_n,
        if squint {
            0.0
        } else if eye <= cut {
            -eye * 1.5
        } else {
            0.0
        },
    );
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
mod tests {
    use super::*;
    use crate::output::OutputDriver;
    use crate::udp::sample_packet;

    fn frame_of(
        drv: &mut VrmDriver,
        out: &mut OutputDriver,
        pkt: &crate::udp::FacePacket,
    ) -> VrmFrame {
        drv.map(&out.update(pkt, None).unwrap()).unwrap()
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
        let t = convert_translation([1.0, 2.0, 3.0]);
        assert_eq!([t.x, t.y, t.z], [2.0, 1.0, 3.0]);
    }

    #[test]
    fn driver_mouth_a_without_perfect_sync() {
        let mut d = VrmDriver::new(VrmCfg {
            perfect_sync: false,
            ..VrmCfg::default()
        });
        let mut out = OutputDriver::new();
        let mut pkt = sample_packet();
        pkt.features[FEAT_MOUTH_OPEN] = 0.3;
        let mut a = 0.0;
        for _ in 0..4 {
            a = frame_of(&mut d, &mut out, &pkt).blend("A").unwrap_or(0.0);
        }
        assert!(a > 0.3, "A {a}");
    }

    #[test]
    fn perfect_sync_mouth_pucker() {
        let mut d = VrmDriver::new(VrmCfg::default());
        let mut out = OutputDriver::new();
        let mut pkt = sample_packet();
        pkt.features[FEAT_MOUTH_PUCKER] = 0.7;
        let got = frame_of(&mut d, &mut out, &pkt);
        assert!((got.blend("MouthPucker").unwrap_or(0.0) - 0.7).abs() < 1e-5);
        assert!(got.blend("A").is_none());
    }
}
