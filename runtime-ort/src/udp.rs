//! Official OpenSee UDP packet: 1785 bytes / face, 14 expression features.

use crate::features::FeatureVec;
use crate::tracker::FaceInfo;

/// 8+4+8+8+1+4+12+12+16+272+544+840+56 = 1785
pub const PACKET_FRAME_SIZE: usize = 1785;
pub const PACKET_FRAME_SIZE_LEGACY: usize = PACKET_FRAME_SIZE;

#[derive(Clone, Debug)]
pub struct FacePacket {
    pub time: f64,
    pub id: i32,
    pub width: f32,
    pub height: f32,
    pub eye_blink: [f32; 2],
    pub success: bool,
    pub pnp_error: f32,
    pub quaternion: [f32; 4],
    pub euler: [f32; 3],
    pub translation: [f32; 3],
    pub lms: Vec<[f32; 3]>,
    pub pts_3d: [[f32; 3]; 70],
    pub features: FeatureVec,
}

impl FacePacket {
    pub fn from_face(face: &FaceInfo, time: f64, width: f32, height: f32, id: i32) -> Self {
        Self {
            time,
            id,
            width,
            height,
            eye_blink: face.eye_blink,
            success: face.success,
            pnp_error: face.pnp_error,
            quaternion: face.quaternion,
            euler: face.euler,
            translation: face.translation,
            lms: face.lms.clone(),
            pts_3d: face.pts_3d,
            features: face.current_features,
        }
    }
}

pub fn encode_face(f: &FacePacket) -> Vec<u8> {
    let mut p = Vec::with_capacity(PACKET_FRAME_SIZE);
    encode_face_append(&mut p, f);
    debug_assert_eq!(p.len(), PACKET_FRAME_SIZE);
    p
}

fn encode_face_append(p: &mut Vec<u8>, f: &FacePacket) {
    p.extend_from_slice(&f.time.to_le_bytes());
    p.extend_from_slice(&f.id.to_le_bytes());
    p.extend_from_slice(&f.width.to_le_bytes());
    p.extend_from_slice(&f.height.to_le_bytes());
    p.extend_from_slice(&f.eye_blink[0].to_le_bytes());
    p.extend_from_slice(&f.eye_blink[1].to_le_bytes());
    p.push(u8::from(f.success));
    p.extend_from_slice(&f.pnp_error.to_le_bytes());
    for v in f.quaternion {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in f.euler {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in f.translation {
        p.extend_from_slice(&v.to_le_bytes());
    }
    let zero = [0.0f32, 0.0, 0.0];
    for i in 0..68 {
        let pt = f.lms.get(i).unwrap_or(&zero);
        p.extend_from_slice(&pt[2].to_le_bytes());
    }
    for i in 0..68 {
        let pt = f.lms.get(i).unwrap_or(&zero);
        p.extend_from_slice(&pt[1].to_le_bytes());
        p.extend_from_slice(&pt[0].to_le_bytes());
    }
    for pt in f.pts_3d {
        p.extend_from_slice(&pt[0].to_le_bytes());
        p.extend_from_slice(&(-pt[1]).to_le_bytes());
        p.extend_from_slice(&(-pt[2]).to_le_bytes());
    }
    for v in f.features {
        p.extend_from_slice(&v.to_le_bytes());
    }
}

pub fn encode_faces(faces: &[FacePacket]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PACKET_FRAME_SIZE * faces.len());
    encode_faces_into(&mut out, faces);
    out
}

pub fn encode_faces_into(out: &mut Vec<u8>, faces: &[FacePacket]) {
    out.clear();
    out.reserve(PACKET_FRAME_SIZE * faces.len());
    for f in faces {
        encode_face_append(out, f);
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
    use crate::features::{FEATURE_COUNT, FEAT_EYE_L, FEAT_MOUTH_OPEN, FEAT_MOUTH_WIDE};

    #[test]
    fn packet_is_1785() {
        assert_eq!(encode_face(&sample_packet()).len(), PACKET_FRAME_SIZE);
        assert_eq!(
            PACKET_FRAME_SIZE,
            8 + 4
                + 2 * 4
                + 2 * 4
                + 1
                + 4
                + 3 * 4
                + 3 * 4
                + 4 * 4
                + 4 * 68
                + 4 * 2 * 68
                + 4 * 3 * 70
                + 4 * FEATURE_COUNT
        );
        assert_eq!(PACKET_FRAME_SIZE_LEGACY, PACKET_FRAME_SIZE);
    }

    #[test]
    fn features_are_appended_in_slot_order() {
        let mut f = sample_packet();
        f.features[FEAT_EYE_L] = -0.75;
        f.features[FEAT_MOUTH_OPEN] = 0.5;
        f.features[FEAT_MOUTH_WIDE] = -0.25;
        let bytes = encode_face(&f);
        let n = bytes.len();
        let eye_l = f32::from_le_bytes(bytes[n - 56..n - 52].try_into().unwrap());
        let mouth_open = f32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap());
        let mouth_wide = f32::from_le_bytes(bytes[n - 4..n].try_into().unwrap());
        assert_eq!(eye_l, -0.75);
        assert_eq!(mouth_open, 0.5);
        assert_eq!(mouth_wide, -0.25);
    }
}
