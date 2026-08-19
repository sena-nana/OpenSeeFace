//! Official OpenSee UDP packet: 1805 bytes / face, 19 expression features.
//! Slots 14–18 are `mouth_pucker`, `mouth_offset_x`, `cheek_puff`, `jaw_open`,
//! `mouth_funnel`. Unity still accepts 1785-byte (14-feature) and 1797-byte
//! (17-feature) packets.

use crate::features::FeatureVec;

/// 8+4+8+8+1+4+12+12+16+272+544+840+76 = 1805
pub const PACKET_FRAME_SIZE: usize = 1805;
/// 17-feature packets from the previous extra-slot revision.
pub const PACKET_FRAME_SIZE_17: usize = 1797;
/// Older 14-feature packets. Tracker no longer emits these.
pub const PACKET_FRAME_SIZE_LEGACY: usize = 1785;

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
mod tests {
    use super::*;
    use crate::features::{
        FEATURE_COUNT, FEAT_CHEEK_PUFF, FEAT_JAW_OPEN, FEAT_MOUTH_FUNNEL, FEAT_MOUTH_OFFSET_X,
        FEAT_MOUTH_PUCKER,
    };

    fn sample() -> FacePacket {
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
            features: [0.0; FEATURE_COUNT],
        }
    }

    #[test]
    fn packet_is_1805() {
        assert_eq!(encode_face(&sample()).len(), PACKET_FRAME_SIZE);
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
        assert_eq!(PACKET_FRAME_SIZE_17, PACKET_FRAME_SIZE - 4 * 2);
        assert_eq!(PACKET_FRAME_SIZE_LEGACY, PACKET_FRAME_SIZE - 4 * 5);
    }

    #[test]
    fn extra_features_are_appended() {
        let mut f = sample();
        f.features[FEAT_MOUTH_PUCKER] = 0.5;
        f.features[FEAT_MOUTH_OFFSET_X] = -0.25;
        f.features[FEAT_CHEEK_PUFF] = 0.8;
        f.features[FEAT_JAW_OPEN] = 0.7;
        f.features[FEAT_MOUTH_FUNNEL] = 0.4;
        let bytes = encode_face(&f);
        let n = bytes.len();
        let pucker = f32::from_le_bytes(bytes[n - 20..n - 16].try_into().unwrap());
        let offset = f32::from_le_bytes(bytes[n - 16..n - 12].try_into().unwrap());
        let puff = f32::from_le_bytes(bytes[n - 12..n - 8].try_into().unwrap());
        let jaw = f32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap());
        let funnel = f32::from_le_bytes(bytes[n - 4..n].try_into().unwrap());
        assert_eq!(pucker, 0.5);
        assert_eq!(offset, -0.25);
        assert_eq!(puff, 0.8);
        assert_eq!(jaw, 0.7);
        assert_eq!(funnel, 0.4);
    }
}
