//! OpenSee UDP packet (1785 bytes / face). Same layout as Unity `OpenSee.packetFrameSize`.

/// 8+4+8+8+1+4+12+12+16+272+544+840+56 = 1785
pub const PACKET_FRAME_SIZE: usize = 1785;

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
    pub features: [f32; 14],
}

pub fn encode_face(f: &FacePacket) -> Vec<u8> {
    let mut p = Vec::with_capacity(PACKET_FRAME_SIZE);
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
    let mut lms = f.lms.clone();
    lms.resize(68, [0.0, 0.0, 0.0]);
    for pt in &lms[..68] {
        p.extend_from_slice(&pt[2].to_le_bytes());
    }
    for pt in &lms[..68] {
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
    debug_assert_eq!(p.len(), PACKET_FRAME_SIZE);
    p
}

pub fn encode_faces(faces: &[FacePacket]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PACKET_FRAME_SIZE * faces.len());
    for f in faces {
        out.extend_from_slice(&encode_face(f));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_is_1785() {
        let f = FacePacket {
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
            features: [0.0; 14],
        };
        assert_eq!(encode_face(&f).len(), PACKET_FRAME_SIZE);
        assert_eq!(
            PACKET_FRAME_SIZE,
            8 + 4 + 2 * 4 + 2 * 4 + 1 + 4 + 3 * 4 + 3 * 4 + 4 * 4 + 4 * 68 + 4 * 2 * 68 + 4 * 3 * 70
                + 4 * 14
        );
    }
}
