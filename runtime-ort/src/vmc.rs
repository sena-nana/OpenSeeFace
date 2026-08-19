//! VMC Protocol (OSC) encoder for a [`VrmFrame`].

use anyhow::{Context, Result};
use rosc::{encoder, OscBundle, OscMessage, OscPacket, OscTime, OscType};

use crate::vrm::VrmFrame;

fn msg(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

fn ffff(v: [f32; 3]) -> [OscType; 3] {
    [
        OscType::Float(v[0]),
        OscType::Float(v[1]),
        OscType::Float(v[2]),
    ]
}

fn bone_args(name: &str, pos: [f32; 3], rot: [f32; 4]) -> Vec<OscType> {
    vec![
        OscType::String(name.to_string()),
        OscType::Float(pos[0]),
        OscType::Float(pos[1]),
        OscType::Float(pos[2]),
        OscType::Float(rot[0]),
        OscType::Float(rot[1]),
        OscType::Float(rot[2]),
        OscType::Float(rot[3]),
    ]
}

/// One UDP datagram: Root, bones, blendshapes, Apply, OK, timestamp.
pub fn encode_vmc(frame: &VrmFrame) -> Result<Vec<u8>> {
    let mut content = Vec::with_capacity(8 + frame.bones.len() + frame.blends.len());
    let mut root = vec![OscType::String("root".into())];
    root.extend_from_slice(&ffff(frame.root_pos));
    root.extend_from_slice(&[
        OscType::Float(frame.root_rot[0]),
        OscType::Float(frame.root_rot[1]),
        OscType::Float(frame.root_rot[2]),
        OscType::Float(frame.root_rot[3]),
    ]);
    content.push(msg("/VMC/Ext/Root/Pos", root));
    for b in &frame.bones {
        content.push(msg("/VMC/Ext/Bone/Pos", bone_args(b.name, b.pos, b.rot)));
    }
    for (name, w) in &frame.blends {
        content.push(msg(
            "/VMC/Ext/Blend/Val",
            vec![OscType::String(name.clone()), OscType::Float(*w)],
        ));
    }
    content.extend(crate::ext::encode_ext(
        &frame.visemes,
        &frame.expression,
        frame.expression_weight,
        frame.audio,
    ));
    content.push(msg("/VMC/Ext/Blend/Apply", vec![]));
    content.push(msg("/VMC/Ext/OK", vec![OscType::Int(1)]));
    content.push(msg("/VMC/Ext/T", vec![OscType::Float(frame.time as f32)]));
    let packet = OscPacket::Bundle(OscBundle {
        timetag: OscTime {
            seconds: 0,
            fractional: 1,
        },
        content,
    });
    encoder::encode(&packet).context("encode VMC OSC")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FEAT_JAW_OPEN, FEAT_MOUTH_PUCKER};
    use crate::vrm::{sample_packet, VrmCfg, VrmDriver};

    #[test]
    fn bundle_contains_apply_and_jaw_open() {
        let mut d = VrmDriver::new(VrmCfg::default());
        let mut pkt = sample_packet();
        pkt.features[FEAT_MOUTH_PUCKER] = 0.5;
        pkt.features[FEAT_JAW_OPEN] = 0.4;
        let frame = d.update(&pkt).unwrap();
        let buf = encode_vmc(&frame).unwrap();
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("/VMC/Ext/Blend/Apply"), "{text:?}");
        assert!(text.contains("JawOpen"), "{text:?}");
        assert!(text.contains("/VMC/Ext/Bone/Pos"));
        assert!(text.contains("Head"));
        assert!(text.contains("/OSF/Ext/Visemes"));
        assert!(text.contains("/OSF/Ext/Expression"));
    }

    #[test]
    fn viseme_ext_maps_aa_to_a() {
        let mut d = VrmDriver::new(VrmCfg {
            perfect_sync: false,
            ..VrmCfg::default()
        });
        let mut ext = crate::ext::ExtState::default();
        ext.visemes[10] = 0.8; // aa
        ext.visemes_at = Some(std::time::Instant::now());
        let frame = d
            .update_with(&crate::vrm::sample_packet(), Some(&ext))
            .unwrap();
        let a = frame.blend("A").unwrap_or(0.0);
        assert!(a > 0.5, "A {a}");
    }
}
