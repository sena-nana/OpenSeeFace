//! `/OSF/Ext/*` OSC: OVRLipSync visemes and SVM/simple expressions.
//!
//! Listen on UDP 39540 (inbound sidecar). The same messages are appended to
//! the VMC bundle (outbound). Viseme floats use OVRLipSync.Viseme order:
//! sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, ih, oh, ou.

use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

use anyhow::{Context, Result};
use rosc::{OscMessage, OscPacket, OscType};

pub const VISEME_COUNT: usize = 15;
pub const VISEME_NAMES: [&str; VISEME_COUNT] = [
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou",
];
const VISEME_BLEND: [&str; VISEME_COUNT] = [
    "SIL", "PP", "FF", "TH", "DD", "KK", "CH", "SS", "NN", "RR", "A", "E", "I", "O", "U",
];
const CATS: [[f32; 6]; VISEME_COUNT] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.0, 0.0, 0.4, 0.0],
    [0.2, 0.4, 0.0, 0.0, 0.0, 0.1],
    [0.4, 0.0, 0.0, 0.0, 0.15, 0.5],
    [0.3, 0.7, 0.0, 0.0, 0.0, 0.7],
    [0.7, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9996, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.8, 0.0, 0.0, 0.0, 0.3],
    [0.2, 0.7, 0.0, 0.0, 0.0, 0.1],
    [0.0, 0.5, 0.0, 0.0, 0.3, 0.4],
    [0.9998, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.9997, 0.0, 0.6],
    [0.5, 0.2, 0.0, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.9999, 1.0],
    [0.0, 0.0, 0.9995, 0.0, 0.0, 1.0],
];
pub const SIL: usize = 0;
const STALE: f32 = 0.35;

pub fn parse_viseme_name(name: &str) -> Option<usize> {
    let n = name.trim();
    VISEME_NAMES
        .iter()
        .position(|s| s.eq_ignore_ascii_case(n))
        .or_else(|| VISEME_BLEND.iter().position(|s| s.eq_ignore_ascii_case(n)))
}

pub fn cats(i: usize) -> [f32; 6] {
    CATS.get(i).copied().unwrap_or([0.0; 6])
}

pub fn winning(v: &[f32; VISEME_COUNT]) -> (usize, f32) {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, w)| (i, *w))
        .unwrap_or((0, 0.0))
}

pub fn expr_blend(name: &str) -> Option<&'static str> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "fun" | "smile" => "Fun",
        "joy" | "shocked" => "Joy",
        "angry" => "Angry",
        "sorrow" | "sad" => "Sorrow",
        "surprise" | "surprised" => "Surprised",
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct ExtState {
    pub visemes: [f32; VISEME_COUNT],
    pub visemes_at: Option<Instant>,
    pub expression: String,
    pub expression_weight: f32,
    pub expression_at: Option<Instant>,
    pub audio: f32,
}

impl Default for ExtState {
    fn default() -> Self {
        Self {
            visemes: [0.0; VISEME_COUNT],
            visemes_at: None,
            expression: String::new(),
            expression_weight: 0.0,
            expression_at: None,
            audio: 0.0,
        }
    }
}

impl ExtState {
    pub fn visemes_live(&self, now: Instant) -> Option<[f32; VISEME_COUNT]> {
        let t = self.visemes_at?;
        (now.duration_since(t).as_secs_f32() <= STALE).then_some(self.visemes)
    }

    pub fn expression_live(&self, now: Instant) -> Option<(&str, f32)> {
        let t = self.expression_at?;
        if now.duration_since(t).as_secs_f32() > STALE || self.expression.is_empty() {
            None
        } else {
            Some((self.expression.as_str(), self.expression_weight))
        }
    }
}

pub fn apply_packet(state: &mut ExtState, packet: &OscPacket, now: Instant) {
    match packet {
        OscPacket::Message(m) => apply_msg(state, m, now),
        OscPacket::Bundle(b) => {
            for p in &b.content {
                apply_packet(state, p, now);
            }
        }
    }
}

fn f32_arg(args: &[OscType], i: usize) -> Option<f32> {
    match args.get(i)? {
        OscType::Float(v) => Some(*v),
        OscType::Double(v) => Some(*v as f32),
        OscType::Int(v) => Some(*v as f32),
        _ => None,
    }
}

fn apply_msg(state: &mut ExtState, m: &OscMessage, now: Instant) {
    match m.addr.as_str() {
        "/OSF/Ext/Visemes" if m.args.len() >= VISEME_COUNT => {
            for i in 0..VISEME_COUNT {
                if let Some(v) = f32_arg(&m.args, i) {
                    state.visemes[i] = v.clamp(0.0, 1.0);
                }
            }
            state.visemes_at = Some(now);
        }
        "/OSF/Ext/Viseme" => {
            if let (Some(OscType::String(name)), Some(w)) = (m.args.first(), f32_arg(&m.args, 1)) {
                if let Some(i) = parse_viseme_name(name) {
                    state.visemes[i] = w.clamp(0.0, 1.0);
                    state.visemes_at = Some(now);
                }
            }
        }
        "/OSF/Ext/Expression" => {
            if let Some(OscType::String(name)) = m.args.first() {
                state.expression = name.clone();
                state.expression_weight = f32_arg(&m.args, 1).unwrap_or(1.0).clamp(0.0, 1.0);
                state.expression_at = Some(now);
            }
        }
        "/OSF/Ext/Audio" => {
            if let Some(v) = f32_arg(&m.args, 0) {
                state.audio = v.clamp(0.0, 1.0);
            }
        }
        _ => {}
    }
}

pub fn encode_ext(
    visemes: &[f32; VISEME_COUNT],
    expression: &str,
    weight: f32,
    audio: f32,
) -> Vec<OscPacket> {
    vec![
        OscPacket::Message(OscMessage {
            addr: "/OSF/Ext/Visemes".into(),
            args: visemes.iter().map(|v| OscType::Float(*v)).collect(),
        }),
        OscPacket::Message(OscMessage {
            addr: "/OSF/Ext/Expression".into(),
            args: vec![
                OscType::String(expression.to_string()),
                OscType::Float(weight),
            ],
        }),
        OscPacket::Message(OscMessage {
            addr: "/OSF/Ext/Audio".into(),
            args: vec![OscType::Float(audio)],
        }),
    ]
}

pub struct ExtListener {
    sock: UdpSocket,
    buf: Vec<u8>,
    pub state: ExtState,
}

impl ExtListener {
    pub fn bind(addr: SocketAddr) -> Result<Self> {
        let sock = UdpSocket::bind(addr).with_context(|| format!("OSF ext bind {addr}"))?;
        sock.set_nonblocking(true)?;
        Ok(Self {
            sock,
            buf: vec![0u8; 65535],
            state: ExtState::default(),
        })
    }

    pub fn poll(&mut self) -> &ExtState {
        let now = Instant::now();
        while let Ok((n, _)) = self.sock.recv_from(&mut self.buf) {
            if let Ok((_, packet)) = rosc::decoder::decode_udp(&self.buf[..n]) {
                apply_packet(&mut self.state, &packet, now);
            }
        }
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aa_cats_and_parse() {
        assert!(cats(10)[0] > 0.99);
        assert_eq!(parse_viseme_name("A"), Some(10));
        assert_eq!(parse_viseme_name("sil"), Some(0));
        assert_eq!(expr_blend("fun"), Some("Fun"));
    }

    #[test]
    fn visemes_roundtrip() {
        let mut v = [0.0f32; VISEME_COUNT];
        v[10] = 0.8;
        let buf = rosc::encoder::encode(&OscPacket::Message(OscMessage {
            addr: "/OSF/Ext/Visemes".into(),
            args: v.iter().map(|x| OscType::Float(*x)).collect(),
        }))
        .unwrap();
        let (_, pkt) = rosc::decoder::decode_udp(&buf).unwrap();
        let mut st = ExtState::default();
        apply_packet(&mut st, &pkt, Instant::now());
        assert!((st.visemes[10] - 0.8).abs() < 1e-5);
    }
}
