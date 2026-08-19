//! Landmark overlay + optional minifb window.

use anyhow::Result;
use minifb::{Key, Window, WindowOptions};

use crate::pnp::{project_points, Camera};
use crate::preprocess::BgrImage;
use crate::tracker::FaceInfo;

pub fn draw_tracking(
    frame: &mut BgrImage,
    faces: &[FaceInfo],
    visualize: i32,
    pnp_points: i32,
    cam: &Camera,
) {
    for f in faces {
        if visualize > 1 {
            put_text(
                frame,
                &f.id.to_string(),
                f.bbox[0] as i32,
                f.bbox[1] as i32,
                [255, 0, 255],
            );
        }
        if visualize > 2 {
            put_text(
                frame,
                &format!("{:.4}", f.conf),
                f.bbox[0] as i32 + 18,
                f.bbox[1] as i32 - 6,
                [0, 0, 255],
            );
        }
        for (pt_num, pt) in f.lms.iter().enumerate() {
            if pt_num == 66 && (f.eye_blink[0] < 0.15 || pt[2] < 0.20) {
                continue;
            }
            if pt_num == 67 && (f.eye_blink[1] < 0.15 || pt[2] < 0.20) {
                continue;
            }
            let x = (pt[0] + 0.5) as i32;
            let y = (pt[1] + 0.5) as i32;
            if visualize > 3 {
                put_text(frame, &pt_num.to_string(), y, x, [0, 255, 255]);
            }
            let color = if pt_num >= 66 {
                [0, 255, 255]
            } else {
                [0, 255, 0]
            };
            if x >= 0 && y >= 0 && x < frame.height as i32 && y < frame.width as i32 {
                frame.set(y, x, color);
            }
        }
        if pnp_points != 0 {
            if let Some(rvec) = f.rotation {
                let pts = if pnp_points > 1 {
                    project_points(
                        &f.face_3d[..66.min(f.face_3d.len())],
                        rvec,
                        f.translation,
                        cam,
                    )
                } else {
                    project_points(&f.contour, rvec, f.translation, cam)
                };
                for p in pts {
                    let mut x = (p[0] + 0.5) as i32;
                    let y = (p[1] + 0.5) as i32;
                    for _ in 0..4 {
                        if x >= 0 && y >= 0 && x < frame.height as i32 && y < frame.width as i32 {
                            frame.set(y, x, [0, 255, 255]);
                        }
                        x += 1;
                    }
                }
            }
        }
    }
}

fn put_text(frame: &mut BgrImage, s: &str, x: i32, y: i32, color: [u8; 3]) {
    let mut cx = x;
    for ch in s.chars() {
        draw_char(frame, ch, cx, y, color);
        cx += 6;
    }
}

fn draw_char(frame: &mut BgrImage, ch: char, x: i32, y: i32, color: [u8; 3]) {
    let bits = glyph(ch);
    for row in 0..7 {
        for col in 0..5 {
            if bits[row] & (1 << (4 - col)) != 0 {
                frame.set(x + col as i32, y + row as i32, color);
            }
        }
    }
}

fn glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
        '3' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '.' => [0, 0, 0, 0, 0, 0b00100, 0b00100],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        _ => [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ],
    }
}

pub struct VizWindow {
    win: Window,
    w: usize,
    h: usize,
}

impl VizWindow {
    pub fn open(width: u32, height: u32) -> Result<Self> {
        let w = width as usize;
        let h = height as usize;
        let win = Window::new(
            "OpenSeeFace Visualization",
            w,
            h,
            WindowOptions {
                resize: false,
                ..WindowOptions::default()
            },
        )?;
        Ok(Self { win, w, h })
    }

    pub fn show(&mut self, frame: &BgrImage) -> bool {
        let mut buf = vec![0u32; self.w * self.h];
        let sw = frame.width as usize;
        let sh = frame.height as usize;
        for y in 0..self.h.min(sh) {
            for x in 0..self.w.min(sw) {
                let c = frame.get(x as i32, y as i32);
                buf[y * self.w + x] = (c[2] as u32) << 16 | (c[1] as u32) << 8 | c[0] as u32;
            }
        }
        let _ = self.win.update_with_buffer(&buf, self.w, self.h);
        self.win.is_open() && !self.win.is_key_down(Key::Q)
    }
}

pub fn dump_symmetric_points(face_3d: &[[f32; 3]], path: &str) -> Result<()> {
    let pairs = [
        (0, 16),
        (1, 15),
        (2, 14),
        (3, 13),
        (4, 12),
        (5, 11),
        (6, 10),
        (7, 9),
        (17, 26),
        (18, 25),
        (19, 24),
        (20, 23),
        (21, 22),
        (31, 35),
        (32, 34),
        (36, 45),
        (37, 44),
        (38, 43),
        (39, 42),
        (40, 47),
        (41, 46),
        (48, 52),
        (49, 51),
        (56, 54),
        (57, 53),
        (58, 62),
        (59, 61),
        (65, 63),
    ];
    let mut points: Vec<[f32; 3]> = face_3d.to_vec();
    for (a, b) in pairs {
        let x = (points[a][0] - points[b][0]) / 2.0;
        let y = (points[a][1] + points[b][1]) / 2.0;
        let z = (points[a][2] + points[b][2]) / 2.0;
        points[a][0] = x;
        points[b][0] = -x;
        points[a][1] = y;
        points[b][1] = y;
        points[a][2] = z;
        points[b][2] = z;
    }
    for i in [8, 27, 28, 29, 33, 50, 55, 60, 64] {
        points[i][0] = 0.0;
    }
    points[30] = [0.0, 0.0, 0.0];
    let mut s = String::from("[");
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n ");
        }
        s.push_str(&format!("[{:.15}, {:.15}, {:.15}]", p[0], p[1], p[2]));
    }
    s.push(']');
    std::fs::write(path, s)?;
    Ok(())
}
