//! Camera, video, stills, and `--raw-rgb` stdin.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::preprocess::BgrImage;

pub struct InputSource {
    inner: Inner,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub is_camera: bool,
    pub is_video: bool,
}

enum Inner {
    Raw { width: u32, height: u32, stdin: std::io::Stdin },
    Still { frame: BgrImage, done: bool, repeat: bool },
    Ffmpeg { child: Child, width: u32, height: u32 },
    Camera(CameraCap),
}

struct CameraCap {
    cam: nokhwa::Camera,
}

impl InputSource {
    pub fn open(
        capture: &str,
        raw_rgb: bool,
        width: u32,
        height: u32,
        fps: u32,
        repeat: bool,
    ) -> Result<Self> {
        if raw_rgb {
            return Ok(Self {
                inner: Inner::Raw {
                    width,
                    height,
                    stdin: std::io::stdin(),
                },
                name: "stdin".into(),
                width,
                height,
                is_camera: false,
                is_video: false,
            });
        }
        let path = Path::new(capture);
        if path.is_file() {
            if let Ok(im) = BgrImage::load(path) {
                return Ok(Self {
                    width: im.width,
                    height: im.height,
                    name: capture.to_string(),
                    inner: Inner::Still {
                        frame: im,
                        done: false,
                        repeat,
                    },
                    is_camera: false,
                    is_video: false,
                });
            }
            let (w, h) = probe_video(path).unwrap_or((width, height));
            let child = spawn_ffmpeg(path)?;
            return Ok(Self {
                inner: Inner::Ffmpeg {
                    child,
                    width: w,
                    height: h,
                },
                name: capture.to_string(),
                width: w,
                height: h,
                is_camera: false,
                is_video: true,
            });
        }
        let idx: u32 = capture.parse().unwrap_or(0);
        let cam = open_camera(idx, width, height, fps)?;
        let res = cam.resolution();
        Ok(Self {
            name: format!("camera {idx}"),
            width: res.width(),
            height: res.height(),
            inner: Inner::Camera(CameraCap { cam }),
            is_camera: true,
            is_video: false,
        })
    }

    pub fn read(&mut self) -> Result<Option<BgrImage>> {
        match &mut self.inner {
            Inner::Raw { width, height, stdin } => {
                let n = (*width as usize) * (*height as usize) * 3;
                let mut buf = vec![0u8; n];
                if stdin.read_exact(&mut buf).is_err() {
                    return Ok(None);
                }
                Ok(Some(BgrImage {
                    width: *width,
                    height: *height,
                    data: buf,
                }))
            }
            Inner::Still { frame, done, repeat } => {
                if *done && !*repeat {
                    return Ok(None);
                }
                *done = true;
                Ok(Some(frame.clone()))
            }
            Inner::Ffmpeg {
                child,
                width,
                height,
            } => {
                let n = (*width as usize) * (*height as usize) * 3;
                let mut buf = vec![0u8; n];
                let Some(out) = child.stdout.as_mut() else {
                    return Ok(None);
                };
                if out.read_exact(&mut buf).is_err() {
                    return Ok(None);
                }
                Ok(Some(BgrImage {
                    width: *width,
                    height: *height,
                    data: buf,
                }))
            }
            Inner::Camera(c) => {
                let frame = c.cam.frame().context("camera frame")?;
                let decoded = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()?;
                let (w, h) = (decoded.width(), decoded.height());
                let mut data = Vec::with_capacity((w * h * 3) as usize);
                for p in decoded.pixels() {
                    data.extend_from_slice(&[p[2], p[1], p[0]]);
                }
                self.width = w;
                self.height = h;
                Ok(Some(BgrImage {
                    width: w,
                    height: h,
                    data,
                }))
            }
        }
    }
}

impl Drop for InputSource {
    fn drop(&mut self) {
        if let Inner::Ffmpeg { child, .. } = &mut self.inner {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn open_camera(idx: u32, _width: u32, _height: u32, _fps: u32) -> Result<nokhwa::Camera> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(CameraIndex::Index(idx), requested)?;
    cam.open_stream()?;
    Ok(cam)
}

pub fn list_cameras() -> Result<Vec<(u32, String)>> {
    use nokhwa::query;
    use nokhwa::utils::ApiBackend;
    let devices = query(ApiBackend::Auto)?;
    Ok(devices
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            let idx = match d.index() {
                nokhwa::utils::CameraIndex::Index(n) => *n,
                _ => i as u32,
            };
            (idx, d.human_name().to_string())
        })
        .collect())
}

fn probe_video(path: &Path) -> Result<(u32, u32)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .context("ffprobe")?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().split([',', 'x', '\n']);
    let w = it.next().unwrap_or("640").trim().parse().unwrap_or(640);
    let h = it.next().unwrap_or("360").trim().parse().unwrap_or(360);
    Ok((w, h))
}

fn spawn_ffmpeg(path: &Path) -> Result<Child> {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("ffmpeg")
}

pub struct VideoOut {
    child: Child,
}

impl VideoOut {
    pub fn open(path: &str, width: u32, height: u32, fps: f32) -> Result<Self> {
        let child = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                &format!("{width}x{height}"),
                "-r",
                &format!("{fps}"),
                "-i",
                "-",
                "-c:v",
                "ffv1",
                path,
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("ffmpeg video-out")?;
        Ok(Self { child })
    }

    pub fn write(&mut self, frame: &BgrImage) -> Result<()> {
        let Some(stdin) = self.child.stdin.as_mut() else {
            bail!("ffmpeg stdin closed");
        };
        stdin.write_all(&frame.data)?;
        Ok(())
    }
}

impl Drop for VideoOut {
    fn drop(&mut self) {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

pub fn mirror_bgr(frame: &BgrImage) -> BgrImage {
    frame.flip_h()
}
