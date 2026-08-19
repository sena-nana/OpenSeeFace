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
    Safe(SafeInner),
    /// Opened camera; not `Send`, so `PipedInput` keeps it on the calling thread.
    Camera(CameraCap),
    /// Camera index to open on the capture worker (backends are not `Send`).
    CameraJob {
        idx: u32,
        width: u32,
        height: u32,
        fps: u32,
    },
}

enum SafeInner {
    Raw {
        width: u32,
        height: u32,
        stdin: std::io::Stdin,
    },
    Still {
        frame: BgrImage,
        done: bool,
        repeat: bool,
    },
    Ffmpeg {
        child: Child,
        width: u32,
        height: u32,
    },
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
                inner: Inner::Safe(SafeInner::Raw {
                    width,
                    height,
                    stdin: std::io::stdin(),
                }),
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
                    inner: Inner::Safe(SafeInner::Still {
                        frame: im,
                        done: false,
                        repeat,
                    }),
                    is_camera: false,
                    is_video: false,
                });
            }
            let (w, h) = probe_video(path).unwrap_or((width, height));
            let child = spawn_ffmpeg(path)?;
            return Ok(Self {
                inner: Inner::Safe(SafeInner::Ffmpeg {
                    child,
                    width: w,
                    height: h,
                }),
                name: capture.to_string(),
                width: w,
                height: h,
                is_camera: false,
                is_video: true,
            });
        }
        let idx: u32 = capture.parse().unwrap_or(0);
        Ok(Self {
            name: format!("camera {idx}"),
            width: width.max(1),
            height: height.max(1),
            inner: Inner::CameraJob {
                idx,
                width,
                height,
                fps,
            },
            is_camera: true,
            is_video: false,
        })
    }

    pub fn read(&mut self) -> Result<Option<BgrImage>> {
        let mut buf = BgrImage::zeros(0, 0);
        if self.read_into(&mut buf)? {
            Ok(Some(buf))
        } else {
            Ok(None)
        }
    }

    /// Fill `dst`, reusing its allocation when the size is unchanged.
    pub fn read_into(&mut self, dst: &mut BgrImage) -> Result<bool> {
        let job = match &self.inner {
            Inner::CameraJob {
                idx,
                width,
                height,
                fps,
            } => Some((*idx, *width, *height, *fps)),
            _ => None,
        };
        if let Some((idx, width, height, fps)) = job {
            let cam = open_camera(idx, width, height, fps)?;
            let res = cam.resolution();
            self.width = res.width();
            self.height = res.height();
            self.inner = Inner::Camera(CameraCap { cam });
        }
        match &mut self.inner {
            Inner::Safe(safe) => safe.read_into(dst),
            Inner::Camera(c) => {
                let ok = read_camera(&mut c.cam, dst)?;
                if ok {
                    self.width = dst.width;
                    self.height = dst.height;
                }
                Ok(ok)
            }
            Inner::CameraJob { .. } => unreachable!("camera job opened above"),
        }
    }
}

impl SafeInner {
    fn read_into(&mut self, dst: &mut BgrImage) -> Result<bool> {
        match self {
            SafeInner::Raw {
                width,
                height,
                stdin,
            } => {
                dst.resize_buffer(*width, *height);
                if stdin.read_exact(&mut dst.data).is_err() {
                    return Ok(false);
                }
                Ok(true)
            }
            SafeInner::Still {
                frame,
                done,
                repeat,
            } => {
                if *done && !*repeat {
                    return Ok(false);
                }
                *done = true;
                dst.resize_buffer(frame.width, frame.height);
                dst.data.copy_from_slice(&frame.data);
                Ok(true)
            }
            SafeInner::Ffmpeg {
                child,
                width,
                height,
            } => {
                dst.resize_buffer(*width, *height);
                let Some(out) = child.stdout.as_mut() else {
                    return Ok(false);
                };
                if out.read_exact(&mut dst.data).is_err() {
                    return Ok(false);
                }
                Ok(true)
            }
        }
    }
}

impl Drop for SafeInner {
    fn drop(&mut self) {
        if let SafeInner::Ffmpeg { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn read_camera(cam: &mut nokhwa::Camera, dst: &mut BgrImage) -> Result<bool> {
    let frame = cam.frame().context("camera frame")?;
    let decoded = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()?;
    let (w, h) = (decoded.width(), decoded.height());
    let raw = decoded.into_raw();
    dst.resize_buffer(w, h);
    if raw.len() != dst.data.len() {
        bail!("camera frame size mismatch");
    }
    dst.data.copy_from_slice(&raw);
    crate::preprocess::rgb_to_bgr_in_place(&mut dst.data);
    Ok(true)
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

/// Capture thread + two recycled BGR buffers so I/O overlaps with inference.
/// Camera backends are not `Send`; they are opened on the worker thread.
pub struct PipedInput {
    inner: PipedInner,
}

enum PipedInner {
    Thread {
        rx: std::sync::mpsc::Receiver<Option<BgrImage>>,
        recycle: Option<std::sync::mpsc::SyncSender<BgrImage>>,
        join: Option<std::thread::JoinHandle<()>>,
    },
    Local {
        src: InputSource,
        buf: BgrImage,
    },
}

impl PipedInput {
    pub fn start(src: InputSource) -> Result<Self> {
        let InputSource {
            inner,
            width,
            height,
            ..
        } = src;
        match inner {
            Inner::CameraJob {
                idx,
                width,
                height,
                fps,
            } => spawn_worker(Worker::Camera {
                idx,
                width,
                height,
                fps,
            }),
            Inner::Safe(safe) => spawn_worker(Worker::Safe {
                inner: safe,
                width,
                height,
            }),
            Inner::Camera(cap) => {
                let w = width.max(1);
                let h = height.max(1);
                Ok(Self {
                    inner: PipedInner::Local {
                        buf: BgrImage::zeros(w, h),
                        src: InputSource {
                            inner: Inner::Camera(cap),
                            name: String::new(),
                            width: w,
                            height: h,
                            is_camera: true,
                            is_video: false,
                        },
                    },
                })
            }
        }
    }

    pub fn next(&mut self) -> Option<BgrImage> {
        match &mut self.inner {
            PipedInner::Thread { rx, .. } => rx.recv().ok().flatten(),
            PipedInner::Local { src, buf } => {
                if src.read_into(buf).ok()? {
                    let mut out = BgrImage::zeros(0, 0);
                    std::mem::swap(&mut out, buf);
                    Some(out)
                } else {
                    None
                }
            }
        }
    }

    pub fn recycle(&mut self, buf: BgrImage) {
        match &mut self.inner {
            PipedInner::Thread { recycle, .. } => {
                if let Some(tx) = recycle {
                    let _ = tx.try_send(buf);
                }
            }
            PipedInner::Local { buf: slot, .. } => {
                *slot = buf;
            }
        }
    }
}

enum Worker {
    Safe {
        inner: SafeInner,
        width: u32,
        height: u32,
    },
    Camera {
        idx: u32,
        width: u32,
        height: u32,
        fps: u32,
    },
}

enum Live {
    Safe(SafeInner),
    Cam(nokhwa::Camera),
}

impl Live {
    fn read_into(&mut self, dst: &mut BgrImage) -> Result<bool> {
        match self {
            Live::Safe(s) => s.read_into(dst),
            Live::Cam(c) => read_camera(c, dst),
        }
    }
}

fn spawn_worker(job: Worker) -> Result<PipedInput> {
    let (w, h) = match &job {
        Worker::Safe { width, height, .. } | Worker::Camera { width, height, .. } => {
            (*width, *height)
        }
    };
    let (w, h) = (w.max(1), h.max(1));
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let (rec_tx, rec_rx) = std::sync::mpsc::sync_channel(2);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let _ = rec_tx.send(BgrImage::zeros(w, h));
    let _ = rec_tx.send(BgrImage::zeros(w, h));
    let join = std::thread::Builder::new()
        .name("osf-capture".into())
        .spawn(move || {
            let mut live = match job {
                Worker::Safe { inner, .. } => {
                    let _ = ready_tx.send(Ok(()));
                    Live::Safe(inner)
                }
                Worker::Camera {
                    idx,
                    width,
                    height,
                    fps,
                } => match open_camera(idx, width, height, fps) {
                    Ok(cam) => {
                        let _ = ready_tx.send(Ok(()));
                        Live::Cam(cam)
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                },
            };
            loop {
                let mut buf = match rec_rx.recv() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                match live.read_into(&mut buf) {
                    Ok(true) => {
                        if tx.send(Some(buf)).is_err() {
                            break;
                        }
                    }
                    _ => {
                        let _ = tx.send(None);
                        break;
                    }
                }
            }
        })
        .context("capture thread")?;
    ready_rx.recv().context("capture handshake")??;
    Ok(PipedInput {
        inner: PipedInner::Thread {
            rx,
            recycle: Some(rec_tx),
            join: Some(join),
        },
    })
}

impl Drop for PipedInput {
    fn drop(&mut self) {
        if let PipedInner::Thread { recycle, join, .. } = &mut self.inner {
            recycle.take();
            if let Some(h) = join.take() {
                let _ = h.join();
            }
        }
    }
}
