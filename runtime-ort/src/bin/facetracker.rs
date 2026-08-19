use std::net::UdpSocket;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    draw_tracking, dump_symmetric_points, encode_faces_into, encode_vmc, list_cameras,
    model_base_path, Device, ExtListener, FacePacket, FilterKind, InputSource, OutputDriver,
    PipedInput, Tracker, TrackerConfig, VideoOut, VizWindow, VrmCfg, VrmDriver, PACKET_FRAME_SIZE,
};

#[derive(Parser, Debug)]
#[command(name = "facetracker", about = "OpenSeeFace tracker (Rust ORT)")]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1")]
    ip: String,
    #[arg(short, long, default_value_t = 11573)]
    port: u16,
    #[arg(short = 'l', long, default_value_t = 0)]
    list_cameras: i32,
    #[arg(short = 'W', long, default_value_t = 640)]
    width: u32,
    #[arg(short = 'H', long, default_value_t = 360)]
    height: u32,
    #[arg(short = 'F', long, default_value_t = 24)]
    fps: u32,
    #[arg(short, long, default_value = "0")]
    capture: String,
    #[arg(short = 'M', long)]
    mirror_input: bool,
    #[arg(short, long, default_value_t = 1)]
    max_threads: usize,
    #[arg(short, long)]
    threshold: Option<f32>,
    #[arg(short = 'd', long, default_value_t = 0.6)]
    detection_threshold: f32,
    #[arg(short, long, default_value_t = 0)]
    visualize: i32,
    #[arg(short = 'P', long, default_value_t = 0)]
    pnp_points: i32,
    #[arg(short, long, default_value_t = 0)]
    silent: i32,
    #[arg(long, default_value_t = 1)]
    faces: usize,
    #[arg(long, default_value_t = 0)]
    scan_retinaface: i32,
    #[arg(long, default_value_t = 3)]
    scan_every: i32,
    #[arg(long, default_value_t = 10)]
    discard_after: i32,
    #[arg(long, default_value_t = 900.0)]
    max_feature_updates: f32,
    #[arg(long, default_value_t = 1)]
    no_3d_adapt: i32,
    #[arg(long, default_value_t = 0)]
    try_hard: i32,
    #[arg(long)]
    video_out: Option<String>,
    #[arg(long, default_value_t = 1)]
    video_scale: u32,
    #[arg(long, default_value_t = 24.0)]
    video_fps: f32,
    #[arg(long, default_value_t = 0)]
    raw_rgb: i32,
    #[arg(long, default_value_t = 3)]
    model: i32,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    gaze_tracking: i32,
    #[arg(long, default_value_t = 0)]
    face_id_offset: i32,
    #[arg(long, default_value_t = 0)]
    repeat_video: i32,
    #[arg(long, default_value = "")]
    dump_points: String,
    /// Output post-process: none | one-euro
    #[arg(long, default_value = "one-euro")]
    filter: String,
    #[arg(long, default_value_t = 1.0)]
    filter_mincutoff: f32,
    #[arg(long, default_value_t = 0.007)]
    filter_beta: f32,
    /// cpu | gpu (CoreML on Apple, CUDA on NVIDIA). GPU needs `--features gpu`.
    #[arg(long, default_value_t = Device::Cpu)]
    device: Device,
    #[arg(long, default_value_t = 0, hide = true)]
    benchmark: i32,
    /// OpenSeeLauncher sends this with `--benchmark`; ignored.
    #[arg(long, hide = true)]
    #[allow(dead_code)]
    priority: Option<i32>,
    /// Send VMC Protocol (OSC) for Unity EVMC4U / UniVRM. 0 disables.
    #[arg(long, default_value_t = 1)]
    vmc: i32,
    #[arg(long, default_value = "127.0.0.1")]
    vmc_ip: String,
    #[arg(long, default_value_t = 39539)]
    vmc_port: u16,
    /// ARKit Perfect Sync blendshapes. 0 = VRM 0.x presets (A/I/U/E/O, Blink, Look*).
    #[arg(long, default_value_t = 1)]
    vrm_perfect_sync: i32,
    #[arg(long, default_value_t = 0)]
    vrm_mirror: i32,
    /// Listen for `/OSF/Ext/Visemes` and `/OSF/Ext/Expression` (OVRLipSync / SVM sidecar). 0 disables.
    #[arg(long, default_value_t = 39540)]
    osf_ext_listen: u16,
}

/// Analog eye openness for the tracker console (`eye_blink` in 0..=1).
/// Glyphs: O open, o half, - slit, x shut.
fn format_eye_open(open: f32) -> String {
    let g = if open > 0.75 {
        'O'
    } else if open > 0.40 {
        'o'
    } else if open > 0.15 {
        '-'
    } else {
        'x'
    };
    format!("{g} {open:.2}")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_cameras > 0 {
        for (i, name) in list_cameras().context("list cameras")? {
            if args.list_cameras == 1 {
                println!("{i}: {name}");
            } else {
                println!("{name}");
            }
        }
        return Ok(());
    }
    if args.benchmark > 0 {
        return run_benchmark(&args);
    }

    let src = InputSource::open(
        &args.capture,
        args.raw_rgb != 0,
        args.width,
        args.height,
        args.fps,
        args.repeat_video != 0,
    )?;
    let is_video = src.is_video;
    let mut input = PipedInput::start(src)?;
    let sock = UdpSocket::bind("0.0.0.0:0").context("udp bind")?;
    let dest = format!("{}:{}", args.ip, args.port);
    let vmc_dest = format!("{}:{}", args.vmc_ip, args.vmc_port);
    let filter: FilterKind = args.filter.parse().context("--filter")?;
    let pace = (args.fps > 0 && !is_video).then(|| Duration::from_secs_f64(1.0 / args.fps as f64));
    let mut vrm = VrmDriver::new(VrmCfg {
        perfect_sync: args.vrm_perfect_sync != 0,
        mirror: args.vrm_mirror != 0,
        ..VrmCfg::default()
    });
    let mut output = OutputDriver::new();
    let mut ext_listen = if args.osf_ext_listen != 0 {
        match ExtListener::bind(([0, 0, 0, 0], args.osf_ext_listen).into()) {
            Ok(l) => Some(l),
            Err(e) => {
                if args.silent == 0 {
                    eprintln!("OSF ext listen :{} failed: {e}", args.osf_ext_listen);
                }
                None
            }
        }
    } else {
        None
    };

    let mut tracker = None;
    let mut viz = None;
    let mut vout = None;
    let mut dump = None;
    let mut tracking_time = 0.0;
    let mut total_tracking_time = 0.0;
    let mut tracking_frames = 0u64;

    let mut udp_buf = Vec::with_capacity(PACKET_FRAME_SIZE);
    loop {
        let tick = Instant::now();
        let Some(mut frame) = input.next() else {
            if args.repeat_video != 0 && is_video {
                if let Ok(src) = InputSource::open(
                    &args.capture,
                    args.raw_rgb != 0,
                    args.width,
                    args.height,
                    args.fps,
                    true,
                ) {
                    input = PipedInput::start(src)?;
                    continue;
                }
            }
            break;
        };
        if args.mirror_input {
            frame.flip_h_in_place();
        }
        let tnow = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        if tracker.is_none() {
            tracker = Some(Tracker::new(TrackerConfig {
                width: frame.width,
                height: frame.height,
                model_type: args.model,
                detection_threshold: args.detection_threshold,
                threshold: args.threshold,
                max_faces: args.faces.max(1),
                discard_after: args.discard_after,
                scan_every: args.scan_every,
                max_threads: args.max_threads,
                silent: args.silent != 0,
                model_dir: args.model_dir.clone(),
                no_gaze: args.gaze_tracking == 0 || args.model == -1,
                use_retinaface: args.scan_retinaface != 0,
                max_feature_updates: args.max_feature_updates,
                static_model: args.no_3d_adapt == 1,
                try_hard: args.try_hard == 1,
                filter,
                filter_mincutoff: args.filter_mincutoff,
                filter_beta: args.filter_beta,
                device: args.device,
                ..TrackerConfig::default()
            })?);
            if args.visualize != 0 {
                viz = VizWindow::open(frame.width, frame.height).ok();
            }
            if let Some(path) = &args.video_out {
                let s = args.video_scale.max(1);
                vout = VideoOut::open(path, frame.width * s, frame.height * s, args.video_fps).ok();
            }
        }
        let tracker = tracker.as_mut().unwrap();
        tracker.set_size(frame.width, frame.height);

        let t0 = Instant::now();
        let faces = tracker.predict(&frame);
        if !faces.is_empty() {
            let dt = t0.elapsed().as_secs_f64();
            total_tracking_time += dt;
            tracking_time += dt / faces.len() as f64;
            tracking_frames += 1;
        }

        let packets: Vec<FacePacket> = faces
            .iter()
            .map(|f| {
                if args.silent == 0 {
                    println!(
                        "Confidence[{}]: {:.4} / 3D fitting error: {:.4} / Eyes L,R: {}, {}",
                        f.id + args.face_id_offset,
                        f.conf,
                        f.pnp_error,
                        format_eye_open(f.eye_blink[1]),
                        format_eye_open(f.eye_blink[0]),
                    );
                }
                FacePacket::from_face(
                    f,
                    tnow,
                    frame.width as f32,
                    frame.height as f32,
                    f.id + args.face_id_offset,
                )
            })
            .collect();
        if !packets.is_empty() && packets.len() < 40 {
            encode_faces_into(&mut udp_buf, &packets);
            if udp_buf.len() % PACKET_FRAME_SIZE == 0 {
                let _ = sock.send_to(&udp_buf, &dest);
            }
        }
        if args.vmc != 0 {
            if let Some(pkt) = packets
                .iter()
                .find(|p| p.success)
                .or_else(|| packets.first())
            {
                if let Some(out) = output.update(pkt, ext_listen.as_mut().map(|l| l.poll())) {
                    if let Some(frame) = vrm.map(&out) {
                        if let Ok(buf) = encode_vmc(&frame) {
                            let _ = sock.send_to(&buf, &vmc_dest);
                        }
                    }
                }
            }
        }

        if args.visualize != 0 || vout.is_some() {
            let cam = osf_ort::Camera::from_frame(frame.width, frame.height);
            draw_tracking(&mut frame, &faces, args.visualize, args.pnp_points, &cam);
        }
        if let Some(v) = vout.as_mut() {
            let s = args.video_scale.max(1);
            let scaled = if s > 1 {
                osf_ort::resize_bgr(&frame, frame.width * s, frame.height * s)
            } else {
                frame.clone()
            };
            let _ = v.write(&scaled);
        }
        if let Some(f) = faces.first() {
            dump = Some(f.face_3d.clone());
        }
        if let Some(w) = viz.as_mut() {
            if !w.show(&frame) {
                break;
            }
        }
        if let Some(d) = pace {
            if let Some(rest) = d.checked_sub(tick.elapsed()) {
                std::thread::sleep(rest);
            }
        }
        input.recycle(frame);
    }

    if !args.dump_points.is_empty() {
        if let Some(pts) = dump {
            dump_symmetric_points(&pts, &args.dump_points)?;
        }
    }
    if args.silent == 0 && tracking_frames > 0 {
        println!(
            "Average tracking time per detected face: {:.2} ms",
            1000.0 * tracking_time / tracking_frames as f64
        );
        println!(
            "Tracking time: {:.3} s\nFrames: {tracking_frames}",
            total_tracking_time
        );
    }
    Ok(())
}

fn run_benchmark(args: &Args) -> Result<()> {
    let dir = model_base_path(args.model_dir.as_deref());
    let im = osf_ort::BgrImage::load(dir.join("benchmark.bin"))?;
    for model_type in [3, 2, 1, 0, -1, -2, -3] {
        let mut tracker = Tracker::new(TrackerConfig {
            width: 224,
            height: 224,
            threshold: Some(0.1),
            max_threads: args.max_threads,
            max_faces: 1,
            discard_after: 0,
            scan_every: 0,
            silent: true,
            model_type,
            model_dir: args.model_dir.clone(),
            no_gaze: model_type == -1,
            detection_threshold: 0.1,
            max_feature_updates: 900.0,
            static_model: args.no_3d_adapt == 1,
            filter: FilterKind::None,
            device: args.device,
            ..TrackerConfig::default()
        })?;
        let mut total = 0.0;
        for _ in 0..100 {
            let t = Instant::now();
            let _ = tracker.predict(&im);
            total += t.elapsed().as_secs_f64();
        }
        println!("{}", 1.0 / (total / 100.0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::format_eye_open;

    #[test]
    fn eye_console_shows_analog_levels() {
        assert_eq!(format_eye_open(0.95), "O 0.95");
        assert_eq!(format_eye_open(0.52), "o 0.52");
        assert_eq!(format_eye_open(0.22), "- 0.22");
        assert_eq!(format_eye_open(0.05), "x 0.05");
    }
}
