use std::net::UdpSocket;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    dump_symmetric_points, draw_tracking, encode_faces, list_cameras, mirror_bgr, model_base_path,
    FacePacket, InputSource, Tracker, TrackerConfig, VideoOut, VizWindow, PACKET_FRAME_SIZE,
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
    #[arg(long, default_value_t = 0, hide = true)]
    benchmark: i32,
    /// OpenSeeLauncher sends this with `--benchmark`; ignored.
    #[arg(long, hide = true)]
    #[allow(dead_code)]
    priority: Option<i32>,
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

    let mut input = InputSource::open(
        &args.capture,
        args.raw_rgb != 0,
        args.width,
        args.height,
        args.fps,
        args.repeat_video != 0,
    )?;
    let sock = UdpSocket::bind("0.0.0.0:0").context("udp bind")?;
    let dest = format!("{}:{}", args.ip, args.port);
    let pace = (args.fps > 0 && !input.is_video).then(|| Duration::from_secs_f64(1.0 / args.fps as f64));

    let mut tracker = None;
    let mut viz = None;
    let mut vout = None;
    let mut dump = None;
    let mut tracking_time = 0.0;
    let mut total_tracking_time = 0.0;
    let mut tracking_frames = 0u64;

    loop {
        let tick = Instant::now();
        let Some(mut frame) = input.read()? else {
            if args.repeat_video != 0 && input.is_video {
                input = InputSource::open(
                    &args.capture,
                    args.raw_rgb != 0,
                    args.width,
                    args.height,
                    args.fps,
                    true,
                )?;
                continue;
            }
            break;
        };
        if args.mirror_input {
            frame = mirror_bgr(&frame);
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
                    let r = if f.eye_blink[0] > 0.30 { "O" } else { "-" };
                    let l = if f.eye_blink[1] > 0.30 { "O" } else { "-" };
                    println!(
                        "Confidence[{}]: {:.4} / 3D fitting error: {:.4} / Eyes: {}, {}",
                        f.id + args.face_id_offset,
                        f.conf,
                        f.pnp_error,
                        l,
                        r
                    );
                }
                FacePacket {
                    time: tnow,
                    id: f.id + args.face_id_offset,
                    width: frame.width as f32,
                    height: frame.height as f32,
                    eye_blink: f.eye_blink,
                    success: f.success,
                    pnp_error: f.pnp_error,
                    quaternion: f.quaternion,
                    euler: f.euler,
                    translation: f.translation,
                    lms: f.lms.clone(),
                    pts_3d: f.pts_3d,
                    features: f.current_features,
                }
            })
            .collect();
        if !packets.is_empty() && packets.len() < 40 {
            let bytes = encode_faces(&packets);
            if bytes.len() % PACKET_FRAME_SIZE == 0 {
                let _ = sock.send_to(&bytes, &dest);
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
