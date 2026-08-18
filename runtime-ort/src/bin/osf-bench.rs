use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    cosine, crop_box, crop_img, detect_faces, imagenet_nchw, iou, max_abs, mean_abs, model_path,
    read_f32_le, retina_nchw, rss, BgrImage, Device, GpuTracker, Latency, LmSpec, OrtModel,
    TensorF16, VERSION,
};
use serde::Serialize;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    #[arg(long, default_value = "models/benchmark.bin")]
    image: PathBuf,
    #[arg(long, default_value_t = 3)]
    model: i32,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 8)]
    warmup: u32,
    #[arg(long, default_value_t = 30)]
    iters: u32,
    /// cpu | gpu (CoreML on Apple, CUDA on NVIDIA)
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    crate_version: &'static str,
    device: String,
    ort_dylib: Option<String>,
    threads: usize,
    models: HashMap<String, ModelReport>,
    pipeline: Pipeline,
}

#[derive(Serialize)]
struct ModelReport {
    filename: String,
    startup_ms: f64,
    first_infer_ms: f64,
    latency: Latency,
    resources_after_infer: osf_ort::Rss,
    accuracy: Option<Acc>,
}

#[derive(Serialize)]
struct Acc {
    compared_elems: usize,
    max_abs: f32,
    mean_abs: f32,
    cosine: f32,
}

#[derive(Serialize)]
struct Pipeline {
    faces: usize,
    detect_ms: f64,
    landmarks_ms: f64,
    e2e_ms: f64,
    det_iou: Option<f32>,
    landmark_mae_px: Option<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::from_str(&args.device)?;
    let frame = BgrImage::load(&args.image)?;
    let spec = LmSpec::from_type(args.model)?;
    let mut models = HashMap::new();

    let dump = |name: &str| args.ref_dir.as_ref().map(|d| d.join(name));
    let load = |name: &str, shape: Vec<i64>, fb: TensorF16| -> Result<TensorF16> {
        match dump(name) {
            Some(p) if p.is_file() => Ok(TensorF16::from_f32(shape, read_f32_le(p)?)),
            _ => Ok(fb),
        }
    };
    let mut go = |key: &str, file: &str, input: &TensorF16, out: &str| -> Result<()> {
        models.insert(
            key.into(),
            bench(
                &model_path(&args.models_dir, file),
                file,
                input,
                args.threads,
                device,
                args.warmup,
                args.iters,
                dump(out),
            )?,
        );
        Ok(())
    };

    go(
        "detection",
        "mnv3_detection_opt.onnx",
        &load(
            "detection_input.bin",
            vec![1, 3, 224, 224],
            imagenet_nchw(&frame, 224),
        )?,
        "detection_output_0.bin",
    )?;
    go(
        spec.file.trim_end_matches(".onnx"),
        spec.file,
        &load(
            "landmarks_input.bin",
            vec![1, 3, spec.size as i64, spec.size as i64],
            imagenet_nchw(&frame, spec.size),
        )?,
        "landmarks_output.bin",
    )?;
    if model_path(&args.models_dir, "mnv3_gaze32_split_opt.onnx").is_file() {
        go(
            "gaze",
            "mnv3_gaze32_split_opt.onnx",
            &load(
                "gaze_input.bin",
                vec![2, 3, 32, 32],
                TensorF16::zeros(vec![2, 3, 32, 32]),
            )?,
            "gaze_output.bin",
        )?;
    }
    if model_path(&args.models_dir, "retinaface_640x640_opt.onnx").is_file() {
        go(
            "retinaface",
            "retinaface_640x640_opt.onnx",
            &load(
                "retinaface_input.bin",
                vec![1, 3, 640, 640],
                retina_nchw(&frame),
            )?,
            "retinaface_output_0.bin",
        )?;
    }

    let report = Report {
        backend: "ort-rust",
        crate_version: VERSION,
        device: device.as_str().into(),
        ort_dylib: std::env::var("ORT_DYLIB_PATH").ok(),
        threads: args.threads,
        models,
        pipeline: pipeline(&args, &frame, spec, device)?,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(p) = &args.out {
        if let Some(dir) = p.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(p, &json)?;
    }
    println!("{json}");
    Ok(())
}

fn bench(
    path: &std::path::Path,
    filename: &str,
    input: &TensorF16,
    threads: usize,
    device: Device,
    warmup: u32,
    iters: u32,
    ref_out: Option<PathBuf>,
) -> Result<ModelReport> {
    let batch = input.shape.first().copied().unwrap_or(1).max(1);
    let mut m =
        OrtModel::open(path, threads, device, batch).with_context(|| path.display().to_string())?;
    let t0 = Instant::now();
    let first = m.run(input)?;
    let first_infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..warmup {
        m.infer()?;
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        m.infer()?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let accuracy = ref_out
        .filter(|p| p.is_file())
        .map(|p| read_f32_le(p))
        .transpose()?
        .map(|r| {
            let got = first[0].to_f32();
            let n = got.len().min(r.len());
            Acc {
                compared_elems: n,
                max_abs: max_abs(&got[..n], &r[..n]),
                mean_abs: mean_abs(&got[..n], &r[..n]),
                cosine: cosine(&got[..n], &r[..n]),
            }
        });
    Ok(ModelReport {
        filename: filename.into(),
        startup_ms: m.load_ms,
        first_infer_ms,
        latency: Latency::from_samples(warmup, &samples),
        resources_after_infer: rss(),
        accuracy,
    })
}

fn pipeline(args: &Args, frame: &BgrImage, spec: LmSpec, device: Device) -> Result<Pipeline> {
    if device == Device::Gpu {
        return gpu_pipeline(args, frame, spec);
    }
    cpu_pipeline(args, frame, spec, device)
}

fn cpu_pipeline(args: &Args, frame: &BgrImage, spec: LmSpec, device: Device) -> Result<Pipeline> {
    let mut det = OrtModel::open(
        model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
        args.threads,
        device,
        1,
    )?;
    let mut lm = OrtModel::open(
        model_path(&args.models_dir, spec.file),
        args.threads,
        device,
        1,
    )?;
    let din = imagenet_nchw(frame, 224);
    let mut crop_lin = None;
    let dout = det.run(&din)?;
    let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
    if let Some(d) = dets.first() {
        let (x1, y1, x2, y2) = crop_box(frame, d);
        if x2 - x1 >= 4 && y2 - y1 >= 4 {
            let crop = crop_img(frame, x1, y1, x2, y2);
            crop_lin = Some((x1, y1, x2, y2, imagenet_nchw(&crop, spec.size)));
            let _ = lm.run(&crop_lin.as_ref().unwrap().4)?;
        }
    }
    for _ in 0..args.warmup.max(1) {
        let _ = det.run(&din)?;
        if let Some((_, _, _, _, lin)) = &crop_lin {
            let _ = lm.run(lin)?;
        }
    }

    let t0 = Instant::now();
    let dout = det.run(&din)?;
    let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut landmarks_ms = 0.0;
    let mut faces = 0;
    let mut pts = None;
    if let Some(d) = dets.first() {
        faces = 1;
        let (x1, y1, x2, y2) = crop_box(frame, d);
        if x2 - x1 >= 4 && y2 - y1 >= 4 {
            let t1 = Instant::now();
            let crop = crop_img(frame, x1, y1, x2, y2);
            let lin = imagenet_nchw(&crop, spec.size);
            let out = lm.run(&lin)?;
            landmarks_ms = t1.elapsed().as_secs_f64() * 1000.0;
            if args.ref_dir.is_some() {
                let scale_x = (x2 - x1) as f32 / spec.size as f32;
                let scale_y = (y2 - y1) as f32 / spec.size as f32;
                pts = Some(
                    osf_ort::decode_landmarks(
                        &out[0],
                        [x1 as f32, y1 as f32, scale_x, scale_y],
                        spec,
                    )
                    .1,
                );
            }
        }
    }
    Ok(finish_pipeline(
        args,
        dets.first(),
        pts.as_deref(),
        faces,
        detect_ms,
        landmarks_ms,
    ))
}

fn gpu_pipeline(args: &Args, frame: &BgrImage, spec: LmSpec) -> Result<Pipeline> {
    let mut tracker = GpuTracker::open(&args.models_dir, spec, args.threads, frame)?;
    let mut dets = tracker.detect(frame)?;
    if let Some(d) = dets.first() {
        let _ = tracker.landmarks(frame, d, spec)?;
    }
    for _ in 0..args.warmup.max(1) {
        dets = tracker.detect(frame)?;
        if let Some(d) = dets.first() {
            let _ = tracker.landmarks(frame, d, spec)?;
        }
    }
    let t0 = Instant::now();
    let dets = tracker.detect(frame)?;
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut landmarks_ms = 0.0;
    let mut faces = 0;
    let mut pts = None;
    if let Some(d) = dets.first() {
        faces = 1;
        let t1 = Instant::now();
        let decoded = tracker.landmarks(frame, d, spec)?;
        landmarks_ms = t1.elapsed().as_secs_f64() * 1000.0;
        pts = Some(decoded.1);
    }
    Ok(finish_pipeline(
        args,
        dets.first(),
        pts.as_deref(),
        faces,
        detect_ms,
        landmarks_ms,
    ))
}

fn finish_pipeline(
    args: &Args,
    det: Option<&[f32; 5]>,
    pts: Option<&[[f32; 3]]>,
    faces: usize,
    detect_ms: f64,
    landmarks_ms: f64,
) -> Pipeline {
    let mut det_iou = None;
    let mut landmark_mae_px = None;
    if let (Some(d), Some(dir)) = (det, &args.ref_dir) {
        if let Ok(meta) = fs::read_to_string(dir.join("meta.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&meta) {
                if let Some(rd) = v["detections"].as_array().and_then(|a| a.first()) {
                    let b = [
                        rd["x"].as_f64().unwrap_or(0.0) as f32,
                        rd["y"].as_f64().unwrap_or(0.0) as f32,
                        rd["w"].as_f64().unwrap_or(0.0) as f32,
                        rd["h"].as_f64().unwrap_or(0.0) as f32,
                    ];
                    det_iou = Some(iou(d, &b));
                }
                if let (Some(pts), Some(lms)) = (pts, v["landmarks"].as_array()) {
                    let n = pts.len().min(lms.len());
                    if n > 0 {
                        let s: f32 = pts
                            .iter()
                            .zip(lms)
                            .take(n)
                            .map(|(p, q)| {
                                let qx = q["x"].as_f64().unwrap_or(0.0) as f32;
                                let qy = q["y"].as_f64().unwrap_or(0.0) as f32;
                                (p[0] - qx).hypot(p[1] - qy)
                            })
                            .sum();
                        landmark_mae_px = Some(s / n as f32);
                    }
                }
            }
        }
    }
    Pipeline {
        faces,
        detect_ms,
        landmarks_ms,
        e2e_ms: detect_ms + landmarks_ms,
        det_iou,
        landmark_mae_px,
    }
}
