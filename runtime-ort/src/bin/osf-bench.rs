use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    cosine, detect_faces, imagenet_nchw, max_abs, mean_abs, model_path, read_f32_le, retina_nchw,
    rss, BgrImage, Latency, LmSpec, OrtModel, VERSION,
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
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    crate_version: &'static str,
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
    let frame = BgrImage::load(&args.image)?;
    let spec = LmSpec::from_type(args.model)?;
    let mut models = HashMap::new();

    let det_in = match args.ref_dir.as_ref() {
        Some(d) => osf_ort::TensorF32 {
            shape: vec![1, 3, 224, 224],
            data: read_f32_le(d.join("detection_input.bin"))?,
        },
        None => imagenet_nchw(&frame, 224),
    };
    models.insert(
        "detection".into(),
        bench(
            &model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
            "mnv3_detection_opt.onnx",
            &det_in,
            args.threads,
            args.warmup,
            args.iters,
            args.ref_dir
                .as_ref()
                .map(|d| d.join("detection_output_0.bin")),
        )?,
    );

    let lm_in = match args.ref_dir.as_ref() {
        Some(d) if d.join("landmarks_input.bin").is_file() => osf_ort::TensorF32 {
            shape: vec![1, 3, spec.size as i64, spec.size as i64],
            data: read_f32_le(d.join("landmarks_input.bin"))?,
        },
        _ => imagenet_nchw(&frame, spec.size),
    };
    models.insert(
        spec.file.trim_end_matches(".onnx").into(),
        bench(
            &model_path(&args.models_dir, spec.file),
            spec.file,
            &lm_in,
            args.threads,
            args.warmup,
            args.iters,
            args.ref_dir
                .as_ref()
                .map(|d| d.join("landmarks_output.bin")),
        )?,
    );

    let gaze = model_path(&args.models_dir, "mnv3_gaze32_split_opt.onnx");
    if gaze.is_file() {
        let gin = match args.ref_dir.as_ref() {
            Some(d) if d.join("gaze_input.bin").is_file() => osf_ort::TensorF32 {
                shape: vec![2, 3, 32, 32],
                data: read_f32_le(d.join("gaze_input.bin"))?,
            },
            _ => osf_ort::TensorF32 {
                shape: vec![2, 3, 32, 32],
                data: vec![0.0; 2 * 3 * 32 * 32],
            },
        };
        models.insert(
            "gaze".into(),
            bench(
                &gaze,
                "mnv3_gaze32_split_opt.onnx",
                &gin,
                args.threads,
                args.warmup,
                args.iters,
                args.ref_dir.as_ref().map(|d| d.join("gaze_output.bin")),
            )?,
        );
    }

    let rf = model_path(&args.models_dir, "retinaface_640x640_opt.onnx");
    if rf.is_file() {
        let rin = match args.ref_dir.as_ref() {
            Some(d) if d.join("retinaface_input.bin").is_file() => osf_ort::TensorF32 {
                shape: vec![1, 3, 640, 640],
                data: read_f32_le(d.join("retinaface_input.bin"))?,
            },
            _ => retina_nchw(&frame),
        };
        models.insert(
            "retinaface".into(),
            bench(
                &rf,
                "retinaface_640x640_opt.onnx",
                &rin,
                args.threads,
                args.warmup,
                args.iters,
                args.ref_dir
                    .as_ref()
                    .map(|d| d.join("retinaface_output_0.bin")),
            )?,
        );
    }

    let pipeline = pipeline(&args, &frame, spec)?;
    let report = Report {
        backend: "ort-rust",
        crate_version: VERSION,
        threads: args.threads,
        models,
        pipeline,
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
    input: &osf_ort::TensorF32,
    threads: usize,
    warmup: u32,
    iters: u32,
    ref_out: Option<PathBuf>,
) -> Result<ModelReport> {
    let mut m = OrtModel::load(path, threads).with_context(|| path.display().to_string())?;
    let t0 = Instant::now();
    let first = m.run(input)?;
    let first_infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..warmup {
        m.run(input)?;
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        m.run(input)?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let accuracy = ref_out
        .filter(|p| p.is_file())
        .map(|p| read_f32_le(p))
        .transpose()?
        .map(|r| {
            let n = first[0].data.len().min(r.len());
            Acc {
                compared_elems: n,
                max_abs: max_abs(&first[0].data[..n], &r[..n]),
                mean_abs: mean_abs(&first[0].data[..n], &r[..n]),
                cosine: cosine(&first[0].data[..n], &r[..n]),
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

fn pipeline(args: &Args, frame: &BgrImage, spec: LmSpec) -> Result<Pipeline> {
    let mut det = OrtModel::load(
        model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
        args.threads,
    )?;
    let t0 = Instant::now();
    let din = imagenet_nchw(frame, 224);
    let dout = det.run(&din)?;
    let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut landmarks_ms = 0.0;
    let mut faces = 0;
    let mut det_iou = None;
    let mut landmark_mae_px = None;
    if let Some(d) = dets.first() {
        faces = 1;
        let (x1, y1, x2, y2) = crop_box(frame, d);
        if x2 - x1 >= 4 && y2 - y1 >= 4 {
            let mut lm = OrtModel::load(model_path(&args.models_dir, spec.file), args.threads)?;
            let t1 = Instant::now();
            let crop = crop_img(frame, x1, y1, x2, y2);
            let lin = imagenet_nchw(&crop, spec.size);
            let _ = lm.run(&lin)?;
            landmarks_ms = t1.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(dir) = &args.ref_dir {
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
                    if let Some(lms) = v["landmarks"].as_array() {
                        if !lms.is_empty() {
                            let crop = crop_img(
                                frame,
                                crop_box(frame, d).0,
                                crop_box(frame, d).1,
                                crop_box(frame, d).2,
                                crop_box(frame, d).3,
                            );
                            let mut lm = OrtModel::load(
                                model_path(&args.models_dir, spec.file),
                                args.threads,
                            )?;
                            let (x1, y1, x2, y2) = crop_box(frame, d);
                            let lin = imagenet_nchw(&crop, spec.size);
                            let out = lm.run(&lin)?;
                            let scale_x = (x2 - x1) as f32 / spec.size as f32;
                            let scale_y = (y2 - y1) as f32 / spec.size as f32;
                            let (_, pts) = osf_ort::decode_landmarks(
                                &out[0],
                                [x1 as f32, y1 as f32, scale_x, scale_y],
                                spec,
                            );
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
        }
    }
    Ok(Pipeline {
        faces,
        detect_ms,
        landmarks_ms,
        e2e_ms: detect_ms + landmarks_ms,
        det_iou,
        landmark_mae_px,
    })
}

fn crop_box(frame: &BgrImage, d: &[f32; 5]) -> (i32, i32, i32, i32) {
    let (x, y, w, h) = (d[0], d[1], d[2], d[3]);
    let x1 = x - (w * 0.1) as i32 as f32;
    let y1 = y - (h * 0.125) as i32 as f32;
    let x2 = x + w + (w * 0.1) as i32 as f32;
    let y2 = y + h + (h * 0.125) as i32 as f32;
    let clamp = |px: f32, py: f32| {
        let px = px.clamp(0.0, frame.width as f32 - 1.0) as i32;
        let py = py.clamp(0.0, frame.height as f32 - 1.0) as i32 + 1;
        (px, py)
    };
    let (x1, y1) = clamp(x1, y1);
    let (x2, y2) = clamp(x2, y2);
    (x1, y1, x2, y2)
}

fn crop_img(im: &BgrImage, x1: i32, y1: i32, x2: i32, y2: i32) -> BgrImage {
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = (x2.max(0) as u32).min(im.width);
    let y2 = (y2.max(0) as u32).min(im.height);
    let w = x2.saturating_sub(x1);
    let h = y2.saturating_sub(y1);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in y1..y2 {
        let s = ((y * im.width + x1) * 3) as usize;
        data.extend_from_slice(&im.data[s..s + (w * 3) as usize]);
    }
    BgrImage {
        width: w,
        height: h,
        data,
    }
}

fn iou(a: &[f32; 5], b: &[f32; 4]) -> f32 {
    let (ax2, ay2) = (a[0] + a[2], a[1] + a[3]);
    let (bx2, by2) = (b[0] + b[2], b[1] + b[3]);
    let l = a[0].max(b[0]);
    let t = a[1].max(b[1]);
    let r = ax2.min(bx2);
    let bot = ay2.min(by2);
    if r <= l || bot <= t {
        return 0.0;
    }
    let inter = (r - l) * (bot - t);
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}
