use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    cosine, crop_box, crop_box_pad, crop_img, decode_landmarks, detect_faces, imagenet_nchw, iou,
    landmark_bbox, max_abs, mean_abs, mean_conf, model_path, read_f32_le, retina_nchw, rss,
    BgrImage, Device, GpuTracker, Latency, LmSpec, OrtModel, TensorF16, EYE_IDX, VERSION,
};
use serde::{Deserialize, Serialize};

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
    #[arg(long, default_value = "micro")]
    suite: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
    #[arg(long)]
    scenario_dir: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    crate_version: &'static str,
    device: String,
    ort_dylib: Option<String>,
    threads: usize,
    models: HashMap<String, ModelReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline: Option<Pipeline>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    scenarios: HashMap<String, ScenarioReport>,
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

#[derive(Clone, Debug, Deserialize)]
struct ScenarioMeta {
    name: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "one")]
    scan_every: u32,
    #[serde(default)]
    gaze: bool,
    #[serde(default = "pad_x_default")]
    pad_x: f32,
    #[serde(default = "pad_y_default")]
    pad_y: f32,
    #[serde(default)]
    glasses: bool,
    frames: Vec<String>,
}

fn one() -> u32 {
    1
}
fn pad_x_default() -> f32 {
    0.1
}
fn pad_y_default() -> f32 {
    0.125
}

#[derive(Serialize)]
struct ScenarioReport {
    name: String,
    tags: Vec<String>,
    frames: usize,
    scan_every: u32,
    gaze: bool,
    glasses: bool,
    detect_ms: Latency,
    crop_ms: Latency,
    pre_ms: Latency,
    lm_ms: Latency,
    decode_ms: Latency,
    gaze_ms: Latency,
    e2e_ms: Latency,
    scan_p50_ms: Option<f64>,
    track_p50_ms: Option<f64>,
    crop_w: u32,
    crop_h: u32,
    faces: usize,
    det_score: Option<f32>,
    lm_conf: Option<f32>,
    eye_conf: Option<f32>,
    gaze_conf: Option<f32>,
    landmarks: Vec<[f32; 3]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::from_str(&args.device)?;
    let spec = LmSpec::from_type(args.model)?;
    let run_micro = args.suite == "micro" || args.suite == "all";
    let run_real = args.suite == "realistic" || args.suite == "all";
    let mut models = HashMap::new();
    let mut pipe = None;

    if run_micro {
        let frame = BgrImage::load(&args.image)?;
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
        pipe = Some(pipeline(&args, &frame, spec, device)?);
    }

    let scenarios = if run_real {
        if let Some(dir) = &args.scenario_dir {
            run_scenario_dir(&args, spec, device, dir)?
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let report = Report {
        backend: "ort-rust",
        crate_version: VERSION,
        device: device.as_str().into(),
        ort_dylib: std::env::var("ORT_DYLIB_PATH").ok(),
        threads: args.threads,
        models,
        pipeline: pipe,
        scenarios,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(p) = &args.out {
        if let Some(dir) = p.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(p, &json)?;
    } else {
        println!("{json}");
    }
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
        let _ = tracker.landmarks(frame, d, spec, 0.1, 0.125)?;
    }
    for _ in 0..args.warmup.max(1) {
        dets = tracker.detect(frame)?;
        if let Some(d) = dets.first() {
            let _ = tracker.landmarks(frame, d, spec, 0.1, 0.125)?;
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
        let decoded = tracker.landmarks(frame, d, spec, 0.1, 0.125)?;
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

struct Row {
    detect_ms: f64,
    crop_ms: f64,
    pre_ms: f64,
    lm_ms: f64,
    decode_ms: f64,
    gaze_ms: f64,
    e2e_ms: f64,
    scanned: bool,
    faces: usize,
    det_score: Option<f32>,
    lm_conf: Option<f32>,
    eye_conf: Option<f32>,
    gaze_conf: Option<f32>,
    crop_w: u32,
    crop_h: u32,
    box5: Option<[f32; 5]>,
    pts: Vec<[f32; 3]>,
}

fn gaze_input(frame: &BgrImage, pts: &[[f32; 3]]) -> Option<TensorF16> {
    if pts.len() < 46 {
        return None;
    }
    let mut data = Vec::with_capacity(2 * 3 * 32 * 32);
    for &(a, b) in &[(36usize, 39usize), (42, 45)] {
        let (cx, cy) = ((pts[a][1] + pts[b][1]) * 0.5, (pts[a][0] + pts[b][0]) * 0.5);
        let rad = ((pts[b][1] - pts[a][1]).hypot(pts[b][0] - pts[a][0]) * 0.5 * 1.4).max(4.0);
        let crop = crop_img(
            frame,
            (cx - rad) as i32,
            (cy - rad * 0.86) as i32,
            (cx + rad) as i32,
            (cy + rad * 0.86) as i32,
        );
        if crop.width < 4 || crop.height < 4 {
            return None;
        }
        data.extend(imagenet_nchw(&crop, 32).data);
    }
    Some(TensorF16 {
        shape: vec![2, 3, 32, 32],
        data,
    })
}

fn mean_opt(vals: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let (n, s) = vals
        .flatten()
        .fold((0usize, 0.0), |(n, s), v| (n + 1, s + v));
    (n > 0).then(|| s / n as f32)
}

fn one_frame(
    mut cpu: Option<&mut (OrtModel, OrtModel)>,
    mut gpu: Option<&mut GpuTracker>,
    gaze: &mut Option<OrtModel>,
    frame: &BgrImage,
    spec: LmSpec,
    mut box5: Option<[f32; 5]>,
    pad_x: f32,
    pad_y: f32,
    do_detect: bool,
    do_gaze: bool,
) -> Result<Row> {
    let t_all = Instant::now();
    let mut detect_ms = 0.0;
    let mut det_score = None;
    if do_detect {
        let t = Instant::now();
        let dets = if let Some((det, _)) = cpu.as_mut() {
            let dout = det.run(&imagenet_nchw(frame, 224))?;
            detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6)
        } else if let Some(tr) = gpu.as_mut() {
            tr.detect(frame)?
        } else {
            Vec::new()
        };
        detect_ms = t.elapsed().as_secs_f64() * 1000.0;
        det_score = dets.first().map(|d| d[4]);
        box5 = dets.first().copied();
    }
    let miss = || Row {
        detect_ms,
        crop_ms: 0.0,
        pre_ms: 0.0,
        lm_ms: 0.0,
        decode_ms: 0.0,
        gaze_ms: 0.0,
        e2e_ms: t_all.elapsed().as_secs_f64() * 1000.0,
        scanned: do_detect,
        faces: 0,
        det_score,
        lm_conf: None,
        eye_conf: None,
        gaze_conf: None,
        crop_w: 0,
        crop_h: 0,
        box5: None,
        pts: Vec::new(),
    };
    let Some(d) = box5 else {
        return Ok(miss());
    };
    let t = Instant::now();
    let (x1, y1, x2, y2) = crop_box_pad(frame, &d, pad_x, pad_y);
    let (crop_w, crop_h) = ((x2 - x1).max(0) as u32, (y2 - y1).max(0) as u32);
    if crop_w < 4 || crop_h < 4 {
        return Ok(miss());
    }
    let crop_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let (conf, pts, pre_ms, lm_ms, decode_ms) = if let Some((_, lm)) = cpu.as_mut() {
        let lin = imagenet_nchw(&crop_img(frame, x1, y1, x2, y2), spec.size);
        let pre_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let out = lm.run(&lin)?;
        let lm_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let decoded = decode_landmarks(
            &out[0],
            [
                x1 as f32,
                y1 as f32,
                (x2 - x1) as f32 / spec.size as f32,
                (y2 - y1) as f32 / spec.size as f32,
            ],
            spec,
        );
        (
            decoded.0,
            decoded.1,
            pre_ms,
            lm_ms,
            t.elapsed().as_secs_f64() * 1000.0,
        )
    } else if let Some(tr) = gpu.as_mut() {
        let decoded = tr.landmarks(frame, &d, spec, pad_x, pad_y)?;
        (
            decoded.0,
            decoded.1,
            0.0,
            t.elapsed().as_secs_f64() * 1000.0,
            0.0,
        )
    } else {
        return Ok(miss());
    };
    let mut next = landmark_bbox(&pts).unwrap_or(d);
    next[4] = conf;
    let mut gconf = None;
    let mut gaze_ms = 0.0;
    if do_gaze {
        let t = Instant::now();
        if let (Some(g), Some(gin)) = (gaze.as_mut(), gaze_input(frame, &pts)) {
            gconf = Some(
                g.run(&gin)?[0]
                    .data
                    .iter()
                    .map(|x| x.to_f32())
                    .fold(0.0f32, f32::max),
            );
        }
        gaze_ms = t.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(Row {
        detect_ms,
        crop_ms,
        pre_ms,
        lm_ms,
        decode_ms,
        gaze_ms,
        e2e_ms: t_all.elapsed().as_secs_f64() * 1000.0,
        scanned: do_detect,
        faces: 1,
        det_score: det_score.or(Some(conf)),
        lm_conf: Some(conf),
        eye_conf: mean_conf(&pts, &EYE_IDX),
        gaze_conf: gconf,
        crop_w,
        crop_h,
        box5: Some(next),
        pts,
    })
}

fn lat(warmup: u32, rows: &[Row], f: impl Fn(&Row) -> f64) -> Latency {
    Latency::from_samples(warmup, &rows.iter().map(f).collect::<Vec<_>>())
}

fn run_scenario_dir(
    args: &Args,
    spec: LmSpec,
    device: Device,
    root: &Path,
) -> Result<HashMap<String, ScenarioReport>> {
    let mut out = HashMap::new();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)
        .with_context(|| root.display().to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.join("meta.json").is_file())
        .collect();
    dirs.sort();
    let mut gaze = model_path(&args.models_dir, "mnv3_gaze32_split_opt.onnx")
        .is_file()
        .then(|| {
            OrtModel::open(
                model_path(&args.models_dir, "mnv3_gaze32_split_opt.onnx"),
                args.threads,
                device,
                2,
            )
        })
        .transpose()?;
    let mut cpu = if device == Device::Cpu {
        Some((
            OrtModel::open(
                model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
                args.threads,
                device,
                1,
            )?,
            OrtModel::open(
                model_path(&args.models_dir, spec.file),
                args.threads,
                device,
                1,
            )?,
        ))
    } else {
        None
    };
    for dir in dirs {
        let meta: ScenarioMeta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?)?;
        let frames: Vec<BgrImage> = meta
            .frames
            .iter()
            .map(|f| BgrImage::load(dir.join(f)))
            .collect::<Result<_>>()?;
        if frames.is_empty() {
            continue;
        }
        let scan_every = meta.scan_every.max(1);
        let do_gaze = meta.gaze && gaze.is_some();
        let mut gpu = (device == Device::Gpu)
            .then(|| GpuTracker::open(&args.models_dir, spec, args.threads, &frames[0]))
            .transpose()?;
        let mut step = |box5: Option<[f32; 5]>, frame: &BgrImage, scanned: bool| {
            one_frame(
                cpu.as_mut(),
                gpu.as_mut(),
                &mut gaze,
                frame,
                spec,
                box5,
                meta.pad_x,
                meta.pad_y,
                scanned,
                do_gaze,
            )
        };
        for _ in 0..args.warmup.max(1) {
            let _ = step(None, &frames[0], true)?;
        }
        let mut box5 = None;
        let mut rows = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let scanned = box5.is_none() || (i as u32 % scan_every == 0);
            let row = step(if scanned { None } else { box5 }, frame, scanned)?;
            box5 = row.box5;
            rows.push(row);
        }
        let last = rows.last();
        let detect_s: Vec<f64> = rows
            .iter()
            .filter(|r| r.scanned)
            .map(|r| r.detect_ms)
            .collect();
        let scan: Vec<f64> = rows
            .iter()
            .filter(|r| r.scanned)
            .map(|r| r.e2e_ms)
            .collect();
        let track: Vec<f64> = rows
            .iter()
            .filter(|r| !r.scanned)
            .map(|r| r.e2e_ms)
            .collect();
        out.insert(
            meta.name.clone(),
            ScenarioReport {
                name: meta.name,
                tags: meta.tags,
                frames: rows.len(),
                scan_every,
                gaze: do_gaze,
                glasses: meta.glasses,
                detect_ms: Latency::from_samples(args.warmup, &detect_s),
                crop_ms: lat(args.warmup, &rows, |r| r.crop_ms),
                pre_ms: lat(args.warmup, &rows, |r| r.pre_ms),
                lm_ms: lat(args.warmup, &rows, |r| r.lm_ms),
                decode_ms: lat(args.warmup, &rows, |r| r.decode_ms),
                gaze_ms: lat(args.warmup, &rows, |r| r.gaze_ms),
                e2e_ms: lat(args.warmup, &rows, |r| r.e2e_ms),
                scan_p50_ms: (!scan.is_empty()).then(|| Latency::from_samples(0, &scan).p50_ms),
                track_p50_ms: (!track.is_empty()).then(|| Latency::from_samples(0, &track).p50_ms),
                crop_w: last.map(|r| r.crop_w).unwrap_or(0),
                crop_h: last.map(|r| r.crop_h).unwrap_or(0),
                faces: last.map(|r| r.faces).unwrap_or(0),
                det_score: mean_opt(rows.iter().map(|r| r.det_score)),
                lm_conf: mean_opt(rows.iter().map(|r| r.lm_conf)),
                eye_conf: mean_opt(rows.iter().map(|r| r.eye_conf)),
                gaze_conf: mean_opt(rows.iter().map(|r| r.gaze_conf)),
                landmarks: last.map(|r| r.pts.clone()).unwrap_or_default(),
            },
        );
    }
    Ok(out)
}
