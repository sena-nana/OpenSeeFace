use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    center_2x, cosine, crop_box, crop_box_pad, crop_img, decode_landmarks, det_window,
    detect_faces, enhance_bgr, face_crop, imagenet_nchw, iou, landmark_bbox, max_abs, mean_abs,
    mean_conf, model_path, nme, paste_bgr, pick_lm, read_f32_le, resize_bgr, retina_nchw, rss,
    synth_canvas, AdaptiveCfg, AdaptiveState, BgrImage, DetWindow, Device, EnhanceCfg, GpuTracker,
    Latency, LmSpec, OrtModel, TensorF16, EYE_IDX, FAST_LM, VERSION,
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
    /// micro | realistic | all | scale | enhance
    #[arg(long, default_value = "micro")]
    suite: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
    #[arg(long)]
    scenario_dir: Option<PathBuf>,
    /// CPU: zoomed detector ROI + 112/224 landmark ladder. GPU: ROI detect only.
    #[arg(long, default_value_t = false)]
    adaptive: bool,
    /// Gray-world WB + CLAHE/AHE before detect/landmarks. Off keeps Python tensor parity.
    /// When set, uses per-frame auto policy (dark/backlight/overexp/noise/cast).
    #[arg(long, default_value_t = false)]
    enhance: bool,
    /// Frames per lighting scene in `--suite enhance` (probes still use 8).
    #[arg(long, default_value_t = 50)]
    frames: usize,
}

impl Args {
    fn enhance_cfg(&self) -> EnhanceCfg {
        if self.enhance {
            EnhanceCfg::auto()
        } else {
            EnhanceCfg::off()
        }
    }
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
    if args.suite == "scale" {
        let path = args.out.clone().unwrap_or_else(default_scale_out);
        run_scale_suite(&args, spec, device, &path)?;
        return Ok(());
    }
    if args.suite == "enhance" {
        let path = args.out.clone().unwrap_or_else(default_enhance_out);
        run_enhance_suite(&args, spec, device, &path)?;
        return Ok(());
    }
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
    let enhanced;
    let frame = if args.enhance {
        enhanced = enhance_bgr(frame, &args.enhance_cfg());
        &enhanced
    } else {
        frame
    };
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
    let mut tracker = GpuTracker::with_enhance(
        &args.models_dir,
        spec,
        args.threads,
        frame,
        args.enhance_cfg(),
    )?;
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

#[derive(Clone)]
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

struct LmBank {
    hi: OrtModel,
    hi_spec: LmSpec,
    lo: Option<OrtModel>,
    lo_spec: LmSpec,
}

impl LmBank {
    fn pair(&mut self, model_type: i32) -> (&mut OrtModel, LmSpec) {
        if model_type == self.lo_spec.model_type {
            if let Some(lo) = self.lo.as_mut() {
                return (lo, self.lo_spec);
            }
        }
        (&mut self.hi, self.hi_spec)
    }
}

struct CpuPipe {
    det: OrtModel,
    lm: LmBank,
}

fn one_frame(
    mut cpu: Option<&mut CpuPipe>,
    mut gpu: Option<&mut GpuTracker>,
    gaze: &mut Option<OrtModel>,
    frame: &BgrImage,
    spec: LmSpec,
    mut box5: Option<[f32; 5]>,
    pad_x: f32,
    pad_y: f32,
    do_detect: bool,
    do_gaze: bool,
    adaptive: Option<(&AdaptiveCfg, &mut AdaptiveState)>,
    enhance: EnhanceCfg,
) -> Result<Row> {
    let enhanced;
    let frame = if gpu.is_none() && !enhance.is_off() {
        enhanced = enhance_bgr(frame, &enhance);
        &enhanced
    } else {
        frame
    };
    let t_all = Instant::now();
    let mut detect_ms = 0.0;
    let mut det_score = None;
    if do_detect {
        let t = Instant::now();
        let last = box5.as_ref();
        let mut window = if let Some((cfg, _)) = adaptive.as_ref() {
            det_window(frame.width, frame.height, last, cfg)
        } else {
            DetWindow::Full
        };
        let mut dets = detect_window(cpu.as_deref_mut(), gpu.as_deref_mut(), frame, window)?;
        if dets.is_empty() && adaptive.is_some() && last.is_none() {
            window = center_2x(frame.width, frame.height);
            dets = detect_window(cpu.as_deref_mut(), gpu.as_deref_mut(), frame, window)?;
        }
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
    let face_h = d[3];
    let lm_type = if let Some((cfg, st)) = adaptive {
        if gpu.is_some() {
            spec.model_type
        } else {
            pick_lm(st, face_h, cfg)
        }
    } else {
        spec.model_type
    };
    let t = Instant::now();
    let (x1, y1, x2, y2) = crop_box_pad(frame, &d, pad_x, pad_y);
    let (crop_w, crop_h) = ((x2 - x1).max(0) as u32, (y2 - y1).max(0) as u32);
    if crop_w < 4 || crop_h < 4 {
        return Ok(miss());
    }
    let crop_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let (conf, pts, pre_ms, lm_ms, decode_ms) = if let Some(pipe) = cpu.as_mut() {
        let (lm, run_spec) = pipe.lm.pair(lm_type);
        match run_landmarks(lm, frame, &d, run_spec, pad_x, pad_y) {
            Ok(v) => v,
            Err(_) => return Ok(miss()),
        }
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

fn detect_window(
    cpu: Option<&mut CpuPipe>,
    gpu: Option<&mut GpuTracker>,
    frame: &BgrImage,
    window: DetWindow,
) -> Result<Vec<[f32; 5]>> {
    let (view, fallback_full) = match window {
        DetWindow::Full => (None, false),
        DetWindow::Roi { x1, y1, x2, y2 } => {
            let crop = crop_img(frame, x1, y1, x2, y2);
            if crop.width < 8 || crop.height < 8 {
                (None, true)
            } else {
                (Some(crop), false)
            }
        }
    };
    if fallback_full {
        return detect_window(cpu, gpu, frame, DetWindow::Full);
    }
    if let Some(pipe) = cpu {
        let src = view.as_ref().unwrap_or(frame);
        let dout = pipe.det.run(&imagenet_nchw(src, 224))?;
        let mut dets = detect_faces(&dout[0], &dout[1], src.width, src.height, 0.6);
        if view.is_some() {
            window.apply_offset(&mut dets);
        }
        return Ok(dets);
    }
    let Some(tr) = gpu else {
        return Ok(Vec::new());
    };
    // CoreML/CUDA sessions are bound to the full-frame size. Stretch the ROI
    // to that size so the fused graph still runs, then map boxes back.
    if let Some(crop) = view.as_ref() {
        let stretched = resize_bgr(crop, frame.width, frame.height);
        match tr.detect(&stretched) {
            Ok(mut dets) => {
                remap_stretched_roi(&mut dets, window, frame);
                return Ok(dets);
            }
            Err(_) => return tr.detect(frame),
        }
    }
    tr.detect(frame)
}

fn remap_stretched_roi(dets: &mut [[f32; 5]], window: DetWindow, frame: &BgrImage) {
    let DetWindow::Roi { x1, y1, x2, y2 } = window else {
        return;
    };
    let rw = (x2 - x1).max(1) as f32;
    let rh = (y2 - y1).max(1) as f32;
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    for d in dets {
        d[0] = x1 as f32 + d[0] * rw / fw;
        d[1] = y1 as f32 + d[1] * rh / fh;
        d[2] *= rw / fw;
        d[3] *= rh / fh;
    }
}

fn run_landmarks(
    lm: &mut OrtModel,
    frame: &BgrImage,
    d: &[f32; 5],
    spec: LmSpec,
    pad_x: f32,
    pad_y: f32,
) -> Result<(f32, Vec<[f32; 3]>, f64, f64, f64)> {
    let t = Instant::now();
    let (x1, y1, x2, y2) = crop_box_pad(frame, d, pad_x, pad_y);
    if x2 - x1 < 4 || y2 - y1 < 4 {
        anyhow::bail!("crop too small");
    }
    let crop = crop_img(frame, x1, y1, x2, y2);
    let lin = imagenet_nchw(&crop, spec.size);
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
    Ok((
        decoded.0,
        decoded.1,
        pre_ms,
        lm_ms,
        t.elapsed().as_secs_f64() * 1000.0,
    ))
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
        let det = OrtModel::open(
            model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
            args.threads,
            device,
            1,
        )?;
        let hi = OrtModel::open(
            model_path(&args.models_dir, spec.file),
            args.threads,
            device,
            1,
        )?;
        let lo_spec = LmSpec::from_type(FAST_LM)?;
        let lo = (args.adaptive && spec.model_type >= 0)
            .then(|| {
                OrtModel::open(
                    model_path(&args.models_dir, lo_spec.file),
                    args.threads,
                    device,
                    1,
                )
            })
            .transpose()?;
        Some(CpuPipe {
            det,
            lm: LmBank {
                hi,
                hi_spec: spec,
                lo,
                lo_spec,
            },
        })
    } else {
        None
    };
    let cfg = args
        .adaptive
        .then(|| AdaptiveCfg::default().with_ceiling(spec.model_type));
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
            .then(|| {
                GpuTracker::with_enhance(
                    &args.models_dir,
                    spec,
                    args.threads,
                    &frames[0],
                    args.enhance_cfg(),
                )
            })
            .transpose()?;
        let enh = args.enhance_cfg();
        let mut state = cfg.map(AdaptiveState::new);
        let mut step = |box5: Option<[f32; 5]>,
                        frame: &BgrImage,
                        scanned: bool,
                        st: Option<&mut AdaptiveState>| {
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
                match (cfg.as_ref(), st) {
                    (Some(c), Some(s)) => Some((c, s)),
                    _ => None,
                },
                enh,
            )
        };
        for _ in 0..args.warmup.max(1) {
            let mut warm = state;
            let _ = step(None, &frames[0], true, warm.as_mut())?;
        }
        let mut box5 = None;
        let mut rows = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let scanned = box5.is_none() || (i as u32 % scan_every == 0);
            let row = step(box5, frame, scanned, state.as_mut())?;
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

fn default_scale_out() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/out/scale_sweep.json")
}

fn default_enhance_out() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/out/enhance_sweep.json")
}

const SCALE_FRACS: [f32; 7] = [0.10, 0.14, 0.20, 0.28, 0.36, 0.48, 0.60];
const SCALE_W: u32 = 1280;
const SCALE_H: u32 = 720;
const SCALE_FRAMES: usize = 8;
const ENHANCE_PROBE_FRAMES: usize = 8;

#[derive(Clone, Serialize)]
struct ScaleScore {
    nme: f32,
    recall: f32,
    e2e_p50_ms: f64,
    far_nme: f32,
    hits: usize,
    frames: usize,
}

#[derive(Clone, Serialize)]
struct FracRow {
    face_frac: f32,
    nme: Option<f32>,
    recall: f32,
    e2e_p50_ms: f64,
}

#[derive(Serialize)]
struct ScaleReport {
    baseline: ScaleScore,
    adaptive: ScaleScore,
    baseline_per_frac: Vec<FracRow>,
    adaptive_per_frac: Vec<FracRow>,
}

#[derive(Clone)]
struct ScaleSeq {
    face_frac: f32,
    frames: Vec<BgrImage>,
}

fn open_cpu_pipe(args: &Args, spec: LmSpec, with_fast: bool) -> Result<CpuPipe> {
    let det = OrtModel::open(
        model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
        args.threads,
        Device::Cpu,
        1,
    )?;
    let hi = OrtModel::open(
        model_path(&args.models_dir, spec.file),
        args.threads,
        Device::Cpu,
        1,
    )?;
    let lo_spec = LmSpec::from_type(FAST_LM)?;
    let lo = with_fast
        .then(|| {
            OrtModel::open(
                model_path(&args.models_dir, lo_spec.file),
                args.threads,
                Device::Cpu,
                1,
            )
        })
        .transpose()?;
    Ok(CpuPipe {
        det,
        lm: LmBank {
            hi,
            hi_spec: spec,
            lo,
            lo_spec,
        },
    })
}

fn extract_face_tile(pipe: &mut CpuPipe, frame: &BgrImage) -> Result<(BgrImage, f32)> {
    let dout = pipe.det.run(&imagenet_nchw(frame, 224))?;
    let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
    let d = dets
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("no face in seed image"))?;
    let (tile, _, face_h) = face_crop(frame, &d, 0.25);
    Ok((tile, face_h))
}

fn make_scale_seq(face: &BgrImage, face_h: f32, frac: f32) -> ScaleSeq {
    make_scale_seq_n(face, face_h, frac, SCALE_FRAMES, false)
}

fn make_scale_seq_n(face: &BgrImage, face_h: f32, frac: f32, n: usize, wander: bool) -> ScaleSeq {
    let n = n.max(1);
    let target = ((SCALE_H as f32 * frac) as u32).max(8);
    let scale = target as f32 / face_h.max(1.0);
    let fw = ((face.width as f32 * scale) as u32).max(8);
    let fh = ((face.height as f32 * scale) as u32).max(8);
    let resized = resize_bgr(face, fw, fh);
    let mut frames = Vec::with_capacity(n);
    for t in 0..n {
        let tf = t as f32;
        let mut jx = (0.035 * fw as f32).max(4.0) * (tf * 0.7).sin();
        let mut jy = (0.025 * fh as f32).max(3.0) * (tf * 0.5).cos();
        if wander {
            jx += (0.04 * fw as f32) * (tf * 0.13).sin();
            jy += (0.03 * fh as f32) * (tf * 0.19).cos();
        }
        let x = (SCALE_W as i32 - fw as i32) / 2 + jx as i32;
        let y = (SCALE_H as i32 - fh as i32) / 2 + jy as i32;
        let mut canvas = synth_canvas(SCALE_W, SCALE_H);
        paste_bgr(&mut canvas, &resized, x, y);
        frames.push(canvas);
    }
    ScaleSeq {
        face_frac: frac,
        frames,
    }
}

struct TeachFrame {
    pts: Vec<[f32; 3]>,
    hit: bool,
}

fn run_seq(
    mut cpu: Option<&mut CpuPipe>,
    mut gpu: Option<&mut GpuTracker>,
    spec: LmSpec,
    seq: &ScaleSeq,
    cfg: Option<&AdaptiveCfg>,
    enhance: EnhanceCfg,
) -> Result<Vec<Row>> {
    let mut state = cfg.copied().map(AdaptiveState::new);
    let mut box5 = None;
    let mut rows = Vec::with_capacity(seq.frames.len());
    let mut gaze = None;
    for frame in &seq.frames {
        let ad = match (cfg, state.as_mut()) {
            (Some(c), Some(s)) => Some((c, s)),
            _ => None,
        };
        let row = one_frame(
            cpu.as_deref_mut(),
            gpu.as_deref_mut(),
            &mut gaze,
            frame,
            spec,
            box5,
            0.1,
            0.125,
            true,
            false,
            ad,
            enhance,
        )?;
        box5 = row.box5;
        rows.push(row);
    }
    Ok(rows)
}

fn mean_f32(v: &[f32]) -> Option<f32> {
    (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
}

fn score_against_teacher(
    seqs: &[ScaleSeq],
    rows: &[Vec<Row>],
    teacher: &[Vec<TeachFrame>],
) -> (ScaleScore, Vec<FracRow>) {
    let mut nmes = Vec::new();
    let mut far_nmes = Vec::new();
    let mut e2e = Vec::new();
    let mut hits = 0usize;
    let mut frames = 0usize;
    let mut per = Vec::new();
    for (seq, (rs, ts)) in seqs.iter().zip(rows.iter().zip(teacher.iter())) {
        let mut local_nme = Vec::new();
        let mut local_e2e = Vec::new();
        let mut local_hits = 0usize;
        for (r, t) in rs.iter().zip(ts.iter()) {
            frames += 1;
            e2e.push(r.e2e_ms);
            local_e2e.push(r.e2e_ms);
            if r.faces > 0 {
                hits += 1;
                local_hits += 1;
            }
            if r.faces > 0 && t.hit {
                if let Some(v) = nme(&r.pts, &t.pts) {
                    local_nme.push(v);
                    nmes.push(v);
                    if seq.face_frac <= 0.14 {
                        far_nmes.push(v);
                    }
                }
            }
        }
        per.push(FracRow {
            face_frac: seq.face_frac,
            nme: mean_f32(&local_nme),
            recall: local_hits as f32 / seq.frames.len().max(1) as f32,
            e2e_p50_ms: Latency::from_samples(0, &local_e2e).p50_ms,
        });
    }
    (
        ScaleScore {
            nme: mean_f32(&nmes).unwrap_or(0.0),
            recall: hits as f32 / frames.max(1) as f32,
            e2e_p50_ms: Latency::from_samples(0, &e2e).p50_ms,
            far_nme: mean_f32(&far_nmes).unwrap_or(0.0),
            hits,
            frames,
        },
        per,
    )
}

fn rows_to_teacher(rows: Vec<Row>) -> Vec<TeachFrame> {
    rows.into_iter()
        .map(|r| TeachFrame {
            hit: r.faces > 0,
            pts: r.pts,
        })
        .collect()
}

fn print_score(tag: &str, s: &ScaleScore) {
    eprintln!(
        "{tag}: rec={:.3} nme={:.4} p50={:.2}ms far_nme={:.4}",
        s.recall, s.nme, s.e2e_p50_ms, s.far_nme
    );
}

fn run_scale_suite(args: &Args, _spec: LmSpec, device: Device, out: &Path) -> Result<()> {
    let spec = LmSpec::from_type(3)?;
    let cfg = AdaptiveCfg::default().with_ceiling(spec.model_type);
    let mut pipe = open_cpu_pipe(args, spec, device == Device::Cpu)?;
    let seed = BgrImage::load(&args.image)?;
    let (tile, face_h) = extract_face_tile(&mut pipe, &seed)?;
    let seqs: Vec<ScaleSeq> = SCALE_FRACS
        .iter()
        .map(|&f| make_scale_seq(&tile, face_h, f))
        .collect();
    eprintln!(
        "scale: seed face_h={face_h:.1}px, {} fracs × {SCALE_FRAMES} frames ({})",
        seqs.len(),
        device.as_str()
    );

    let enh = args.enhance_cfg();
    let (base_rows, ad_rows) = if device == Device::Gpu {
        let mut gpu = GpuTracker::with_enhance(
            &args.models_dir,
            spec,
            args.threads,
            &seqs[0].frames[0],
            enh,
        )?;
        for _ in 0..args.warmup.max(2) {
            let _ = run_seq(None, Some(&mut gpu), spec, &seqs[0], None, enh)?;
        }
        let mut base = Vec::new();
        let mut ad = Vec::new();
        for seq in &seqs {
            base.push(run_seq(None, Some(&mut gpu), spec, seq, None, enh)?);
            ad.push(run_seq(None, Some(&mut gpu), spec, seq, Some(&cfg), enh)?);
        }
        (base, ad)
    } else {
        let mut base = Vec::new();
        let mut ad = Vec::new();
        for seq in &seqs {
            base.push(run_seq(Some(&mut pipe), None, spec, seq, None, enh)?);
            ad.push(run_seq(Some(&mut pipe), None, spec, seq, Some(&cfg), enh)?);
        }
        (base, ad)
    };

    let teacher: Vec<Vec<TeachFrame>> = base_rows.iter().cloned().map(rows_to_teacher).collect();
    let (baseline, baseline_per_frac) = score_against_teacher(&seqs, &base_rows, &teacher);
    let (adaptive, adaptive_per_frac) = score_against_teacher(&seqs, &ad_rows, &teacher);
    print_score("baseline", &baseline);
    print_score("adaptive", &adaptive);
    eprintln!(
        "Δ rec={:+.3}  nme={:.4}  p50={:+.2}ms",
        adaptive.recall - baseline.recall,
        adaptive.nme,
        adaptive.e2e_p50_ms - baseline.e2e_p50_ms
    );
    for (b, a) in baseline_per_frac.iter().zip(adaptive_per_frac.iter()) {
        eprintln!(
            "  frac={:.2} rec {:.3}->{:.3} p50 {:.2}->{:.2}",
            b.face_frac, b.recall, a.recall, b.e2e_p50_ms, a.e2e_p50_ms
        );
    }
    let report = ScaleReport {
        baseline,
        adaptive,
        baseline_per_frac,
        adaptive_per_frac,
    };
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn degrade_gamma(src: &BgrImage, exposure: f32, gamma: f32) -> BgrImage {
    let mut d = src.clone();
    for p in &mut d.data {
        let x = (*p as f32 / 255.0 * exposure).clamp(0.0, 1.0);
        *p = (x.powf(gamma) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    d
}

fn degrade_contrast(src: &BgrImage, k: f32) -> BgrImage {
    let mut d = src.clone();
    for p in &mut d.data {
        *p = ((*p as f32 - 128.0) * k + 128.0).round().clamp(0.0, 255.0) as u8;
    }
    d
}

fn degrade_cast(src: &BgrImage, gb: f32, gg: f32, gr: f32) -> BgrImage {
    let mut d = src.clone();
    for p in d.data.chunks_exact_mut(3) {
        p[0] = ((p[0] as f32) * gb).round().clamp(0.0, 255.0) as u8;
        p[1] = ((p[1] as f32) * gg).round().clamp(0.0, 255.0) as u8;
        p[2] = ((p[2] as f32) * gr).round().clamp(0.0, 255.0) as u8;
    }
    d
}

fn degrade_backlight(src: &BgrImage, face_exp: f32, bg_gain: f32) -> BgrImage {
    let mut d = src.clone();
    let w = d.width as f32;
    let h = d.height as f32;
    let cx = w * 0.5;
    let cy = h * 0.5;
    let sig = 0.16 * w.min(h);
    let inv = 1.0 / (2.0 * sig * sig);
    for y in 0..d.height {
        for x in 0..d.width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let face = (-(dx * dx + dy * dy) * inv).exp();
            let g = face * face_exp + (1.0 - face) * bg_gain;
            let i = ((y * d.width + x) * 3) as usize;
            for c in 0..3 {
                d.data[i + c] = ((d.data[i + c] as f32) * g).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    d
}

fn degrade_noise(src: &BgrImage, sigma: f32, seed: u32) -> BgrImage {
    let mut d = src.clone();
    let mut s = seed | 1;
    for p in &mut d.data {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = (s as f32 / 4_294_967_296.0).clamp(1e-6, 1.0);
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let v = s as f32 / 4_294_967_296.0;
        let n = (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos() * sigma;
        *p = (*p as f32 + n).round().clamp(0.0, 255.0) as u8;
    }
    d
}

fn map_seq(seq: &ScaleSeq, f: impl Fn(&BgrImage) -> BgrImage) -> ScaleSeq {
    ScaleSeq {
        face_frac: seq.face_frac,
        frames: seq.frames.iter().map(f).collect(),
    }
}

fn map_seq_i(seq: &ScaleSeq, f: impl Fn(usize, &BgrImage) -> BgrImage) -> ScaleSeq {
    ScaleSeq {
        face_frac: seq.face_frac,
        frames: seq
            .frames
            .iter()
            .enumerate()
            .map(|(i, fr)| f(i, fr))
            .collect(),
    }
}

#[derive(Clone, Serialize)]
struct SceneRow {
    nme: f32,
    recall: f32,
}

#[derive(Clone, Serialize)]
struct EnhanceTrial {
    name: String,
    cfg: EnhanceCfg,
    scenes: HashMap<String, SceneRow>,
    e2e_p50_ms: f64,
    score: f32,
}

#[derive(Serialize)]
struct SceneProbe {
    name: String,
    detail: String,
    off_recall: f32,
}

#[derive(Serialize)]
struct EnhanceReport {
    face_frac: f32,
    frames: usize,
    dark_exposure: f32,
    dark_gamma: f32,
    probes: Vec<SceneProbe>,
    winner: EnhanceCfg,
    winner_name: String,
    baseline: EnhanceTrial,
    trials: Vec<EnhanceTrial>,
}

fn trial_score(t: &EnhanceTrial, base: &EnhanceTrial) -> f32 {
    let rec = |n: &str| t.scenes.get(n).map(|s| s.recall).unwrap_or(0.0);
    let nme = |n: &str| t.scenes.get(n).map(|s| s.nme).unwrap_or(1.0);
    let brec = |n: &str| base.scenes.get(n).map(|s| s.recall).unwrap_or(0.0);
    if rec("clean") + 0.01 < brec("clean") {
        return -1.0e9;
    }
    let hard = ["dark", "over", "back", "noise", "lowcon"];
    let rec_gain: f32 = hard.iter().map(|n| rec(n) - brec(n)).sum();
    rec_gain * 12.0 + hard.iter().map(|n| rec(n)).sum::<f32>() * 4.0 + rec("clean")
        - [
            "clean", "dark", "over", "back", "noise", "lowcon", "warm", "cool",
        ]
        .iter()
        .map(|n| nme(n))
        .sum::<f32>()
}

fn score_seq(
    cpu: Option<&mut CpuPipe>,
    gpu: Option<&mut GpuTracker>,
    spec: LmSpec,
    seq: &ScaleSeq,
    teacher: &[Vec<TeachFrame>],
    enhance: EnhanceCfg,
) -> Result<(f32, f32, f64)> {
    let rows = run_seq(cpu, gpu, spec, seq, None, enhance)?;
    let (s, _) = score_against_teacher(
        &[ScaleSeq {
            face_frac: seq.face_frac,
            frames: seq.frames.clone(),
        }],
        &[rows],
        teacher,
    );
    Ok((s.nme, s.recall, s.e2e_p50_ms))
}

fn eval_cfg(
    args: &Args,
    spec: LmSpec,
    device: Device,
    pipe: &mut CpuPipe,
    seqs: &[(&'static str, ScaleSeq)],
    teacher: &[Vec<TeachFrame>],
    name: &str,
    cfg: EnhanceCfg,
) -> Result<EnhanceTrial> {
    let mut scenes = HashMap::new();
    let mut e2e_p50_ms = 0.0;
    if device == Device::Gpu {
        let mut gpu = GpuTracker::with_enhance(
            &args.models_dir,
            spec,
            args.threads,
            &seqs[0].1.frames[0],
            cfg,
        )?;
        for (i, (n, seq)) in seqs.iter().enumerate() {
            let (nme, recall, e2e) = score_seq(None, Some(&mut gpu), spec, seq, teacher, cfg)?;
            if i == 0 {
                e2e_p50_ms = e2e;
            }
            scenes.insert((*n).to_string(), SceneRow { nme, recall });
        }
    } else {
        for (i, (n, seq)) in seqs.iter().enumerate() {
            let (nme, recall, e2e) = score_seq(Some(pipe), None, spec, seq, teacher, cfg)?;
            if i == 0 {
                e2e_p50_ms = e2e;
            }
            scenes.insert((*n).to_string(), SceneRow { nme, recall });
        }
    }
    Ok(EnhanceTrial {
        name: name.to_string(),
        cfg,
        scenes,
        e2e_p50_ms,
        score: 0.0,
    })
}

fn seq_recall(
    pipe: &mut CpuPipe,
    spec: LmSpec,
    seq: &ScaleSeq,
    enhance: EnhanceCfg,
) -> Result<f32> {
    let rows = run_seq(Some(pipe), None, spec, seq, None, enhance)?;
    let hits = rows.iter().filter(|r| r.faces > 0).count();
    Ok(hits as f32 / rows.len().max(1) as f32)
}

#[derive(Clone, Copy)]
struct DarkLight {
    frac: f32,
    exposure: f32,
    gamma: f32,
    off_recall: f32,
}

fn pick_dark_lighting(
    pipe: &mut CpuPipe,
    spec: LmSpec,
    face: &BgrImage,
    face_h: f32,
    probe_n: usize,
) -> Result<DarkLight> {
    const GAMMA: f32 = 1.7;
    let mut fallback: Option<DarkLight> = None;
    for frac in [0.20f32, 0.14, 0.10] {
        let clean = make_scale_seq_n(face, face_h, frac, probe_n, false);
        for exp in [0.40f32, 0.32, 0.26, 0.22, 0.16, 0.12, 0.09, 0.06] {
            let dark = map_seq(&clean, |f| degrade_gamma(f, exp, GAMMA));
            let rec = seq_recall(pipe, spec, &dark, EnhanceCfg::off())?;
            eprintln!("  probe frac={frac:.2} exp={exp:.2} γ={GAMMA} off_rec={rec:.3}");
            let lit = DarkLight {
                frac,
                exposure: exp,
                gamma: GAMMA,
                off_recall: rec,
            };
            let worse = match &fallback {
                None => true,
                Some(b) => rec < b.off_recall - 1e-6,
            };
            if worse {
                fallback = Some(lit);
            }
            if rec < 0.5 {
                return Ok(lit);
            }
        }
    }
    fallback.ok_or_else(|| anyhow::anyhow!("no dark probe frames"))
}

fn probe_param<T: Copy>(
    pipe: &mut CpuPipe,
    spec: LmSpec,
    clean: &ScaleSeq,
    name: &'static str,
    levels: &[T],
    apply: impl Fn(&BgrImage, T) -> BgrImage,
    label: impl Fn(T) -> String,
) -> Result<(T, SceneProbe)> {
    let mut fallback: Option<(T, SceneProbe)> = None;
    for &lv in levels {
        let seq = map_seq(clean, |f| apply(f, lv));
        let rec = seq_recall(pipe, spec, &seq, EnhanceCfg::off())?;
        let detail = label(lv);
        eprintln!("  probe {name} {detail} off_rec={rec:.3}");
        let probe = SceneProbe {
            name: name.to_string(),
            detail,
            off_recall: rec,
        };
        let worse = match &fallback {
            None => true,
            Some((_, b)) => rec < b.off_recall - 1e-6,
        };
        if worse {
            fallback = Some((lv, probe));
        }
        if rec < 0.5 {
            return Ok(fallback.unwrap());
        }
    }
    fallback.ok_or_else(|| anyhow::anyhow!("no {name} probe frames"))
}

fn fmt_row(t: &EnhanceTrial, name: &str) -> String {
    t.scenes
        .get(name)
        .map(|s| format!("rec={:.3} nme={:.4}", s.recall, s.nme))
        .unwrap_or_else(|| "-".into())
}

fn run_enhance_suite(args: &Args, _spec: LmSpec, device: Device, out: &Path) -> Result<()> {
    let spec = LmSpec::from_type(3)?;
    let mut pipe = open_cpu_pipe(args, spec, false)?;
    let seed = BgrImage::load(&args.image)?;
    let (tile, face_h) = extract_face_tile(&mut pipe, &seed)?;
    let n = args.frames.max(1);
    let probe_n = n.min(ENHANCE_PROBE_FRAMES);
    eprintln!(
        "enhance: seed face_h={face_h:.1}px, {n} frames/scene (probe {probe_n}) ({})",
        device.as_str()
    );
    let lit = pick_dark_lighting(&mut pipe, spec, &tile, face_h, probe_n)?;
    let clean = make_scale_seq_n(&tile, face_h, lit.frac, n, true);
    let probe_clean = ScaleSeq {
        face_frac: clean.face_frac,
        frames: clean.frames[..probe_n.min(clean.frames.len())].to_vec(),
    };
    eprintln!(
        "enhance: frac={:.2} dark exp={:.2} γ={:.1} (probe off rec={:.3})",
        lit.frac, lit.exposure, lit.gamma, lit.off_recall
    );

    let (over_g, over_p) = probe_param(
        &mut pipe,
        spec,
        &probe_clean,
        "over",
        &[1.35f32, 1.55, 1.75, 2.1, 2.6, 3.4],
        |f, g| degrade_gamma(f, g, 0.85),
        |g| format!("gain={g:.2} γ=0.85"),
    )?;
    let (back_lv, back_p) = probe_param(
        &mut pipe,
        spec,
        &probe_clean,
        "back",
        &[
            (0.55f32, 2.2f32),
            (0.45, 2.8),
            (0.36, 3.6),
            (0.28, 4.6),
            (0.20, 5.8),
        ],
        |f, (fe, bg)| degrade_backlight(f, fe, bg),
        |(fe, bg)| format!("face={fe:.2} bg={bg:.1}"),
    )?;
    let (noise_s, noise_p) = probe_param(
        &mut pipe,
        spec,
        &probe_clean,
        "noise",
        &[16.0f32, 24.0, 34.0, 46.0, 60.0, 78.0],
        |f, s| degrade_noise(f, s, 0xC0FFEE),
        |s| format!("σ={s:.0}"),
    )?;

    let teacher_rows = run_seq(Some(&mut pipe), None, spec, &clean, None, EnhanceCfg::off())?;
    let teacher = vec![rows_to_teacher(teacher_rows)];
    let seqs: Vec<(&'static str, ScaleSeq)> = vec![
        ("clean", clean.clone()),
        (
            "dark",
            map_seq(&clean, |f| degrade_gamma(f, lit.exposure, lit.gamma)),
        ),
        ("over", map_seq(&clean, |f| degrade_gamma(f, over_g, 0.85))),
        (
            "back",
            map_seq(&clean, |f| degrade_backlight(f, back_lv.0, back_lv.1)),
        ),
        (
            "noise",
            map_seq_i(&clean, |i, f| {
                degrade_noise(f, noise_s, 0xC0FFEE ^ (i as u32).wrapping_mul(0x9E3779B9))
            }),
        ),
        (
            "lowcon",
            map_seq(&clean, |f| {
                degrade_gamma(&degrade_contrast(f, 0.35), 0.45, 1.4)
            }),
        ),
        (
            "warm",
            map_seq(&clean, |f| degrade_cast(f, 0.55, 0.75, 1.15)),
        ),
        (
            "cool",
            map_seq(&clean, |f| degrade_cast(f, 1.15, 0.75, 0.55)),
        ),
    ];

    let policies: Vec<(&str, EnhanceCfg)> = vec![
        ("off", EnhanceCfg::off()),
        ("clahe", EnhanceCfg::clahe()),
        ("auto", EnhanceCfg::auto()),
    ];

    let mut trials = Vec::new();
    for (i, (name, cfg)) in policies.iter().enumerate() {
        eprintln!(
            "  [{}/{}] {name} he={} auto={}",
            i + 1,
            policies.len(),
            cfg.he.as_str(),
            cfg.auto
        );
        match eval_cfg(args, spec, device, &mut pipe, &seqs, &teacher, name, *cfg) {
            Ok(t) => {
                eprintln!(
                    "      clean {}  dark {}  over {}  back {}  noise {}  lowcon {}",
                    fmt_row(&t, "clean"),
                    fmt_row(&t, "dark"),
                    fmt_row(&t, "over"),
                    fmt_row(&t, "back"),
                    fmt_row(&t, "noise"),
                    fmt_row(&t, "lowcon")
                );
                trials.push(t);
            }
            Err(e) => eprintln!("      skip: {e}"),
        }
    }
    let baseline = trials
        .iter()
        .find(|t| t.name == "off")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing off trial"))?;
    for t in &mut trials {
        t.score = trial_score(t, &baseline);
    }
    let winner = trials
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .cloned()
        .unwrap();
    eprintln!(
        "winner: {} he={} auto={} score={:.3}",
        winner.name,
        winner.cfg.he.as_str(),
        winner.cfg.auto,
        winner.score
    );
    let rec_of = |t: &EnhanceTrial, n: &str| t.scenes.get(n).map(|s| s.recall).unwrap_or(0.0);
    eprintln!(
        "  vs off: dark {:.2}->{:.2}  over {:.2}->{:.2}  back {:.2}->{:.2}  noise {:.2}->{:.2}",
        rec_of(&baseline, "dark"),
        rec_of(&winner, "dark"),
        rec_of(&baseline, "over"),
        rec_of(&winner, "over"),
        rec_of(&baseline, "back"),
        rec_of(&winner, "back"),
        rec_of(&baseline, "noise"),
        rec_of(&winner, "noise")
    );
    let mut probes = vec![SceneProbe {
        name: "dark".into(),
        detail: format!("exp={:.2} γ={:.1}", lit.exposure, lit.gamma),
        off_recall: lit.off_recall,
    }];
    probes.push(over_p);
    probes.push(back_p);
    probes.push(noise_p);
    let report = EnhanceReport {
        face_frac: lit.frac,
        frames: n,
        dark_exposure: lit.exposure,
        dark_gamma: lit.gamma,
        probes,
        winner: winner.cfg,
        winner_name: winner.name.clone(),
        baseline,
        trials,
    };
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
