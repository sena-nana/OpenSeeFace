use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use osf_ort::{
    cosine, crop_box, crop_box_pad, crop_img, decode_landmarks, decode_landmarks_data,
    detect_faces, face_crop, imagenet_nchw, imagenet_nchw_roi_into, iou, max_abs, mean_abs,
    model_path, paste_bgr, read_f32_le, resize_bgr, retina_nchw, rss, set_simd_mode, simd_backend,
    synth_canvas, unwrap_deg, BgrImage, FilterCfg, FilterKind, Latency, LmSpec, OrtModel,
    OutputFilter, SimdMode, TensorF16, Tracker, TrackerConfig, VERSION,
};
use serde::{Deserialize, Serialize};

const SCALE_W: u32 = 1280;
const SCALE_H: u32 = 720;

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
    /// CPU preprocess: auto | on | off. auto = SIMD on x86, scalar on Apple Silicon.
    #[arg(long, default_value_t = SimdMode::Auto)]
    simd: SimdMode,
    /// micro | realistic | all | filter | pre
    #[arg(long, default_value = "micro")]
    suite: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    ref_dir: Option<PathBuf>,
    #[arg(long)]
    scenario_dir: Option<PathBuf>,
    /// Frames per scene in `--suite filter`.
    #[arg(long, default_value_t = 50)]
    frames: usize,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    crate_version: &'static str,
    ort_dylib: Option<String>,
    threads: usize,
    models: HashMap<String, ModelReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline: Option<Pipeline>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    scenarios: HashMap<String, ScenarioReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preprocess: Option<PreprocessReport>,
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
struct PreprocessReport {
    simd: String,
    src_w: u32,
    src_h: u32,
    full_224: Latency,
    full_224_scalar: Latency,
    roi_112: Latency,
    roi_112_scalar: Latency,
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
    gaze_conf: Option<f32>,
    landmarks: Vec<[f32; 3]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    set_simd_mode(args.simd);
    let spec = LmSpec::from_type(args.model)?;
    if args.suite == "filter" {
        let path = args.out.clone().unwrap_or_else(default_filter_out);
        run_filter_suite(&args, &path)?;
        return Ok(());
    }
    if args.suite == "pre" {
        let frame = if args.image.is_file() {
            BgrImage::load(&args.image)?
        } else {
            synth_canvas(SCALE_W, SCALE_H)
        };
        let preprocess = bench_preprocess(&frame, args.warmup, args.iters, args.simd);
        eprint_preprocess(&preprocess);
        let report = Report {
            backend: "ort-rust",
            crate_version: VERSION,
            ort_dylib: std::env::var("ORT_DYLIB_PATH").ok(),
            threads: args.threads,
            models: HashMap::new(),
            pipeline: None,
            scenarios: HashMap::new(),
            preprocess: Some(preprocess),
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
        return Ok(());
    }
    let run_micro = args.suite == "micro" || args.suite == "all";
    let run_real = args.suite == "realistic" || args.suite == "all";
    let mut models = HashMap::new();
    let mut pipe = None;
    let mut preprocess = None;

    if run_micro {
        let frame = BgrImage::load(&args.image)?;
        preprocess = Some(bench_preprocess(&frame, args.warmup, args.iters, args.simd));
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
        pipe = Some(pipeline(&args, &frame, spec)?);
    }

    let scenarios = if run_real {
        if let Some(dir) = &args.scenario_dir {
            run_scenario_dir(&args, spec, dir)?
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    if let Some(p) = &preprocess {
        eprint_preprocess(p);
    }
    if let Some(p) = &pipe {
        eprintln!(
            "pipeline  detect={:.3}ms  landmarks={:.3}ms  e2e={:.3}ms  faces={}",
            p.detect_ms, p.landmarks_ms, p.e2e_ms, p.faces
        );
    }
    let report = Report {
        backend: "ort-rust",
        crate_version: VERSION,
        ort_dylib: std::env::var("ORT_DYLIB_PATH").ok(),
        threads: args.threads,
        models,
        pipeline: pipe,
        scenarios,
        preprocess,
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

fn eprint_preprocess(p: &PreprocessReport) {
    eprintln!(
        "preprocess {} vs scalar  {}x{}  224 p50={:.3}/{:.3}ms ({:.2}x)  roi112={:.3}/{:.3}ms ({:.2}x)",
        p.simd,
        p.src_w,
        p.src_h,
        p.full_224.p50_ms,
        p.full_224_scalar.p50_ms,
        p.full_224_scalar.p50_ms / p.full_224.p50_ms.max(1e-9),
        p.roi_112.p50_ms,
        p.roi_112_scalar.p50_ms,
        p.roi_112_scalar.p50_ms / p.roi_112.p50_ms.max(1e-9),
    );
}

fn bench_preprocess(
    frame: &BgrImage,
    warmup: u32,
    iters: u32,
    restore: SimdMode,
) -> PreprocessReport {
    let src = if frame.width == SCALE_W && frame.height == SCALE_H {
        frame.clone()
    } else {
        resize_bgr(frame, SCALE_W, SCALE_H)
    };
    let mut buf224 = vec![half::f16::ZERO; 3 * 224 * 224];
    let mut buf112 = vec![half::f16::ZERO; 3 * 112 * 112];
    let time_ms = |f: &mut dyn FnMut()| -> Vec<f64> {
        for _ in 0..warmup {
            f();
        }
        (0..iters)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect()
    };
    let run224 = |buf: &mut [half::f16]| {
        imagenet_nchw_roi_into(&src, 0, 0, src.width as i32, src.height as i32, 224, buf);
        std::hint::black_box(buf[0]);
    };
    let run112 = |buf: &mut [half::f16]| {
        imagenet_nchw_roi_into(&src, 440, 160, 840, 560, 112, buf);
        std::hint::black_box(buf[0]);
    };
    set_simd_mode(SimdMode::On);
    let s224 = time_ms(&mut || run224(&mut buf224));
    let s112 = time_ms(&mut || run112(&mut buf112));
    let simd = simd_backend().to_string();
    set_simd_mode(SimdMode::Off);
    let s224s = time_ms(&mut || run224(&mut buf224));
    let s112s = time_ms(&mut || run112(&mut buf112));
    set_simd_mode(restore);
    PreprocessReport {
        simd,
        src_w: src.width,
        src_h: src.height,
        full_224: Latency::from_samples(warmup, &s224),
        full_224_scalar: Latency::from_samples(warmup, &s224s),
        roi_112: Latency::from_samples(warmup, &s112),
        roi_112_scalar: Latency::from_samples(warmup, &s112s),
    }
}

fn bench(
    path: &Path,
    filename: &str,
    input: &TensorF16,
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
        .map(read_f32_le)
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

fn pipeline(args: &Args, frame: &BgrImage, spec: LmSpec) -> Result<Pipeline> {
    let mut det = OrtModel::load(
        model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
        args.threads,
    )?;
    let mut lm = OrtModel::load(model_path(&args.models_dir, spec.file), args.threads)?;
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
    let din = imagenet_nchw(frame, 224);
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
                    decode_landmarks(&out[0], [x1 as f32, y1 as f32, scale_x, scale_y], spec).1,
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
    gaze_conf: Option<f32>,
    crop_w: u32,
    crop_h: u32,
    box5: Option<[f32; 5]>,
    pts: Vec<[f32; 3]>,
}

struct Pipe {
    det: OrtModel,
    lm: OrtModel,
}

fn open_pipe(args: &Args, spec: LmSpec) -> Result<Pipe> {
    Ok(Pipe {
        det: OrtModel::load(
            model_path(&args.models_dir, "mnv3_detection_opt.onnx"),
            args.threads,
        )?,
        lm: OrtModel::load(model_path(&args.models_dir, spec.file), args.threads)?,
    })
}

fn landmark_bbox(pts: &[[f32; 3]], conf: f32) -> Option<[f32; 5]> {
    if pts.is_empty() {
        return None;
    }
    let mut min_r = f32::MAX;
    let mut max_r = f32::MIN;
    let mut min_c = f32::MAX;
    let mut max_c = f32::MIN;
    for p in pts.iter().take(66) {
        min_r = min_r.min(p[0]);
        max_r = max_r.max(p[0]);
        min_c = min_c.min(p[1]);
        max_c = max_c.max(p[1]);
    }
    Some([
        min_c,
        min_r,
        (max_c - min_c).max(1.0),
        (max_r - min_r).max(1.0),
        conf,
    ])
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

fn mean_f32(v: &[f32]) -> Option<f32> {
    (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
}

fn nme(pred: &[[f32; 3]], gt: &[[f32; 3]]) -> Option<f32> {
    if pred.len() < 46 || gt.len() < 46 {
        return None;
    }
    let iod = (gt[36][0] - gt[45][0])
        .hypot(gt[36][1] - gt[45][1])
        .max(1.0);
    let n = pred.len().min(gt.len()).min(66);
    let sum: f32 = pred
        .iter()
        .zip(gt)
        .take(n)
        .map(|(p, g)| (p[0] - g[0]).hypot(p[1] - g[1]))
        .sum();
    Some(sum / (n as f32 * iod))
}

#[allow(clippy::too_many_arguments)]
fn one_frame(
    pipe: &mut Pipe,
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
        let dout = pipe.det.run(&imagenet_nchw(frame, 224))?;
        let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
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
    let (conf, pts, pre_ms, lm_ms, decode_ms) =
        match run_landmarks(&mut pipe.lm, frame, &d, spec, pad_x, pad_y) {
            Ok(v) => v,
            Err(_) => return Ok(miss()),
        };
    let next = landmark_bbox(&pts, conf).unwrap_or(d);
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
        gaze_conf: gconf,
        crop_w,
        crop_h,
        box5: Some(next),
        pts,
    })
}

#[allow(clippy::type_complexity)]
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
    let crop = [
        x1 as f32,
        y1 as f32,
        (x2 - x1) as f32 / spec.size as f32,
        (y2 - y1) as f32 / spec.size as f32,
    ];
    let pre_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let size = spec.size;
    let shape = [1i64, 3, size as i64, size as i64];
    let decoded = lm.run_prep(
        &shape,
        |buf| imagenet_nchw_roi_into(frame, x1, y1, x2, y2, size, buf),
        |outs| Ok(decode_landmarks_data(outs[0], crop, spec)),
    )?;
    let lm_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((decoded.0, decoded.1, pre_ms, lm_ms, 0.0))
}

fn lat(warmup: u32, rows: &[Row], f: impl Fn(&Row) -> f64) -> Latency {
    Latency::from_samples(warmup, &rows.iter().map(f).collect::<Vec<_>>())
}

fn run_scenario_dir(
    args: &Args,
    spec: LmSpec,
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
            OrtModel::load(
                model_path(&args.models_dir, "mnv3_gaze32_split_opt.onnx"),
                args.threads,
            )
        })
        .transpose()?;
    let mut pipe = open_pipe(args, spec)?;
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
        for _ in 0..args.warmup.max(1) {
            let _ = one_frame(
                &mut pipe, &mut gaze, &frames[0], spec, None, meta.pad_x, meta.pad_y, true, do_gaze,
            )?;
        }
        let mut box5 = None;
        let mut rows = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let scanned = box5.is_none() || (i as u32).is_multiple_of(scan_every);
            let row = one_frame(
                &mut pipe, &mut gaze, frame, spec, box5, meta.pad_x, meta.pad_y, scanned, do_gaze,
            )?;
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
                gaze_conf: mean_opt(rows.iter().map(|r| r.gaze_conf)),
                landmarks: last.map(|r| r.pts.clone()).unwrap_or_default(),
            },
        );
    }
    Ok(out)
}

fn default_filter_out() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/out/filter.json")
}

fn extract_face_tile(pipe: &mut Pipe, frame: &BgrImage) -> Result<(BgrImage, f32)> {
    let dout = pipe.det.run(&imagenet_nchw(frame, 224))?;
    let dets = detect_faces(&dout[0], &dout[1], frame.width, frame.height, 0.6);
    let d = dets
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("no face in seed image"))?;
    let (tile, _, face_h) = face_crop(frame, &d, 0.25);
    Ok((tile, face_h))
}

fn make_scale_seq_n(
    face: &BgrImage,
    face_h: f32,
    frac: f32,
    n: usize,
    wander: bool,
) -> Vec<BgrImage> {
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
    frames
}

fn make_step_seq(face: &BgrImage, face_h: f32, frac: f32, n: usize, dx: f32) -> Vec<BgrImage> {
    let n = n.max(8);
    let target = ((SCALE_H as f32 * frac) as u32).max(8);
    let scale = target as f32 / face_h.max(1.0);
    let fw = ((face.width as f32 * scale) as u32).max(8);
    let fh = ((face.height as f32 * scale) as u32).max(8);
    let resized = resize_bgr(face, fw, fh);
    let mut frames = Vec::with_capacity(n);
    let mid = n / 2;
    for i in 0..n {
        let extra = if i >= mid { dx } else { 0.0 };
        let x = (SCALE_W as i32 - fw as i32) / 2 + extra as i32;
        let y = (SCALE_H as i32 - fh as i32) / 2;
        let mut canvas = synth_canvas(SCALE_W, SCALE_H);
        paste_bgr(&mut canvas, &resized, x, y);
        frames.push(canvas);
    }
    frames
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

#[derive(Clone)]
struct RawTrack {
    euler: [f32; 3],
    translation: [f32; 3],
    quaternion: [f32; 4],
    lms: Vec<[f32; 3]>,
    pts_3d: [[f32; 3]; 70],
    conf: f32,
    pnp_error: f32,
    hit: bool,
    e2e_ms: f64,
}

#[derive(Serialize)]
struct FilterSceneScore {
    scene: String,
    filter: String,
    jitter_euler: f32,
    jitter_center: f32,
    lag_frames: f32,
    nme_vs_teacher: f32,
    pose_rmse: f32,
    filter_us: f64,
    e2e_p50_ms: f64,
    mean_conf: f32,
    mean_pnp_error: f32,
    hits: usize,
    frames: usize,
}

#[derive(Serialize)]
struct FilterReport {
    frames: usize,
    dt: f32,
    static_jitter_drop: f32,
    noisy_nme_ratio: f32,
    step_lag_extra: f32,
    filter_us: f64,
    scenes: Vec<FilterSceneScore>,
}

fn open_filter_tracker(args: &Args, w: u32, h: u32) -> Result<Tracker> {
    Tracker::new(TrackerConfig {
        width: w,
        height: h,
        model_type: 3,
        threshold: Some(0.1),
        detection_threshold: 0.5,
        max_threads: args.threads.max(1),
        silent: true,
        model_dir: Some(args.models_dir.clone()),
        no_gaze: true,
        max_feature_updates: 900.0,
        static_model: true,
        filter: FilterKind::None,
        discard_after: 60,
        scan_every: 3,
        ..TrackerConfig::default()
    })
}

fn capture_raw(tracker: &mut Tracker, frames: &[BgrImage]) -> Result<Vec<RawTrack>> {
    tracker.set_size(SCALE_W, SCALE_H);
    let mut out = Vec::with_capacity(frames.len());
    for frame in frames {
        let t0 = Instant::now();
        let faces = tracker.predict(frame);
        let e2e_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(f) = faces.first() {
            out.push(RawTrack {
                euler: f.euler,
                translation: f.translation,
                quaternion: f.quaternion,
                lms: f.lms.clone(),
                pts_3d: f.pts_3d,
                conf: f.conf,
                pnp_error: f.pnp_error,
                hit: true,
                e2e_ms,
            });
        } else {
            out.push(RawTrack {
                euler: [0.0; 3],
                translation: [0.0; 3],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                lms: Vec::new(),
                pts_3d: [[0.0; 3]; 70],
                conf: 0.0,
                pnp_error: 0.0,
                hit: false,
                e2e_ms,
            });
        }
    }
    Ok(out)
}

fn replay_filter(raw: &[RawTrack], cfg: FilterCfg, dt: f32) -> (Vec<RawTrack>, f64) {
    let mut flt = OutputFilter::new(cfg);
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let mut s = r.clone();
        if s.hit {
            flt.apply(
                &mut s.euler,
                &mut s.translation,
                &mut s.quaternion,
                &mut s.lms,
                &mut s.pts_3d,
                dt,
            );
        } else {
            flt.reset();
        }
        out.push(s);
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / raw.len().max(1) as f64;
    (out, us)
}

fn raw_mean(raw: &[RawTrack], f: fn(&RawTrack) -> f32) -> f32 {
    mean_f32(&raw.iter().filter(|r| r.hit).map(f).collect::<Vec<_>>()).unwrap_or(0.0)
}

fn lms_center(lms: &[[f32; 3]]) -> [f32; 2] {
    let n = lms.len().min(66);
    if n == 0 {
        return [0.0, 0.0];
    }
    let s = n as f32;
    [
        lms.iter().take(n).map(|p| p[0]).sum::<f32>() / s,
        lms.iter().take(n).map(|p| p[1]).sum::<f32>() / s,
    ]
}

fn euler_delta(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = unwrap_deg(a[0], b[0]) - a[0];
    let d1 = unwrap_deg(a[1], b[1]) - a[1];
    let d2 = unwrap_deg(a[2], b[2]) - a[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

fn jitter_of(rows: &[RawTrack]) -> (f32, f32) {
    let mut eul = Vec::new();
    let mut ctr = Vec::new();
    let mut prev: Option<&RawTrack> = None;
    for r in rows {
        if !r.hit {
            prev = None;
            continue;
        }
        if let Some(p) = prev {
            eul.push(euler_delta(p.euler, r.euler));
            let a = lms_center(&p.lms);
            let b = lms_center(&r.lms);
            ctr.push((a[0] - b[0]).hypot(a[1] - b[1]));
        }
        prev = Some(r);
    }
    (mean_f32(&eul).unwrap_or(0.0), mean_f32(&ctr).unwrap_or(0.0))
}

fn nme_pose(pred: &[RawTrack], teacher: &[RawTrack]) -> (f32, f32) {
    let mut nmes = Vec::new();
    let mut poses = Vec::new();
    for (p, t) in pred.iter().zip(teacher) {
        if p.hit && t.hit && p.lms.len() >= 46 && t.lms.len() >= 46 {
            if let Some(v) = nme(&p.lms, &t.lms) {
                nmes.push(v);
            }
            poses.push(euler_delta(t.euler, p.euler));
        }
    }
    (
        mean_f32(&nmes).unwrap_or(0.0),
        mean_f32(&poses).unwrap_or(0.0),
    )
}

fn lag_frames(raw: &[RawTrack], filt: &[RawTrack]) -> f32 {
    let n = raw.len();
    if n < 8 {
        return 0.0;
    }
    let mid = n / 2;
    let xs: Vec<f32> = raw.iter().map(|r| lms_center(&r.lms)[1]).collect();
    let ys: Vec<f32> = filt.iter().map(|r| lms_center(&r.lms)[1]).collect();
    let pre = median_f32(&xs[..mid]);
    let post = median_f32(&xs[mid..]);
    let span = (post - pre).abs();
    if span < 8.0 {
        return 0.0;
    }
    let target = pre + 0.5 * (post - pre);
    let rising = post > pre;
    let cross = |v: &[f32]| {
        v.iter()
            .enumerate()
            .skip(mid.saturating_sub(1))
            .find(|(_, y)| if rising { **y >= target } else { **y <= target })
            .map(|(i, _)| i as f32)
            .unwrap_or(n as f32)
    };
    (cross(&ys) - cross(&xs)).max(0.0)
}

fn median_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

fn print_filter_row(s: &FilterSceneScore) {
    eprintln!(
        "  {:8} {:10} jitter_e={:.3} jitter_c={:.3} lag={:.1} nme={:.4} pose={:.3} filt={:.1}us e2e={:.2}ms rec={:.2} conf={:.3} pnp={:.1}",
        s.scene,
        s.filter,
        s.jitter_euler,
        s.jitter_center,
        s.lag_frames,
        s.nme_vs_teacher,
        s.pose_rmse,
        s.filter_us,
        s.e2e_p50_ms,
        s.hits as f32 / s.frames.max(1) as f32,
        s.mean_conf,
        s.mean_pnp_error
    );
}

fn run_filter_suite(args: &Args, out: &Path) -> Result<()> {
    let mut pipe = open_pipe(args, LmSpec::from_type(3)?)?;
    let seed = BgrImage::load(&args.image)?;
    let (tile, face_h) = extract_face_tile(&mut pipe, &seed)?;
    let n = args.frames.max(30);
    let dt = 1.0 / 30.0;
    eprintln!("filter: seed face_h={face_h:.1}px, {n} frames/scene");

    let mut static_clean = make_scale_seq_n(&tile, face_h, 0.20, n, false);
    if let Some(first) = static_clean.first().cloned() {
        static_clean = vec![first; n];
    }
    let static_seq: Vec<BgrImage> = static_clean
        .iter()
        .enumerate()
        .map(|(i, f)| degrade_noise(f, 16.0, 0x51A71C ^ (i as u32 * 131)))
        .collect();
    let wander = make_scale_seq_n(&tile, face_h, 0.20, n, true);
    let far = make_scale_seq_n(&tile, face_h, 0.08, n, true);
    let noisy: Vec<BgrImage> = wander
        .iter()
        .enumerate()
        .map(|(i, f)| degrade_noise(f, 34.0, 0xC0FFEE ^ (i as u32 * 997)))
        .collect();
    let step = make_step_seq(&tile, face_h, 0.20, n, 160.0);
    let static_teacher_seq = static_clean;
    let scenes: [(&str, Vec<BgrImage>); 5] = [
        ("static", static_seq),
        ("wander", wander),
        ("far", far),
        ("noisy", noisy),
        ("step", step),
    ];

    let kinds: [(&str, FilterCfg); 2] = [
        ("none", FilterCfg::default()),
        ("one-euro", FilterCfg::new(FilterKind::OneEuro, 1.0, 0.007)),
    ];

    let mut scores = Vec::new();
    for (scene_name, seq) in &scenes {
        let mut tracker = open_filter_tracker(args, SCALE_W, SCALE_H)?;
        let raw = capture_raw(&mut tracker, seq)?;
        let mean_c = raw_mean(&raw, |r| r.conf);
        let mean_p = raw_mean(&raw, |r| r.pnp_error);
        let teacher = if *scene_name == "noisy" {
            let mut t = open_filter_tracker(args, SCALE_W, SCALE_H)?;
            let clean = scenes
                .iter()
                .find(|(n, _)| *n == "wander")
                .map(|(_, s)| s)
                .unwrap();
            capture_raw(&mut t, clean)?
        } else if *scene_name == "static" {
            let mut t = open_filter_tracker(args, SCALE_W, SCALE_H)?;
            capture_raw(&mut t, &static_teacher_seq)?
        } else {
            raw.clone()
        };
        let e2e = Latency::from_samples(0, &raw.iter().map(|r| r.e2e_ms).collect::<Vec<_>>());
        for (fname, cfg) in &kinds {
            let (filt, filter_us) = if *fname == "none" {
                (raw.clone(), 0.0)
            } else {
                replay_filter(&raw, *cfg, dt)
            };
            let (je, jc) = jitter_of(&filt);
            let (nme_t, pose) = nme_pose(&filt, &teacher);
            let lag = if *scene_name == "step" {
                lag_frames(&raw, &filt)
            } else {
                0.0
            };
            let sc = FilterSceneScore {
                scene: (*scene_name).into(),
                filter: (*fname).into(),
                jitter_euler: je,
                jitter_center: jc,
                lag_frames: lag,
                nme_vs_teacher: nme_t,
                pose_rmse: pose,
                filter_us,
                e2e_p50_ms: e2e.p50_ms,
                mean_conf: mean_c,
                mean_pnp_error: mean_p,
                hits: filt.iter().filter(|r| r.hit).count(),
                frames: filt.len(),
            };
            print_filter_row(&sc);
            scores.push(sc);
        }
    }

    let pick = |scene: &str, filt: &str, f: fn(&FilterSceneScore) -> f32| -> f32 {
        scores
            .iter()
            .find(|s| s.scene == scene && s.filter == filt)
            .map(f)
            .unwrap_or(0.0)
    };
    let jitter = |s: &FilterSceneScore| s.jitter_euler.max(s.jitter_center);
    let none_j = pick("static", "none", jitter);
    let euro_j = pick("static", "one-euro", jitter);
    let drop = if none_j > 1e-6 {
        (none_j - euro_j) / none_j
    } else {
        0.0
    };
    let none_nme = pick("noisy", "none", |s| s.nme_vs_teacher);
    let euro_nme = pick("noisy", "one-euro", |s| s.nme_vs_teacher);
    let nme_ratio = if none_nme > 1e-6 {
        euro_nme / none_nme
    } else {
        1.0
    };
    let none_lag = pick("step", "none", |s| s.lag_frames);
    let euro_lag = pick("step", "one-euro", |s| s.lag_frames);
    let lag_extra = (euro_lag - none_lag).max(0.0);
    let filter_us = scores
        .iter()
        .filter(|s| s.filter == "one-euro")
        .map(|s| s.filter_us)
        .fold(0.0, f64::max);
    eprintln!(
        "one-euro vs none: drop={:.1}% nme_ratio={nme_ratio:.3} lag_extra={lag_extra:.1} filt={filter_us:.1}us",
        drop * 100.0
    );

    let report = FilterReport {
        frames: n,
        dt,
        static_jitter_drop: drop,
        noisy_nme_ratio: nme_ratio,
        step_lag_extra: lag_extra,
        filter_us,
        scenes: scores,
    };
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
