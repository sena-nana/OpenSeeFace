#!/usr/bin/env python3
"""A/B: Python onnxruntime vs Rust ort on the same OpenSeeFace models."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import imagenet_nchw, retina_nchw  # noqa: E402

BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from fetch_fixtures import cached_photos, fetch_all  # noqa: E402
from scenarios import (  # noqa: E402
    EYE_IDX,
    REALISTIC,
    face_crop,
    generate_scenarios,
    list_scenario_dirs,
    load_scenario,
)


def _ort_dylib() -> Path | None:
    capi = Path(ort.__file__).resolve().parent / "capi"
    for pat in ("libonnxruntime*.dylib", "libonnxruntime.so*", "onnxruntime.dll"):
        hits = sorted(capi.glob(pat))
        if hits:
            return hits[0]
    return None


def rss() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return {
        "rss_bytes": int(psutil.Process(os.getpid()).memory_info().rss),
        "rss_peak_bytes": int(peak),
    }


def pct(s: list[float], p: float) -> float:
    if not s:
        return 0.0
    r = (p / 100.0) * (len(s) - 1)
    lo, hi = int(math.floor(r)), int(math.ceil(r))
    w = r - lo
    return s[lo] * (1 - w) + s[hi] * w


def latency(warmup: int, samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "warmup": warmup,
        "iters": len(s),
        "mean_ms": statistics.fmean(s) if s else 0.0,
        "p50_ms": pct(s, 50),
        "p90_ms": pct(s, 90),
        "p99_ms": pct(s, 99),
        "min_ms": s[0] if s else 0.0,
        "max_ms": s[-1] if s else 0.0,
    }


def adapt_feed(sess: ort.InferenceSession, feed: dict) -> dict:
    name = sess.get_inputs()[0].name
    if name not in feed and len(feed) == 1:
        feed = {name: next(iter(feed.values()))}
    return {k: np.asarray(v, np.float16) for k, v in feed.items()}


def as_f32(outs) -> list:
    return [np.asarray(o, np.float32) for o in outs]


def session(path: Path, threads: int) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.inter_op_num_threads = 1
    opt.intra_op_num_threads = max(threads, 1)
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opt, providers=["CPUExecutionProvider"])


def lm_meta(model: int) -> tuple[str, int]:
    if model == -1:
        return "lm_modelT_opt.onnx", 56
    if model == -2:
        return "lm_modelV_opt.onnx", 112
    if model == -3:
        return "lm_modelU_opt.onnx", 112
    return f"lm_model{model}_opt.onnx", 224


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def bench_one(path: Path, feed: dict, threads: int, warmup: int, iters: int) -> tuple[dict, list]:
    t0 = time.perf_counter()
    sess = session(path, threads)
    startup = (time.perf_counter() - t0) * 1000
    feed = adapt_feed(sess, feed)
    t1 = time.perf_counter()
    first = as_f32(sess.run(None, feed))
    first_ms = (time.perf_counter() - t1) * 1000
    for _ in range(warmup):
        sess.run(None, feed)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feed)
        samples.append((time.perf_counter() - t) * 1000)
    return {
        "filename": path.name,
        "startup_ms": startup,
        "first_infer_ms": first_ms,
        "latency": latency(warmup, samples),
        "resources_after_infer": rss(),
        "accuracy": None,
    }, first


def detect_faces(outputs, maxpool, frame, thresh=0.6):
    outputs = np.array(outputs).copy()
    maxpool = np.array(maxpool)
    outputs[0, 0, np.abs(outputs[0, 0] - maxpool[0, 0]) > 1e-6] = 0
    idx = int(np.argmax(outputs[0, 0]))
    y, x = divmod(idx, 56)
    c = float(outputs[0, 0, y, x])
    if c < thresh:
        return np.zeros((0, 5), np.float32)
    r = float(outputs[0, 1, y, x]) * 112.0
    x, y = x * 4.0, y * 4.0
    box = np.array([[x - r, y - r, 2 * r, 2 * r, c]], np.float32)
    box[:, [0, 2]] *= frame.shape[1] / 224.0
    box[:, [1, 3]] *= frame.shape[0] / 224.0
    return box


def logit(p, factor=16.0):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)) / factor


def decode_lms(tensor, crop, model):
    res = 223.0 if model >= 0 else (55.0 if model == -1 else 111.0)
    out_res = 27.0 if model >= 0 else (6.0 if model == -1 else 13.0)
    gi = int(out_res) + 1
    factor = 8.0 if model == -1 else 16.0
    c0 = 30 if model == -1 else 66
    crop_x1, crop_y1, sx, sy = crop
    main = tensor[0:c0].reshape(c0, gi * gi)
    tm = main.argmax(1)
    idx = tm[:, None]
    conf = np.take_along_axis(main, idx, 1).reshape(c0)
    ox = res * logit(np.take_along_axis(tensor[c0 : 2 * c0].reshape(c0, gi * gi), idx, 1).reshape(c0), factor)
    oy = res * logit(np.take_along_axis(tensor[2 * c0 : 3 * c0].reshape(c0, gi * gi), idx, 1).reshape(c0), factor)
    tx = crop_y1 + sy * (res * np.floor(tm / gi) / out_res + ox)
    ty = crop_x1 + sx * (res * np.floor(np.mod(tm, gi)) / out_res + oy)
    return float(conf.mean()), np.stack([tx, ty, conf], 1)


def crop_xyxy(frame, box, pad_x=0.1, pad_y=0.125):
    x, y, w, h = [float(v) for v in box[:4]]
    h_img, w_img = frame.shape[:2]

    def clamp(px, py):
        return int(min(max(px, 0), w_img - 1)), int(min(max(py, 0), h_img - 1)) + 1

    x1, y1 = clamp(x - int(w * pad_x), y - int(h * pad_y))
    x2, y2 = clamp(x + w + int(w * pad_x), y + h + int(h * pad_y))
    return x1, y1, x2, y2


def lms_bbox(lms: np.ndarray) -> np.ndarray:
    rows, cols = lms[:, 0], lms[:, 1]
    x, y = float(cols.min()), float(rows.min())
    return np.array([x, y, max(float(cols.max() - x), 1.0), max(float(rows.max() - y), 1.0), 1.0], np.float32)


def eye_conf(lms: np.ndarray) -> float | None:
    if lms is None or len(lms) < 48:
        return None
    return float(lms[list(EYE_IDX), 2].mean())


def gaze_input(frame: np.ndarray, lms: np.ndarray) -> np.ndarray | None:
    if lms is None or len(lms) < 46:
        return None
    eyes = []
    for a, b, flip in ((36, 39, False), (42, 45, True)):
        cx = (float(lms[a, 1]) + float(lms[b, 1])) / 2.0
        cy = (float(lms[a, 0]) + float(lms[b, 0])) / 2.0
        rad = math.hypot(float(lms[b, 1]) - float(lms[a, 1]), float(lms[b, 0]) - float(lms[a, 0])) / 2.0
        rad = max(rad * 1.4, 4.0)
        x1, y1 = int(cx - rad), int(cy - rad * 0.86)
        x2, y2 = int(cx + rad), int(cy + rad * 0.86)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        if flip:
            crop = cv2.flip(crop, 1)
        eyes.append(imagenet_nchw(crop, 32)[0])
    return np.stack(eyes, 0)


def decode_gaze_conf(out0: np.ndarray) -> float:
    arr = np.asarray(out0, np.float32)
    return float(arr.max()) if arr.size else 0.0


def _xy(p) -> tuple[float, float]:
    return (p["x"], p["y"]) if isinstance(p, dict) else (float(p[0]), float(p[1]))


def seed_landmarks(frame, det_sess, lm_sess, lm_size, model):
    dout = as_f32(det_sess.run(None, adapt_feed(det_sess, {"input": imagenet_nchw(frame, 224)})))
    dets = detect_faces(dout[0], dout[1], frame)
    if not len(dets):
        raise SystemExit("seed image has no face (detection empty)")
    box = dets[0]
    x1, y1, x2, y2 = crop_xyxy(frame, box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise SystemExit("seed face crop too small")
    crop = imagenet_nchw(frame[y1:y2, x1:x2], lm_size)
    out = as_f32(lm_sess.run(None, adapt_feed(lm_sess, {"input": crop})))[0]
    _, lms = decode_lms(out[0], (x1, y1, (x2 - x1) / lm_size, (y2 - y1) / lm_size), model)
    return box, lms


def run_frame(frame, det_sess, lm_sess, gaze_sess, lm_size, model, box, pad_x, pad_y, do_detect, do_gaze):
    detect_ms = crop_ms = pre_ms = lm_ms = decode_ms = gaze_ms = 0.0
    dets, lms, gconf = [], None, None
    crop_w = crop_h = 0
    t_all = time.perf_counter()
    if do_detect:
        t = time.perf_counter()
        dout = as_f32(det_sess.run(None, adapt_feed(det_sess, {"input": imagenet_nchw(frame, 224)})))
        detect_ms = (time.perf_counter() - t) * 1000
        dets = detect_faces(dout[0], dout[1], frame)
        if len(dets):
            box = dets[0]
    if box is None:
        return {
            "detect_ms": detect_ms,
            "crop_ms": 0.0,
            "pre_ms": 0.0,
            "lm_ms": 0.0,
            "decode_ms": 0.0,
            "gaze_ms": 0.0,
            "e2e_ms": (time.perf_counter() - t_all) * 1000,
            "scanned": do_detect,
            "faces": 0,
            "det_score": None,
            "lm_conf": None,
            "eye_conf": None,
            "gaze_conf": None,
            "crop_w": 0,
            "crop_h": 0,
            "box": None,
            "lms": None,
        }
    t = time.perf_counter()
    x1, y1, x2, y2 = crop_xyxy(frame, box, pad_x, pad_y)
    crop_w, crop_h = max(x2 - x1, 0), max(y2 - y1, 0)
    patch = frame[y1:y2, x1:x2] if crop_w >= 4 and crop_h >= 4 else None
    crop_ms = (time.perf_counter() - t) * 1000
    if patch is None:
        box = None
        lms = None
    else:
        t = time.perf_counter()
        lin = imagenet_nchw(patch, lm_size)
        pre_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        out = as_f32(lm_sess.run(None, adapt_feed(lm_sess, {"input": lin})))[0]
        lm_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        conf, lms = decode_lms(out[0], (x1, y1, (x2 - x1) / lm_size, (y2 - y1) / lm_size), model)
        decode_ms = (time.perf_counter() - t) * 1000
        box = lms_bbox(lms)
        box[4] = conf
        if do_gaze and gaze_sess is not None:
            t = time.perf_counter()
            gin = gaze_input(frame, lms)
            if gin is not None:
                gout = as_f32(gaze_sess.run(None, adapt_feed(gaze_sess, {"input": gin})))
                gconf = decode_gaze_conf(gout[0])
            gaze_ms = (time.perf_counter() - t) * 1000
    e2e_ms = (time.perf_counter() - t_all) * 1000
    faces = 1 if lms is not None else 0
    det_score = float(dets[0][4]) if len(dets) else (float(box[4]) if box is not None else None)
    return {
        "detect_ms": detect_ms,
        "crop_ms": crop_ms,
        "pre_ms": pre_ms,
        "lm_ms": lm_ms,
        "decode_ms": decode_ms,
        "gaze_ms": gaze_ms,
        "e2e_ms": e2e_ms,
        "scanned": do_detect,
        "faces": faces,
        "det_score": det_score,
        "lm_conf": float(box[4]) if box is not None else None,
        "eye_conf": eye_conf(lms) if lms is not None else None,
        "gaze_conf": gconf,
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "box": box,
        "lms": lms,
    }


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def run_scenario(meta, frames, det_sess, lm_sess, gaze_sess, lm_size, model, warmup: int) -> dict:
    pad_x, pad_y = float(meta.get("pad_x", 0.1)), float(meta.get("pad_y", 0.125))
    scan_every = max(int(meta.get("scan_every", 1)), 1)
    do_gaze = bool(meta.get("gaze")) and gaze_sess is not None
    box = None
    for _ in range(max(warmup, 1)):
        run_frame(frames[0], det_sess, lm_sess, gaze_sess, lm_size, model, None, pad_x, pad_y, True, do_gaze)
    rows = []
    for i, frame in enumerate(frames):
        scanned = box is None or (i % scan_every == 0)
        row = run_frame(
            frame, det_sess, lm_sess, gaze_sess, lm_size, model, None if scanned else box, pad_x, pad_y, scanned, do_gaze
        )
        box = row["box"]
        rows.append(row)
    last = rows[-1] if rows else {}
    lms = last.get("lms")
    stage = {}
    for key in ("crop_ms", "pre_ms", "lm_ms", "decode_ms", "gaze_ms", "e2e_ms"):
        stage[key] = latency(warmup, [r[key] for r in rows])
    stage["detect_ms"] = latency(warmup, [r["detect_ms"] for r in rows if r["scanned"] or r["detect_ms"] > 0])
    scan = [r["e2e_ms"] for r in rows if r["scanned"]]
    track = [r["e2e_ms"] for r in rows if not r["scanned"]]
    return {
        "name": meta["name"],
        "tags": meta.get("tags", []),
        "frames": len(frames),
        "scan_every": scan_every,
        "gaze": do_gaze,
        "glasses": bool(meta.get("glasses")),
        "detect_ms": stage["detect_ms"],
        "crop_ms": stage["crop_ms"],
        "pre_ms": stage["pre_ms"],
        "lm_ms": stage["lm_ms"],
        "decode_ms": stage["decode_ms"],
        "gaze_ms": stage["gaze_ms"],
        "e2e_ms": stage["e2e_ms"],
        "scan_p50_ms": pct(sorted(scan), 50) if scan else None,
        "track_p50_ms": pct(sorted(track), 50) if track else None,
        "crop_w": last.get("crop_w", 0),
        "crop_h": last.get("crop_h", 0),
        "faces": last.get("faces", 0),
        "det_score": mean_or_none([r["det_score"] for r in rows]),
        "lm_conf": mean_or_none([r["lm_conf"] for r in rows]),
        "eye_conf": mean_or_none([r["eye_conf"] for r in rows]),
        "gaze_conf": mean_or_none([r["gaze_conf"] for r in rows]),
        "landmarks": lms.tolist() if lms is not None else [],
    }


def glasses_delta(scenarios: dict) -> dict:
    bases = {s.name: s.baseline for s in REALISTIC if s.baseline}
    out = {}
    for name, sc in scenarios.items():
        base_name = bases.get(name)
        if not base_name or base_name not in scenarios:
            continue
        base = scenarios[base_name]
        a, b = base.get("landmarks") or [], sc.get("landmarks") or []
        n = min(len(a), len(b))
        eye_mae = None
        if n >= 48:
            dists = [
                math.hypot(_xy(a[i])[0] - _xy(b[i])[0], _xy(a[i])[1] - _xy(b[i])[1]) for i in EYE_IDX
            ]
            eye_mae = float(sum(dists) / len(dists))
        ae, be = base.get("eye_conf"), sc.get("eye_conf")
        out[name] = {
            "baseline": base_name,
            "eye_conf_delta": (be - ae) if ae is not None and be is not None else None,
            "eye_mae_px": eye_mae,
        }
    return out


def python_scenarios(args, seed, dump: Path) -> dict:
    md = Path(args.models_dir)
    lm_name, lm_size = lm_meta(args.model)
    det_sess = session(md / "mnv3_detection_opt.onnx", args.threads)
    lm_sess = session(md / lm_name, args.threads)
    gaze_path = md / "mnv3_gaze32_split_opt.onnx"
    gaze_sess = session(gaze_path, args.threads) if gaze_path.is_file() else None
    box, lms = seed_landmarks(seed, det_sess, lm_sess, lm_size, args.model)
    face, origin, face_h = face_crop(seed, box)
    names = set(args.scenarios.split(",")) if args.scenarios else None
    if names:
        for spec in REALISTIC:
            if spec.name in names and spec.baseline:
                names.add(spec.baseline)
    photos = []
    want_photo = names is None or "glasses_photo" in names or any(n.startswith("glasses_photo") for n in (names or []))
    if want_photo:
        photos = cached_photos()
        if not photos:
            try:
                photos = fetch_all()
            except Exception as e:
                print(f"glasses_photo skipped: {e}", file=sys.stderr)
                photos = []
    extra = []
    if args.wflw_root:
        root = Path(args.wflw_root)
        extra = [p for p in list(root.rglob("*.jpg"))[:4] + list(root.rglob("*.png"))[:4] if p.is_file()]
        photos = list(photos) + extra
    dirs = generate_scenarios(
        dump,
        face,
        lms,
        origin,
        face_h,
        names=names,
        n_frames_override=args.frames,
        photos=photos,
    )
    if args.scan_every is not None:
        for d in dirs:
            meta_p = d / "meta.json"
            meta = json.loads(meta_p.read_text())
            meta["scan_every"] = args.scan_every
            meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    scenarios = {}
    for d in list_scenario_dirs(dump):
        if names and d.name not in names and not (
            "glasses_photo" in names and d.name.startswith("glasses_photo")
        ):
            continue
        meta, frames = load_scenario(d)
        scenarios[meta["name"]] = run_scenario(
            meta, frames, det_sess, lm_sess, gaze_sess, lm_size, args.model, args.warmup
        )
    return {
        "backend": "onnxruntime-python",
        "runtime_version": ort.__version__,
        "python_version": sys.version.split()[0],
        "threads": args.threads,
        "scenarios": scenarios,
        "glasses_delta": glasses_delta(scenarios),
    }


def python_bench(args, dump: Path) -> dict:
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")
    md = Path(args.models_dir)
    models = {}
    tensors = {}

    det_in = imagenet_nchw(frame, 224)
    det_rep, det_out = bench_one(md / "mnv3_detection_opt.onnx", {"input": det_in}, args.threads, args.warmup, args.iters)
    models["detection"] = det_rep
    tensors["detection_input"] = det_in
    tensors["detection_output_0"] = det_out[0]
    tensors["detection_output_1"] = det_out[1]

    lm_name, lm_size = lm_meta(args.model)
    lm_in = imagenet_nchw(frame, lm_size)
    lm_rep, lm_out = bench_one(md / lm_name, {"input": lm_in}, args.threads, args.warmup, args.iters)
    models[lm_name.replace(".onnx", "")] = lm_rep
    tensors["landmarks_input"] = lm_in
    tensors["landmarks_output"] = lm_out[0]

    gaze = md / "mnv3_gaze32_split_opt.onnx"
    if gaze.is_file():
        gin = np.zeros((2, 3, 32, 32), np.float32)
        grep, gout = bench_one(gaze, {"input": gin}, args.threads, args.warmup, args.iters)
        models["gaze"] = grep
        tensors["gaze_input"] = gin
        tensors["gaze_output"] = gout[0]

    rf = md / "retinaface_640x640_opt.onnx"
    if rf.is_file():
        rin = retina_nchw(frame)
        rrep, rout = bench_one(rf, {"input0": rin}, args.threads, args.warmup, args.iters)
        models["retinaface"] = rrep
        tensors["retinaface_input"] = rin
        tensors["retinaface_output_0"] = rout[0]

    det_sess = session(md / "mnv3_detection_opt.onnx", args.threads)
    lm_sess = session(md / lm_name, args.threads)

    def run_pipeline():
        t0 = time.perf_counter()
        dout = as_f32(det_sess.run(None, adapt_feed(det_sess, {"input": imagenet_nchw(frame, 224)})))
        dets = detect_faces(dout[0], dout[1], frame)
        detect_ms = (time.perf_counter() - t0) * 1000
        detections, landmarks, faces, landmarks_ms = [], [], 0, 0.0
        if len(dets):
            x, y, w, h, score = dets[0]
            detections.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "score": float(score)})
            x1, y1, x2, y2 = crop_xyxy(frame, dets[0])
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                t1 = time.perf_counter()
                crop = imagenet_nchw(frame[y1:y2, x1:x2], lm_size)
                out = as_f32(lm_sess.run(None, adapt_feed(lm_sess, {"input": crop})))[0]
                landmarks_ms = (time.perf_counter() - t1) * 1000
                _, lms = decode_lms(out[0], (x1, y1, (x2 - x1) / lm_size, (y2 - y1) / lm_size), args.model)
                faces = 1
                landmarks = [{"x": float(a), "y": float(b), "conf": float(c)} for a, b, c in lms]
        return detect_ms, landmarks_ms, faces, detections, landmarks

    for _ in range(max(args.warmup, 1)):
        run_pipeline()
    detect_ms, landmarks_ms, faces, detections, landmarks = run_pipeline()

    dump.mkdir(parents=True, exist_ok=True)
    meta = {"tensors": {}, "detections": detections, "landmarks": landmarks}
    for k, arr in tensors.items():
        write_f32(dump / f"{k}.bin", arr)
        meta["tensors"][k] = {"file": f"{k}.bin", "shape": list(arr.shape)}
    (dump / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "backend": "onnxruntime-python",
        "runtime_version": ort.__version__,
        "python_version": sys.version.split()[0],
        "threads": args.threads,
        "models": models,
        "pipeline": {
            "faces": faces,
            "detect_ms": detect_ms,
            "landmarks_ms": landmarks_ms,
            "e2e_ms": detect_ms + landmarks_ms,
        },
    }


def _stage_p50(sc: dict, key: str) -> float:
    v = sc.get(key)
    if isinstance(v, dict):
        return float(v.get("p50_ms") or 0.0)
    return float(v or 0.0)


def compare_scenarios(py: dict, rs: dict) -> None:
    pa, pb = py.get("scenarios") or {}, rs.get("scenarios") or {}
    if not pa:
        return
    print()
    header = (
        f"{'scenario':<22} {'e2e_p50':>9} {'crop_p50':>9} {'lm_p50':>8} "
        f"{'eye_cf':>8} {'crop':>11} {'rust/py':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(set(pa) | set(pb)):
        a, b = pa.get(name), pb.get(name)
        if not a:
            continue
        e2e = _stage_p50(a, "e2e_ms")
        crop = _stage_p50(a, "crop_ms")
        lm = _stage_p50(a, "lm_ms")
        eye = a.get("eye_conf")
        eye_s = f"{eye:.3f}" if eye is not None else "-"
        geom = f"{a.get('crop_w', 0)}x{a.get('crop_h', 0)}"
        ratio = "-"
        if b:
            be = _stage_p50(b, "e2e_ms")
            ratio = f"{be / e2e:.2f}x" if e2e else "-"
        print(f"{name:<22} {e2e:9.3f} {crop:9.3f} {lm:8.3f} {eye_s:>8} {geom:>11} {ratio:>8}")
        if b and a.get("landmarks") and b.get("landmarks"):
            al, bl = a["landmarks"], b["landmarks"]
            n = min(len(al), len(bl))
            if n:
                mae = sum(
                    math.hypot(_xy(al[i])[0] - _xy(bl[i])[0], _xy(al[i])[1] - _xy(bl[i])[1])
                    for i in range(n)
                ) / n
                print(f"{'':<22} {'lms_mae_px':<9} {mae:.4f}")
        scan, track = a.get("scan_p50_ms"), a.get("track_p50_ms")
        if scan is not None or track is not None:
            ss = f"{scan:.3f}" if scan is not None else "-"
            ts = f"{track:.3f}" if track is not None else "-"
            print(f"{'':<22} scan_p50={ss}  track_p50={ts}")
    delta = py.get("glasses_delta") or {}
    if delta:
        print()
        print(f"{'glasses':<22} {'eye_dconf':>10} {'eye_mae_px':>12}")
        for name, d in delta.items():
            dc = d.get("eye_conf_delta")
            mae = d.get("eye_mae_px")
            dc_s = f"{dc:.4f}" if isinstance(dc, (int, float)) else "-"
            mae_s = f"{mae:.4f}" if isinstance(mae, (int, float)) else "-"
            print(f"{name:<22} {dc_s:>10} {mae_s:>12}")


def compare(py: dict, rs: dict) -> None:
    print("A/B  Python onnxruntime  vs  Rust ort")
    dylib = rs.get("ort_dylib") or "static"
    print(
        f"  python {py.get('runtime_version')}  rust crate {rs.get('crate_version')}  "
        f"device={rs.get('device', 'cpu')}  ort={Path(str(dylib)).name}"
    )
    if py.get("models") and rs.get("models"):
        header = f"{'model':<18} {'metric':<16} {'python':>10} {'ort-rust':>10} {'rust/py':>8}"
        print(header)
        print("-" * len(header))

        def line(model, metric, a, b, kind="ms"):
            if kind == "mib":
                pa, pb = f"{a/1024/1024:.1f}", f"{b/1024/1024:.1f}"
            else:
                pa, pb = f"{a:.3f}", f"{b:.3f}"
            ratio = f"{b/a:.2f}x" if a else "-"
            print(f"{model:<18} {metric:<16} {pa:>10} {pb:>10} {ratio:>8}")

        for key in sorted(set(py["models"]) & set(rs["models"])):
            a, b = py["models"][key], rs["models"][key]
            line(key, "startup_ms", a["startup_ms"], b["startup_ms"])
            line(key, "first_ms", a["first_infer_ms"], b["first_infer_ms"])
            line(key, "p50_ms", a["latency"]["p50_ms"], b["latency"]["p50_ms"])
            line(key, "rss_mib", a["resources_after_infer"]["rss_bytes"], b["resources_after_infer"]["rss_bytes"], "mib")
            acc = b.get("accuracy")
            if acc:
                print(f"{key:<18} {'max_abs':<16} {'0':>10} {acc['max_abs']:<10.3g} {'-':>8}")
                print(f"{key:<18} {'cosine':<16} {'1':>10} {acc['cosine']:<10.6f} {'-':>8}")
        if py.get("pipeline") and rs.get("pipeline"):
            pa, pb = py["pipeline"], rs["pipeline"]
            line("pipeline", "e2e_ms", pa["e2e_ms"], pb["e2e_ms"])
            if pb.get("det_iou") is not None:
                print(f"{'pipeline':<18} {'det_iou':<16} {'1':>10} {pb['det_iou']:<10.4f} {'-':>8}")
            if pb.get("landmark_mae_px") is not None:
                print(f"{'pipeline':<18} {'lms_mae_px':<16} {'0':>10} {pb['landmark_mae_px']:<10.4f} {'-':>8}")
    compare_scenarios(py, rs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--image", default=str(ROOT / "models" / "benchmark.bin"))
    p.add_argument("--model", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "out"))
    p.add_argument("--suite", default="micro", choices=["micro", "realistic", "all"])
    p.add_argument("--scenarios", default=None, help="comma-separated scenario names")
    p.add_argument("--frames", type=int, default=None, help="override synthetic frame count")
    p.add_argument("--scan-every", type=int, default=None)
    p.add_argument("--wflw-root", default=None, help="optional WFLW image root")
    args = p.parse_args()

    out = Path(args.out_dir)
    dump = out / "dump"
    scen_dir = out / "scenarios"
    out.mkdir(parents=True, exist_ok=True)

    run_micro = args.suite in ("micro", "all")
    run_real = args.suite in ("realistic", "all")
    py: dict = {
        "backend": "onnxruntime-python",
        "runtime_version": ort.__version__,
        "python_version": sys.version.split()[0],
        "threads": args.threads,
        "suite": args.suite,
    }
    if run_micro:
        micro = python_bench(args, dump)
        py.update({k: v for k, v in micro.items() if k not in ("backend", "runtime_version", "python_version", "threads")})
    if run_real:
        seed = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if seed is None:
            raise SystemExit(f"cannot read {args.image}")
        real = python_scenarios(args, seed, scen_dir)
        py["scenarios"] = real["scenarios"]
        py["glasses_delta"] = real["glasses_delta"]
        (out / "scenarios.json").write_text(json.dumps({"scenarios": real["scenarios"], "glasses_delta": real["glasses_delta"]}, indent=2))
    (out / "python.json").write_text(json.dumps(py, indent=2))

    if args.device == "gpu":
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "runtime-ort" / "scripts" / "wrap_preprocess.py"),
                "--models-dir",
                args.models_dir,
                "--out-dir",
                str(Path(args.models_dir) / "pre"),
            ],
            check=False,
        )

    crate = ROOT / "runtime-ort"
    rust_bin = crate / "target" / "release" / "osf-bench"
    features = []
    if args.device == "gpu":
        features.append("gpu")
    dylib = _ort_dylib()
    if dylib:
        features.append("shared-ort")
    build = ["cargo", "build", "--release", "--bin", "osf-bench"]
    if features:
        build += ["--features", ",".join(features)]
    subprocess.run(build, cwd=crate, check=True)
    env = os.environ.copy()
    if dylib:
        env["ORT_DYLIB_PATH"] = str(dylib)
    common = [
        "--models-dir", args.models_dir, "--image", args.image, "--model", str(args.model),
        "--threads", str(args.threads), "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--device", args.device, "--suite", args.suite,
    ]
    rust_cmd = [str(rust_bin), *common, "--out", str(out / "rust.json")]
    if run_micro:
        rust_cmd += ["--ref-dir", str(dump)]
    if run_real:
        rust_cmd += ["--scenario-dir", str(scen_dir)]
    subprocess.run(rust_cmd, env=env, check=True)
    rs = json.loads((out / "rust.json").read_text())
    compare(py, rs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
