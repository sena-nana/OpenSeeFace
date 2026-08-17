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


def session(path: Path, threads: int) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.inter_op_num_threads = 1
    opt.intra_op_num_threads = max(threads, 1)
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opt, providers=["CPUExecutionProvider"])


_MEAN = np.float32([0.485, 0.456, 0.406])
_STD = np.float32([0.229, 0.224, 0.225])
_MEAN = -(_MEAN / _STD)
_STD = 1.0 / (_STD * 255.0)


def imagenet_nchw(bgr: np.ndarray, size: int) -> np.ndarray:
    im = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    im = np.float32(im) * _STD + _MEAN
    return np.transpose(np.expand_dims(im, 0), (0, 3, 1, 2)).astype(np.float32)


def retina_nchw(bgr: np.ndarray) -> np.ndarray:
    im = np.float32(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR))
    im -= (104, 117, 123)
    return np.expand_dims(im.transpose(2, 0, 1), 0)


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
    name = sess.get_inputs()[0].name
    if name not in feed and len(feed) == 1:
        feed = {name: next(iter(feed.values()))}
    t1 = time.perf_counter()
    first = sess.run(None, feed)
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
    t0 = time.perf_counter()
    dout = det_sess.run(None, {det_sess.get_inputs()[0].name: imagenet_nchw(frame, 224)})
    dets = detect_faces(dout[0], dout[1], frame)
    detect_ms = (time.perf_counter() - t0) * 1000
    detections, landmarks, faces, landmarks_ms = [], [], 0, 0.0
    if len(dets):
        x, y, w, h, score = dets[0]
        detections.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "score": float(score)})
        h_img, w_img = frame.shape[:2]

        def clamp(px, py):
            return int(min(max(px, 0), w_img - 1)), int(min(max(py, 0), h_img - 1)) + 1

        x1, y1 = clamp(x - int(w * 0.1), y - int(h * 0.125))
        x2, y2 = clamp(x + w + int(w * 0.1), y + h + int(h * 0.125))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            t1 = time.perf_counter()
            crop = np.float32(frame[y1:y2, x1:x2, ::-1])
            crop = cv2.resize(crop, (lm_size, lm_size), interpolation=cv2.INTER_LINEAR)
            crop = np.transpose(np.expand_dims(crop * _STD + _MEAN, 0), (0, 3, 1, 2)).astype(np.float32)
            out = lm_sess.run(None, {lm_sess.get_inputs()[0].name: crop})[0]
            landmarks_ms = (time.perf_counter() - t1) * 1000
            _, lms = decode_lms(out[0], (x1, y1, (x2 - x1) / lm_size, (y2 - y1) / lm_size), args.model)
            faces = 1
            landmarks = [{"x": float(a), "y": float(b), "conf": float(c)} for a, b, c in lms]

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


def compare(py: dict, rs: dict) -> None:
    print("A/B  Python onnxruntime  vs  Rust ort")
    print(f"  python {py.get('runtime_version')}  rust crate {rs.get('crate_version')}")
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
    pa, pb = py["pipeline"], rs["pipeline"]
    line("pipeline", "e2e_ms", pa["e2e_ms"], pb["e2e_ms"])
    if pb.get("det_iou") is not None:
        print(f"{'pipeline':<18} {'det_iou':<16} {'1':>10} {pb['det_iou']:<10.4f} {'-':>8}")
    if pb.get("landmark_mae_px") is not None:
        print(f"{'pipeline':<18} {'lms_mae_px':<16} {'0':>10} {pb['landmark_mae_px']:<10.4f} {'-':>8}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--image", default=str(ROOT / "models" / "benchmark.bin"))
    p.add_argument("--model", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "out"))
    args = p.parse_args()

    out = Path(args.out_dir)
    dump = out / "dump"
    out.mkdir(parents=True, exist_ok=True)

    py = python_bench(args, dump)
    (out / "python.json").write_text(json.dumps(py, indent=2))

    crate = ROOT / "runtime-ort"
    rust_bin = crate / "target" / "release" / "osf-bench"
    if not rust_bin.is_file():
        subprocess.run(["cargo", "build", "--release", "--bin", "osf-bench"], cwd=crate, check=True)
    common = [
        "--models-dir", args.models_dir, "--image", args.image, "--model", str(args.model),
        "--threads", str(args.threads), "--warmup", str(args.warmup), "--iters", str(args.iters),
    ]
    subprocess.run(
        [str(rust_bin), *common, "--out", str(out / "rust.json"), "--ref-dir", str(dump)],
        check=True,
    )
    rs = json.loads((out / "rust.json").read_text())
    compare(py, rs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
