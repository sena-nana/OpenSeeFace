#!/usr/bin/env python3
"""Export `train/model.py` checkpoints to ONNX names the Rust tracker loads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    OpenSeeFaceDetect,
    OpenSeeFaceGaze,
    OpenSeeFaceLandmarks,
    OpenSeeFaceLandmarks30Pt,
)


def export(model: torch.nn.Module, dummy: torch.Tensor, dest: Path, names: tuple[str, ...] | None = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    kwargs = dict(
        f=str(dest),
        input_names=["input"],
        opset_version=11,
    )
    if names:
        kwargs["output_names"] = list(names)
    try:
        torch.onnx.export(model, dummy, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, **kwargs)
    print(f"wrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=ROOT / "models")
    p.add_argument(
        "--which",
        nargs="*",
        default=["detect", "gaze", "lm0", "lm1", "lm2", "lm3", "lmT"],
        help="Subset: detect gaze lm0 lm1 lm2 lm3 lmT",
    )
    args = p.parse_args()
    w = args.weights_dir
    out = args.out_dir

    specs = {
        "detect": (
            OpenSeeFaceDetect(),
            w / "detection.pth",
            torch.randn(1, 3, 224, 224),
            out / "mnv3_detection_opt.onnx",
            ("output", "maxpool"),
        ),
        "gaze": (
            OpenSeeFaceGaze(),
            w / "gaze.pth",
            torch.randn(2, 3, 32, 32),
            out / "mnv3_gaze32_split_opt.onnx",
            None,
        ),
        "lm0": (
            OpenSeeFaceLandmarks("small", 0.5),
            w / "lm_model0.pth",
            torch.randn(1, 3, 224, 224),
            out / "lm_model0_opt.onnx",
            None,
        ),
        "lm1": (
            OpenSeeFaceLandmarks("small", 1.0),
            w / "lm_model1.pth",
            torch.randn(1, 3, 224, 224),
            out / "lm_model1_opt.onnx",
            None,
        ),
        "lm2": (
            OpenSeeFaceLandmarks("large", 0.75),
            w / "lm_model2.pth",
            torch.randn(1, 3, 224, 224),
            out / "lm_model2_opt.onnx",
            None,
        ),
        "lm3": (
            OpenSeeFaceLandmarks("large", 1.0),
            w / "lm_model3.pth",
            torch.randn(1, 3, 224, 224),
            out / "lm_model3_opt.onnx",
            None,
        ),
        "lmT": (
            OpenSeeFaceLandmarks30Pt(),
            w / "lm_modelT.pth",
            torch.randn(1, 3, 56, 56),
            out / "lm_modelT_opt.onnx",
            None,
        ),
    }
    for key in args.which:
        if key not in specs:
            raise SystemExit(f"unknown model {key}")
        net, ckpt, dummy, dest, names = specs[key]
        if ckpt.is_file():
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else:
            print(f"skip {key}: missing {ckpt} (architecture still exported)")
        export(net, dummy, dest, names)


if __name__ == "__main__":
    main()
