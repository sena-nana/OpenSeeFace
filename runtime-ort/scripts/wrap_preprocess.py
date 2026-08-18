#!/usr/bin/env python3
"""Build GPU preprocess ONNX graphs (Resize+Normalize, optional fuse).

Resize uses half_pixel + linear to match runtime-ort `nchw`. ImageNet swaps BGR→RGB
and applies (x/255-mean)/std.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
        sys.stderr.write("wrap_preprocess.py needs the `onnx` package (uv sync in repo root)\n")
    raise

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGENET_SCALE = 1.0 / (IMAGENET_STD * np.float32(255.0))
IMAGENET_BIAS = -(IMAGENET_MEAN / IMAGENET_STD)

OPSET = 17
SIZES = (56, 112, 224)
LM_FILES = (
    ("lm_model0_opt.onnx", 224),
    ("lm_model1_opt.onnx", 224),
    ("lm_model2_opt.onnx", 224),
    ("lm_model3_opt.onnx", 224),
    ("lm_model4_opt.onnx", 224),
    ("lm_modelT_opt.onnx", 56),
    ("lm_modelU_opt.onnx", 112),
    ("lm_modelV_opt.onnx", 112),
)


def _const(name: str, arr: np.ndarray) -> onnx.NodeProto:
    t = numpy_helper.from_array(arr, name=name)
    return helper.make_node("Constant", [], [name], name=f"const_{name}", value=t)


def _pre_nodes(size: int, src: str, dst: str, prefix: str) -> list:
    s = prefix
    nodes = [
        helper.make_node("Cast", [src], [f"{s}f32"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", [f"{s}f32"], [f"{s}nchw0"], perm=[0, 3, 1, 2]),
        _const(f"{s}roi", np.array([], np.float32)),
        _const(f"{s}scales", np.array([], np.float32)),
        _const(f"{s}sizes", np.array([1, 3, size, size], np.int64)),
        helper.make_node(
            "Resize",
            [f"{s}nchw0", f"{s}roi", f"{s}scales", f"{s}sizes"],
            [f"{s}resized"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        ),
        _const(f"{s}swap", np.array([2, 1, 0], np.int64)),
        helper.make_node("Gather", [f"{s}resized", f"{s}swap"], [f"{s}rgb"], axis=1),
        _const(f"{s}scale", IMAGENET_SCALE.reshape(1, 3, 1, 1)),
        _const(f"{s}bias", IMAGENET_BIAS.reshape(1, 3, 1, 1)),
        helper.make_node("Mul", [f"{s}rgb", f"{s}scale"], [f"{s}scaled"]),
        helper.make_node("Add", [f"{s}scaled", f"{s}bias"], [f"{s}normed"]),
        helper.make_node("Cast", [f"{s}normed"], [dst], to=TensorProto.FLOAT16),
    ]
    return nodes


def _model(nodes, inputs, outputs, name: str) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=8,
        producer_name="osf-wrap-preprocess",
    )
    model.doc_string = "OpenSeeFace GPU preprocess (half_pixel bilinear + affine norm)"
    onnx.checker.check_model(model, full_check=True)
    return model


def imagenet_pre(size: int) -> onnx.ModelProto:
    image = helper.make_tensor_value_info("image", TensorProto.UINT8, [1, "height", "width", 3])
    nchw = helper.make_tensor_value_info("nchw", TensorProto.FLOAT16, [1, 3, size, size])
    return _model(_pre_nodes(size, "image", "nchw", ""), [image], [nchw], f"pre_imagenet_{size}")


def imagenet_crop(size: int) -> onnx.ModelProto:
    image = helper.make_tensor_value_info("image", TensorProto.UINT8, [1, "height", "width", 3])
    starts = helper.make_tensor_value_info("starts", TensorProto.INT64, [4])
    ends = helper.make_tensor_value_info("ends", TensorProto.INT64, [4])
    nchw = helper.make_tensor_value_info("nchw", TensorProto.FLOAT16, [1, 3, size, size])
    nodes = [
        _const("axes", np.array([0, 1, 2, 3], np.int64)),
        _const("steps", np.array([1, 1, 1, 1], np.int64)),
        helper.make_node("Slice", ["image", "starts", "ends", "axes", "steps"], ["crop"]),
        *_pre_nodes(size, "crop", "nchw", "c_"),
    ]
    return _model(nodes, [image, starts, ends], [nchw], f"pre_crop_imagenet_{size}")


def fuse(pre: onnx.ModelProto, infer: onnx.ModelProto, out_name: str) -> onnx.ModelProto:
    pre_out = pre.graph.output[0].name
    inf_in = infer.graph.input[0].name
    pre_p = onnx.compose.add_prefix(pre, prefix="pre/")
    inf_p = onnx.compose.add_prefix(infer, prefix="inf/")
    src = f"pre/{pre_out}"
    dst = f"inf/{inf_in}"
    for node in inf_p.graph.node:
        for i, name in enumerate(node.input):
            if name == dst:
                node.input[i] = src
    opsets: dict[str, int] = {}
    for m in (pre_p, inf_p):
        for o in m.opset_import:
            opsets[o.domain] = max(opsets.get(o.domain, 0), o.version)
    graph = helper.make_graph(
        list(pre_p.graph.node) + list(inf_p.graph.node),
        f"fused_{Path(out_name).stem}",
        list(pre_p.graph.input),
        list(inf_p.graph.output),
        list(pre_p.graph.initializer) + list(inf_p.graph.initializer),
    )
    graph.value_info.extend(pre_p.graph.value_info)
    graph.value_info.extend(inf_p.graph.value_info)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()],
        ir_version=max(pre.ir_version, infer.ir_version, 8),
        producer_name="osf-wrap-preprocess",
    )
    model.doc_string = f"fused preprocess + {out_name}"
    return model


def save(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def self_check(size: int = 8) -> None:
    import onnxruntime as ort

    bgr = np.zeros((size, size, 3), np.uint8)
    bgr[..., 2] = 255
    model = imagenet_pre(size)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run(None, {"image": bgr[None]})[0], np.float32)
    assert abs(out[0, 0, 0, 0] - (255.0 * IMAGENET_SCALE[0] + IMAGENET_BIAS[0])) < 2e-3
    assert abs(out[0, 1, 0, 0] - IMAGENET_BIAS[1]) < 2e-3
    assert abs(out[0, 2, 0, 0] - IMAGENET_BIAS[2]) < 2e-3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        print("self-check ok")
        return 0
    out = args.out_dir or (args.models_dir / "pre")
    for s in SIZES:
        save(imagenet_pre(s), out / f"imagenet_{s}.onnx")
        save(imagenet_crop(s), out / f"imagenet_crop_{s}.onnx")
    mapping = [("mnv3_detection_opt.onnx", "imagenet_224.onnx")]
    mapping += [(f, f"imagenet_{size}.onnx") for f, size in LM_FILES]
    for src, pre_name in mapping:
        src_p = args.models_dir / src
        if not src_p.is_file():
            continue
        save(fuse(onnx.load(str(out / pre_name)), onnx.load(str(src_p)), src), out / src)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
