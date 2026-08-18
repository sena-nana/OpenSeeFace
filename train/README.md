# Training (Python)

PyTorch architectures and Adaptive Wing Loss live in `model.py`. There is no in-repo dataloader or training loop; original extra data is not redistributable (see the root README).

```bash
uv sync
uv run python train/export_onnx.py --help
```

Checkpoints (`*.pth`) can be exported to the ONNX names the Rust `facetracker` loads (`lm_model*_opt.onnx`, `mnv3_detection_opt.onnx`, `mnv3_gaze32_split_opt.onnx`).

`geffnet.mobilenetv3._gen_mobilenet_v3` must return constructor kwargs (patch based on geffnet commit `c450c12`).
