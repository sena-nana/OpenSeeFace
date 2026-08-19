![OSF.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/OSF.png)

# Overview

This project is a facial landmark tracking library based on MobileNetV3. Tracking runs in Rust (`runtime-ort` / `facetracker`) on the shipped ONNX models via [ort](https://github.com/pykeio/ort). There are four models, with different speed to tracking quality trade-offs.

It is **not** a stand-alone avatar puppeteering program. [VSeeFace](https://www.vseeface.icu/), [VTube Studio](https://denchisoft.com/), and a [Godot renderer](https://github.com/virtual-puppet-project/vpuppr) consume the UDP stream. Unity scripts here are a compatibility layer.

If anyone is curious, the name is a silly pun on the open seas and seeing faces. There's no deeper meaning.

An up to date sample video can be found [here](https://www.youtube.com/watch?v=AaNap_ud_3I&vq=hd1080), showing the default tracking model's performance under different noise and light levels.

# Tracking quality

Since the landmarks used by OpenSeeFace are a bit different from those used by other approaches (they are close to iBUG 68, with two less points in the mouth corners and quasi-3D face contours instead of face contours that follow the visible outline) it is hard to numerically compare its accuracy to that of other approaches found commonly in scientific literature. The tracking performance is also more optimized for making landmarks that are useful for animating an avatar than for exactly fitting the face image. For example, as long as the eye landmarks show whether the eyes are opened or closed, even if their location is somewhat off, they can still be useful for this purpose.

From general observation, OpenSeeFace performs well in adverse conditions (low light, high noise, low resolution) and keeps tracking faces through a very wide range of head poses with relatively high stability of landmark positions. Compared to MediaPipe, OpenSeeFace landmarks remain more stable in challenging conditions and it accurately represents a wider range of mouth poses. However, tracking of the eye region can be less accurate.

I ran OpenSeeFace on a sample clip from the video presentation for [3D Face Reconstruction with Dense Landmarks](https://microsoft.github.io/DenseLandmarks/) by Wood et al. to compare it to MediaPipe and their approach. You can watch the result [here](https://www.vseeface.icu/assets/media/OSFMediaPipe3DFR.mp4).

# Usage

The tracker is the `facetracker` CLI in `runtime-ort`. It reads a webcam, still, or video file, and writes one 1797-byte OpenSee packet per face to UDP (`127.0.0.1:11573` by default): 17 expression features, with slots 14–16 `MouthPucker`, `MouthOffsetX` (positive = person's right), and `CheekPuff`. Unity still reads older 1785-byte / 14-feature packets. Tracking can run on a different machine from the consumer.

If you downloaded a release on Windows, run `facetracker.exe` from the `Binary` folder (`run.bat` is a short camera demo). From source:

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin facetracker -- --help

After `./make_exe.sh` / `make_exe.bat`:

    dist/facetracker/facetracker --help

Webcam or video with the built-in overlay (no Unity required):

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin facetracker -- -c 0 --visualize 3 --pnp-points 1 --max-threads 4
    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin facetracker -- -c video.mp4 --visualize 3 --pnp-points 1 --max-threads 4

CPU is the default. GPU (CoreML on Apple, CUDA on NVIDIA) uses the same loop; detect/landmarks run on the EP (`--features gpu`, `--device gpu`). Gaze, RetinaFace, PnP, and UDP stay on CPU:

    cargo run --release --features gpu --manifest-path runtime-ort/Cargo.toml --bin facetracker -- -c 0 --device gpu --visualize 3 --pnp-points 1 --max-threads 4

# General notes

* The tracking seems to be quite robust even with partial occlusion of the face, glasses or bad lighting conditions.
* With glasses, next-frame crop switches to brows+nose when eye corners disagree with that fit. Blink skips EAR outliers / low eye conf; gaze holds the last pupil when heatmap conf drops and suppresses specular only when the eye crop is blown out. Sunglasses / missing iris still cannot recover appearance-based gaze.
* The highest quality model is selected with `--model 3`, the fastest model with the lowest tracking quality is `--model 0`.
* Lower tracking quality mainly means more rigid tracking, making it harder to detect blinking and eyebrow motion.
* Depending on the frame rate, face tracking can easily use up a whole CPU core. At 30fps for a single face, it should still use less than 100% of one core on a decent CPU. If tracking uses too much CPU, try lowering the frame rate. A frame rate of 20 is probably fine and anything above 30 should rarely be necessary.
* Once all `--faces` slots are filled, full-frame detection is skipped. The next crop comes from the previous frame's landmarks (eyes+nose fit). Detection runs again only after a lost face, or every `--scan-every` frames when fewer faces are tracked than `--faces`. Set `--faces` no higher than the number of faces you are actually tracking.
* `--filter` (default `one-euro`) smooths UDP pose and 2D landmarks after PnP. Strength follows landmark confidence, PnP stability, and motion speed. `--filter none` keeps raw measurements. Crop tracking and expression features are unchanged. If Unity `OpenSeeIKTarget.smooth` is also on, lower it to 0–0.1 to avoid double smoothing. A/B notes: [benchmarks/filter-eval.md](benchmarks/filter-eval.md).

# Unity compatibility

The binary is still named `facetracker` and sends the 1797-byte OpenSee packet, so `Unity/OpenSeeLauncher.cs` can start it. `--benchmark` / `--priority` are accepted for that launcher. Receiver scripts are in `Unity/` and `Examples/`; copy `OpenSee.trackingData` before use (it is written from another thread). Sample project: [OpenSeeFaceSample](https://github.com/emilianavt/OpenSeeFaceSample).

# Models

Four pretrained face landmark models are included. Using the `--model` switch, it is possible to select them for tracking. The given fps values are for running the model on a single face video on a single CPU core. Lowering the frame rate would reduce CPU usage by a corresponding degree.

* Model **-1**: This model is for running on toasters, so it's a very very fast and very low accuracy model. (213fps without gaze tracking)
* Model **0**: This is a very fast, low accuracy model. (68fps)
* Model **1**: This is a slightly slower model with better accuracy. (59fps)
* Model **2**: This is a slower model with good accuracy. (50fps)
* Model **3** (default): This is the slowest and highest accuracy model. (44fps)

FPS measurements are from running on one core of my CPU.

The shipped ONNX files use native float16 weights and I/O. Runtimes convert float32 buffers at the session edge.

Pytorch weights for use with `train/model.py` can be found [here](https://mega.nz/file/vvYXlYQT#h7FpEg4tmOCJNxjpsDEw0JomJIkVGKwrt4OUV0RNDDU). Some unoptimized ONNX models can be found [here](https://github.com/emilianavt/OpenSeeFace/issues/48).

# Results

## Landmarks

![Results1.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/Results1.png)

![Results2.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/Results2.png)

More samples: [Results3.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/Results3.png), [Results4.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/Results4.png)

## Face detection

The landmark model is quite robust with respect to the size and orientation of the faces, so the custom face detection model gets away with rougher bounding boxes than other approaches. It has a favorable speed to accuracy ratio for the purposes of this project.

![EmiFace.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/EmiFace.png)

# Release builds

The builds in the release section of this repository contain a `facetracker` binary (Windows: `facetracker.exe`) inside a `Binary` folder, built with Cargo from `runtime-ort`.

To run it, at least the `models` folder has to be placed in the same folder as `facetracker`. Placing it in a common parent folder should work too.

When distributing it, you should also distribute the `Licenses` folder along with it to make sure you conform to requirements set forth by some of the third party libraries. Unused models can be removed from redistributed packages without issue.

Local package:

     ./make_exe.sh
     # Windows: make_exe.bat

# Training (Python)

Tracking no longer uses Python. `train/` keeps the PyTorch architectures (`model.py`) and `export_onnx.py` so new weights can be written to the ONNX files the Rust tracker loads. See [train/README.md](train/README.md).

     uv sync
     uv run python train/export_onnx.py --help

`geffnet` and `torch` are required only for training/export. GPU fused preprocess graphs (`models/pre/`) are generated by `runtime-ort/scripts/wrap_preprocess.py` (needs `onnx`) at pack time or on first `--device gpu` run.

# Rust ORT runtime

`runtime-ort` runs the ONNX files in `models/` via [ort](https://github.com/pykeio/ort). The tracker binary is `facetracker` (`--device cpu|gpu`). Micro-benchmarks:

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --model 3 --threads 4

GPU (CoreML on Apple, CUDA on NVIDIA). Per-model `bench()` times bound inference only.
The GPU *pipeline* runs Resize+Normalize on the EP (fused MLProgram on CoreML; fused detect + CUDA Graph on NVIDIA)
so the CPU does not build f16 NCHW or read back full heatmaps between detect and landmarks:

    cargo run --release --features gpu --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --model 3 --threads 4 --device gpu

Realistic crop / glasses scenarios (synthetic webcam canvases, tracking loop, optional Wikimedia glasses photos). Default `--suite micro` is unchanged:

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite realistic --model 3 --threads 4
    cargo run --release --features gpu --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite realistic --device gpu --frames 8

Face-size adaptive tracking: CPU zooms the 224 detector on small faces and switches to the 112px landmark model on large ones; GPU only zooms detect. `--adaptive` on `osf-bench` enables it for the realistic suite. `--suite scale` A/Bs default thresholds against a fixed model-3 baseline:

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite scale
    cargo run --release --features gpu --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite scale --device gpu

Next-frame crop uses an eyes+nose similarity fit. After a face is locked, that crop is reused and the 224 detector is not run (`osf-bench --suite crop`):

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite crop --model 3 --threads 4

Glasses runtime path (bare vs synthetic rims/glare on the seed face; Wikimedia stills if `benchmarks/fixtures/cache/` is populated). Download the CC/PD portraits first:

    python3 benchmarks/fixtures/fetch.py
    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite glasses --model 3 --threads 4

Output post-process (`none` vs default `one-euro`) on the same raw pose and landmarks (`osf-bench --suite filter`). Decision notes: [benchmarks/filter-eval.md](benchmarks/filter-eval.md).

    cargo run --release --manifest-path runtime-ort/Cargo.toml --bin osf-bench -- --suite filter --model 3 --threads 4

Symmetry / relative-distance / 3D-projection landmark correction after PnP (alone and stacked with One Euro) was tried and rejected: dark-frame NME got worse and clean measurements were pulled off. Notes: [benchmarks/geom-eval.md](benchmarks/geom-eval.md).

Velocity / confidence / trajectory outlier rejection after PnP (alone and stacked with One Euro) was tried and rejected: hold lag pulled clean wander off, injected spikes did not improve NME, and the stack added step lag. Notes: [benchmarks/outlier-eval.md](benchmarks/outlier-eval.md).

# References

## Training dataset

The model was trained on a 66 point version of the [LS3D-W](https://www.adrianbulat.com/face-alignment) dataset.

    @inproceedings{bulat2017far,
      title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
      author={Bulat, Adrian and Tzimiropoulos, Georgios},
      booktitle={International Conference on Computer Vision},
      year={2017}
    }

Additional training has been done on the WFLW dataset after reducing it to 66 points and replacing the contour points and tip of the nose with points predicted by the model trained up to this point. This additional training is done to improve fitting to eyes and eyebrows.

    @inproceedings{wayne2018lab,
      author = {Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
      title = {Look at Boundary: A Boundary-Aware Face Alignment Algorithm},
      booktitle = {CVPR},
      month = June,
      year = {2018}
    }

For the training the gaze and blink detection model, the [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) dataset was used. Additionally, around 125000 synthetic eyes generated with [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) were used during training.

It should be noted that additional custom data was also used during the training process and that the reference landmarks from the original datasets have been modified in certain ways to address various issues. It is likely not possible to reproduce these models with just the original LS3D-W and WFLW datasets, however the additional data is not redistributable.

The heatmap regression based face detection model was trained on random 224x224 crops from the WIDER FACE dataset.

	@inproceedings{yang2016wider,
	  Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  Title = {WIDER FACE: A Face Detection Benchmark},
	  Year = {2016}
    }

## Algorithm

The algorithm is inspired by:

* [Designing Neural Network Architectures for Different Applications: From Facial Landmark Tracking to Lane Departure Warning System](https://www.synopsys.com/designware-ip/technical-bulletin/ulsee-designing-neural-network.html) by YiTa Wu, Vice President of Engineering, ULSee
* [Real-time Human Pose Estimation in the Browser with TensorFlow.js](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) by Olaf Ronneberger, Philipp Fischer, Thomas Brox
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam

The MobileNetV3 code was taken from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

For all training a modified version of [Adaptive Wing Loss](https://github.com/tankrant/Adaptive-Wing-Loss) was used.

* [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) by Xinyao Wang, Liefeng Bo, Li Fuxin

For expression detection, [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is used.

Face detection is done using a custom heatmap regression based face detection model or RetinaFace.

    @inproceedings{deng2019retinaface,
      title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
      author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
      booktitle={arxiv},
      year={2019}
    }

RetinaFace detection is based on [this](https://github.com/biubug6/Pytorch_Retinaface) implementation. The pretrained model was modified to remove unnecessary landmark detection and converted to ONNX format for a resolution of 640x640.

# Thanks!

Many thanks to everyone who helped me test things!

* [@Virtual_Deat](https://twitter.com/Virtual_Deat), who also inspired me to start working on this.
* [@ENiwatori](https://twitter.com/eniwatori) and family.
* [@ArgamaWitch](https://twitter.com/ArgamaWitch)
* [@AngelVayuu](https://twitter.com/AngelVayuu)
* [@DapperlyYours](https://twitter.com/DapperlyYours)
* [@comdost_art](https://twitter.com/comdost_art)
* [@Ponoki_Chan](https://twitter.com/Ponoki_Chan)

# License

The code and models are distributed under the BSD 2-clause license. 

You can find licenses of third party libraries used for binary builds in the `Licenses` folder.

