"""Realistic bench scenario specs and frame synthesis from a seed face."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# 66-point eye contour (OpenSeeFace / iBUG-style).
EYE_IDX = tuple(range(36, 48))


@dataclass
class ScenarioSpec:
    name: str
    tags: list[str]
    canvas: tuple[int, int]
    face_frac: float
    placement: str = "center"
    n_frames: int = 8
    scan_every: int = 1
    glasses: bool = False
    gaze: bool = False
    pad_x: float = 0.1
    pad_y: float = 0.125
    motion: bool = True
    baseline: str | None = None
    source: str = "seed"


# face_frac is detected-face-height / canvas-height (not the padded seed tile).
REALISTIC: list[ScenarioSpec] = [
    ScenarioSpec("webcam_720p_mid", ["crop"], (1280, 720), 0.32, motion=True),
    ScenarioSpec("webcam_1080p_far", ["crop"], (1920, 1080), 0.20, motion=True),
    ScenarioSpec("close_720p", ["crop"], (1280, 720), 0.55, motion=True),
    ScenarioSpec("edge_clamp", ["crop"], (1280, 720), 0.38, placement="edge", motion=True),
    ScenarioSpec(
        "pad_jitter",
        ["crop"],
        (1280, 720),
        0.32,
        pad_x=0.30,
        pad_y=0.325,
        motion=False,
    ),
    ScenarioSpec(
        "track_scan3",
        ["crop"],
        (1280, 720),
        0.32,
        n_frames=24,
        scan_every=3,
        motion=True,
    ),
    ScenarioSpec(
        "glasses_synth_off",
        ["glasses"],
        (1280, 720),
        0.38,
        n_frames=8,
        motion=True,
    ),
    ScenarioSpec(
        "glasses_synth",
        ["glasses"],
        (1280, 720),
        0.38,
        n_frames=8,
        glasses=True,
        motion=True,
        baseline="glasses_synth_off",
    ),
    ScenarioSpec(
        "gaze_glasses",
        ["glasses", "gaze"],
        (1280, 720),
        0.38,
        n_frames=8,
        glasses=True,
        gaze=True,
        motion=True,
    ),
]


def lm_xy(lms: np.ndarray, i: int) -> tuple[int, int]:
    """Image (col, row) from OpenSeeFace landmark (row, col, conf)."""
    return int(round(float(lms[i, 1]))), int(round(float(lms[i, 0])))


def overlay_glasses(bgr: np.ndarray, lms: np.ndarray) -> np.ndarray:
    if lms is None or len(lms) < 48:
        return bgr
    img = bgr.copy()
    r_o, r_i = lm_xy(lms, 36), lm_xy(lms, 39)
    l_i, l_o = lm_xy(lms, 42), lm_xy(lms, 45)
    ang = math.degrees(math.atan2(l_o[1] - r_o[1], l_o[0] - r_o[0]))
    rad = math.radians(ang)
    thick = max(int(math.hypot(r_i[0] - r_o[0], r_i[1] - r_o[1]) * 0.18), 3)

    def lens(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int, int, int]:
        c = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        w = max(int(dist * 0.92), 8)
        h = max(int(dist * 0.62), 6)
        layer = img.copy()
        cv2.ellipse(layer, c, (w, h), ang, 0, 360, (25, 40, 55), -1)
        cv2.addWeighted(layer, 0.72, img, 0.28, 0, img)
        cv2.ellipse(img, c, (w, h), ang, 0, 360, (8, 8, 8), thick)
        spec = (
            int(c[0] - 0.35 * w * math.cos(rad) + 0.2 * h * math.sin(rad)),
            int(c[1] - 0.35 * w * math.sin(rad) - 0.2 * h * math.cos(rad)),
        )
        cv2.ellipse(img, spec, (max(w // 4, 3), max(h // 5, 2)), ang, 0, 360, (220, 230, 240), -1)
        return c[0], c[1], w, h

    lens(r_o, r_i)
    lens(l_i, l_o)
    cv2.line(img, r_i, l_i, (8, 8, 8), thick)
    span = math.hypot(l_o[0] - r_o[0], l_o[1] - r_o[1])
    arm = max(int(span * 0.55), 12)

    def temple(outer: tuple[int, int], sign: float) -> None:
        dx, dy = math.cos(rad) * sign, math.sin(rad) * sign
        end = (int(outer[0] + arm * dx), int(outer[1] + arm * dy + arm * 0.18))
        cv2.line(img, outer, end, (8, 8, 8), thick)

    temple(r_o, -1.0)
    temple(l_o, 1.0)
    return img


def face_crop(
    frame: np.ndarray, box: np.ndarray, margin: float = 0.25
) -> tuple[np.ndarray, tuple[int, int], float]:
    x, y, w, h = [float(v) for v in box[:4]]
    x1 = int(max(x - w * margin, 0))
    y1 = int(max(y - h * margin, 0))
    x2 = int(min(x + w * (1 + margin), frame.shape[1]))
    y2 = int(min(y + h * (1 + margin), frame.shape[0]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame, (0, 0), float(frame.shape[0])
    inner_h = max(min(y + h, y2) - max(y, y1), 8.0)
    return frame[y1:y2, x1:x2].copy(), (x1, y1), inner_h


def _paste(
    canvas: np.ndarray,
    face: np.ndarray,
    lms: np.ndarray,
    origin: tuple[int, int],
    face_frac: float,
    face_h: float,
    placement: str,
    t: int,
    motion: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ch, cw = canvas.shape[:2]
    target = max(int(ch * face_frac), 8)
    scale = target / max(face_h, 1.0)
    fw = max(int(face.shape[1] * scale), 8)
    fh = max(int(face.shape[0] * scale), 8)
    resized = cv2.resize(face, (fw, fh), interpolation=cv2.INTER_LINEAR)
    jx = int(max(0.035 * fw, 4) * math.sin(t * 0.7)) if motion else 0
    jy = int(max(0.025 * fh, 3) * math.cos(t * 0.5)) if motion else 0
    if placement == "edge":
        x = 2 + jx
        y = max((ch - fh) // 2 + jy, 0)
    else:
        x = (cw - fw) // 2 + jx
        y = (ch - fh) // 2 + jy
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, cw), min(y + fh, ch)
    sx, sy = x1 - x, y1 - y
    out = canvas.copy()
    src = resized[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
    if src.size:
        out[y1:y2, x1:x2] = src
    ox, oy = origin
    mapped = lms.copy()
    mapped[:, 0] = (lms[:, 0] - oy) * scale + y
    mapped[:, 1] = (lms[:, 1] - ox) * scale + x
    return out, mapped


def write_png(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write {path}")


def _meta(spec: ScenarioSpec, frames: list[str]) -> dict:
    d = asdict(spec)
    d.pop("canvas")
    d.pop("face_frac")
    d.pop("placement")
    d.pop("n_frames")
    d.pop("motion")
    d.pop("source")
    d["width"] = spec.canvas[0]
    d["height"] = spec.canvas[1]
    d["frames"] = frames
    return d


def generate_seed_sequence(
    spec: ScenarioSpec,
    face: np.ndarray,
    lms: np.ndarray,
    origin: tuple[int, int],
    face_h: float,
    n_frames: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cw, ch = spec.canvas
    bg = np.full((ch, cw, 3), 42, np.uint8)
    bg += (np.arange(ch * cw * 3, dtype=np.uint8).reshape(ch, cw, 3) % 7)
    out = []
    for t in range(n_frames):
        frame, mapped = _paste(
            bg, face, lms, origin, spec.face_frac, face_h, spec.placement, t, spec.motion
        )
        if spec.glasses:
            frame = overlay_glasses(frame, mapped)
        out.append((frame, mapped))
    return out


def generate_scenarios(
    out_dir: Path,
    face: np.ndarray,
    lms: np.ndarray,
    origin: tuple[int, int],
    face_h: float,
    names: set[str] | None = None,
    n_frames_override: int | None = None,
    photos: list[Path] | None = None,
    specs: list[ScenarioSpec] | None = None,
) -> list[Path]:
    """Write scenario folders with PNG frames + meta.json. Returns dirs written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    specs = specs or list(REALISTIC)
    cached: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def want(spec: ScenarioSpec) -> bool:
        return names is None or spec.name in names

    for spec in specs:
        if spec.source != "seed" or not want(spec):
            continue
        n = n_frames_override or spec.n_frames
        if spec.name == "glasses_synth" and "glasses_synth_off" in cached:
            seq = [(overlay_glasses(f.copy(), m), m) for f, m in cached["glasses_synth_off"]]
        else:
            seq = generate_seed_sequence(spec, face, lms, origin, face_h, n)
        cached[spec.name] = seq
        dest = out_dir / spec.name
        dest.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, (bgr, _) in enumerate(seq):
            name = f"frame_{i:03d}.png"
            write_png(dest / name, bgr)
            frames.append(name)
        (dest / "meta.json").write_text(json.dumps(_meta(spec, frames), indent=2) + "\n")
        written.append(dest)

    if photos:
        want_photo = names is None or "glasses_photo" in names or any(
            n.startswith("glasses_photo") for n in names
        )
        if want_photo:
            for path in photos:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                h, w = bgr.shape[:2]
                scale = min(1280 / max(w, 1), 720 / max(h, 1), 1.0)
                if scale < 1.0:
                    bgr = cv2.resize(
                        bgr,
                        (max(int(w * scale), 8), max(int(h * scale), 8)),
                        interpolation=cv2.INTER_AREA,
                    )
                name = f"glasses_photo_{path.stem}"
                if names is not None and name not in names and "glasses_photo" not in names:
                    continue
                dest = out_dir / name
                dest.mkdir(parents=True, exist_ok=True)
                write_png(dest / "frame_000.png", bgr)
                spec = ScenarioSpec(
                    name=name,
                    tags=["glasses"],
                    canvas=(bgr.shape[1], bgr.shape[0]),
                    face_frac=1.0,
                    n_frames=1,
                    glasses=True,
                    motion=False,
                    source="photo",
                )
                (dest / "meta.json").write_text(
                    json.dumps(_meta(spec, ["frame_000.png"]), indent=2) + "\n"
                )
                written.append(dest)
    return written


def load_scenario(dir_path: Path) -> tuple[dict, list[np.ndarray]]:
    meta = json.loads((dir_path / "meta.json").read_text())
    frames = []
    for name in meta["frames"]:
        im = cv2.imread(str(dir_path / name), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cannot read {dir_path / name}")
        frames.append(im)
    return meta, frames


def list_scenario_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())
