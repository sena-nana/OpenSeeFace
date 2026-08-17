"""BGR-source ImageNet / RetinaFace normalization. Affine is baked into u8 LUTs."""
from __future__ import annotations

import numpy as np
import cv2

IMAGENET_BIAS = np.float32([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
IMAGENET_SCALE = 1.0 / (np.float32([0.229, 0.224, 0.225]) * np.float32(255.0))
RETINA_MEAN_BGR = np.float32([104.0, 117.0, 123.0])

_U8 = np.arange(256, dtype=np.float32)
IMAGENET_LUT = np.stack([_U8 * IMAGENET_SCALE[c] + IMAGENET_BIAS[c] for c in range(3)])
RETINA_LUT = np.stack([_U8 - RETINA_MEAN_BGR[c] for c in range(3)])


def resize_bgr(bgr: np.ndarray, size) -> np.ndarray:
    dw, dh = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if bgr.shape[1] == dw and bgr.shape[0] == dh:
        return bgr
    return cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)


def _nchw(bgr: np.ndarray, lut: np.ndarray, swap_rb: bool) -> np.ndarray:
    if swap_rb:
        planes = (lut[0][bgr[:, :, 2]], lut[1][bgr[:, :, 1]], lut[2][bgr[:, :, 0]])
    else:
        planes = (lut[0][bgr[:, :, 0]], lut[1][bgr[:, :, 1]], lut[2][bgr[:, :, 2]])
    return np.stack(planes, 0)[None]


def imagenet_hwc(bgr: np.ndarray) -> np.ndarray:
    return np.stack(
        (IMAGENET_LUT[0][bgr[:, :, 2]], IMAGENET_LUT[1][bgr[:, :, 1]], IMAGENET_LUT[2][bgr[:, :, 0]]),
        axis=-1,
    )


def imagenet_nchw(bgr: np.ndarray, size: int) -> np.ndarray:
    return _nchw(resize_bgr(bgr, size), IMAGENET_LUT, True)


def imagenet_nchw_rgb_float(rgb: np.ndarray) -> np.ndarray:
    return np.transpose(rgb * IMAGENET_SCALE + IMAGENET_BIAS, (2, 0, 1))[None]


def retina_nchw(bgr: np.ndarray, size=640) -> np.ndarray:
    return _nchw(resize_bgr(bgr, size), RETINA_LUT, False)
