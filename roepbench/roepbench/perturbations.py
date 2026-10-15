from typing import Any, Dict, Tuple

import numpy as np


ArrayDict = Dict[str, Any]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_occlusion_patch(
    image: np.ndarray,
    severity: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, int]:
    if image is None or not isinstance(image, np.ndarray) or image.ndim < 2:
        return image, 0

    sev = _clamp01(severity)
    if sev <= 0.0:
        return image, 0

    out = image.copy()
    h, w = out.shape[0], out.shape[1]
    patch = max(1, int(min(h, w) * sev))
    patch_h = min(h, patch)
    patch_w = min(w, patch)

    y0 = int(rng.randint(0, h - patch_h + 1))
    x0 = int(rng.randint(0, w - patch_w + 1))

    out[y0 : y0 + patch_h, x0 : x0 + patch_w, ...] = 0
    return out, int(patch_h * patch_w)


def _apply_depth_dropout(
    depth: np.ndarray,
    drop_prob: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, int]:
    if depth is None or not isinstance(depth, np.ndarray) or depth.ndim < 2:
        return depth, 0

    p = _clamp01(drop_prob)
    if p <= 0.0:
        return depth, 0

    out = depth.copy()
    h, w = out.shape[0], out.shape[1]
    mask = rng.rand(h, w) < p
    out[mask, ...] = 0
    return out, int(mask.sum())


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(max(1e-6, sigma))
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.array([1.0], dtype=np.float32)
    return (kernel / kernel_sum).astype(np.float32)


def _convolve_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[axis] = (pad, pad)
    padded = np.pad(arr, pad_spec, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for k, weight in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(k, k + arr.shape[axis])
        out += float(weight) * padded[tuple(sl)]
    return out


def _apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.ndim < 2:
        return image
    if float(sigma) <= 0.0:
        return image

    kernel = _gaussian_kernel1d(float(sigma))
    arr = image.astype(np.float32, copy=False)
    arr = _convolve_axis(arr, kernel, axis=0)
    arr = _convolve_axis(arr, kernel, axis=1)
    if image.dtype == np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(image.dtype)
    return arr


def apply_perturbations(
    obs_dict: ArrayDict,
    occlusion_severity: float,
    depth_dropout_prob: float,
    gaussian_blur_sigma: float,
    obs_blackout_active: bool,
    rng: np.random.RandomState,
) -> Tuple[ArrayDict, Dict[str, int]]:
    """Apply perturbations only to RGB/Depth numpy arrays in an observation dict."""
    out: ArrayDict = {}
    stats = {
        "occluded_pixels": 0,
        "dropped_depth_pixels": 0,
        "blackout_pixels": 0,
    }

    for key, value in obs_dict.items():
        if key.endswith("_rgb") and isinstance(value, np.ndarray):
            updated, count = _apply_occlusion_patch(value, occlusion_severity, rng)
            updated = _apply_gaussian_blur(updated, gaussian_blur_sigma)
            if bool(obs_blackout_active):
                blackout = np.zeros_like(updated)
                stats["blackout_pixels"] += int(np.prod(updated.shape[:2]))
                updated = blackout
            out[key] = updated
            stats["occluded_pixels"] += count
            continue

        if key.endswith("_depth") and isinstance(value, np.ndarray):
            updated, count = _apply_depth_dropout(value, depth_dropout_prob, rng)
            if bool(obs_blackout_active):
                blackout = np.zeros_like(updated)
                stats["blackout_pixels"] += int(np.prod(updated.shape[:2]))
                updated = blackout
            out[key] = updated
            stats["dropped_depth_pixels"] += count
            continue

        out[key] = value

    return out, stats
