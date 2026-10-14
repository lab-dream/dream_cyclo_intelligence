#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Image shape helpers for LeRobot inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


IMAGE_KEY_PREFIX = "observation.images."


def infer_image_preprocessing(model_path: str, policy_type: str) -> dict[str, str]:
    """Read the transferred Diffusion training image recipe, nearest file first."""
    if policy_type != "diffusion":
        return {}
    root = Path(model_path)
    for directory in (root, *root.parents):
        path = directory / "image_preprocessing.json"
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text())
        if metadata.get("resize_mode") == "native":
            return {"resize_mode": "native"}
        recipe = str(metadata.get("image_recipe", "")).lower()
        if not all(term in recipe for term in (
            "bilinear", "antialias", "aspect-preserving", "zero padding"
        )):
            raise ValueError(f"Unsupported Diffusion image_recipe in {path}: {recipe!r}")
        alignment = metadata.get("padding_alignment", "center")
        if alignment not in {"center", "top_left", "bottom_right"}:
            raise ValueError(f"Unsupported padding_alignment in {path}: {alignment!r}")
        return {"resize_mode": "letterbox", "padding_alignment": alignment}
    return {}


def infer_image_resize_targets(
    input_features: Mapping[str, Any],
) -> Dict[str, Tuple[int, int]]:
    """Return per-policy-key resize targets as ``(width, height)``."""
    targets: Dict[str, Tuple[int, int]] = {}
    for key, feature in (input_features or {}).items():
        if not str(key).startswith(IMAGE_KEY_PREFIX):
            continue
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, Mapping):
            shape = feature.get("shape")
        if shape and len(shape) == 3:
            _channels, height, width = shape
            targets[str(key)] = (int(width), int(height))
    return targets


def apply_rotation(image: np.ndarray, rotation_deg: int | float | None) -> np.ndarray:
    """Apply camera rotation metadata using the same direction as OpenCV."""
    rotation = int(rotation_deg or 0) % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return np.rot90(image, k=3)
    if rotation == 180:
        return np.rot90(image, k=2)
    if rotation == 270:
        return np.rot90(image, k=1)
    raise ValueError(f"unsupported camera rotation_deg={rotation_deg}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    *,
    resize_mode: str = "stretch",
    padding_alignment: str = "center",
) -> np.ndarray:
    """Resize to (width, height), optionally preserving aspect ratio and padding."""
    if resize_mode == "native":
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Native policy requires HWC uint8 RGB images")
        if not target_size or image.shape[:2] != (target_size[1], target_size[0]):
            raise ValueError(
                f"Native image shape mismatch: got {image.shape}; expected "
                f"HWC=({target_size[1] if target_size else '?'}, "
                f"{target_size[0] if target_size else '?'}, 3). "
                "Set the camera stream to the training resolution; resizing is disabled."
            )
        return image
    if not target_size:
        return image
    width, height = target_size
    if resize_mode == "letterbox" and image.dtype != np.uint8:
        raise ValueError("Training letterbox requires uint8 RGB images before resize and padding")
    if image.shape[1] == width and image.shape[0] == height:
        return image

    if resize_mode == "letterbox":
        import torch
        from torchvision.transforms.v2 import functional as VF

        source_h, source_w = image.shape[:2]
        scale = min(height / source_h, width / source_w)
        resized_h = max(1, min(height, round(source_h * scale)))
        resized_w = max(1, min(width, round(source_w * scale)))
        # cached_images.py stores resized uint8 frames during training. Use
        # the same torchvision v2 operation before float conversion/scaling.
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = VF.resize(tensor, [resized_h, resized_w], antialias=True)
        extra_h, extra_w = height - resized_h, width - resized_w
        if padding_alignment == "center":
            top, left = extra_h // 2, extra_w // 2
        elif padding_alignment == "top_left":
            top, left = extra_h, extra_w
        elif padding_alignment == "bottom_right":
            top, left = 0, 0
        else:
            raise ValueError(f"Unsupported padding_alignment: {padding_alignment!r}")
        tensor = VF.pad(tensor, [left, top, extra_w - left, extra_h - top])
        return tensor.permute(1, 2, 0).numpy()
    if resize_mode != "stretch":
        raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")

    import cv2

    return cv2.resize(image, (width, height))


def prepare_policy_image(
    image: np.ndarray,
    *,
    rotation_deg: int | float | None = 0,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "stretch",
    padding_alignment: str = "center",
) -> np.ndarray:
    """Rotate first, then resize to the policy feature's expected shape."""
    return resize_image(apply_rotation(image, rotation_deg), target_size,
                        resize_mode=resize_mode, padding_alignment=padding_alignment)
