from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_color_transfer(target_crop: np.ndarray[Any, Any], source_crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Transfers color characteristics from target_crop to source_crop using Lab color space.
    This ensures the swapped face matches the lighting/color of the target video.
    """
    bgr_to_lab = getattr(cv2, "COLOR_BGR2LAB", getattr(cv2, "COLOR_BGR2Lab"))
    lab_to_bgr = getattr(cv2, "COLOR_Lab2BGR")
    source_lab = cv2.cvtColor(source_crop, bgr_to_lab)
    target_lab = cv2.cvtColor(target_crop, bgr_to_lab)

    source_mean, source_std = _get_mean_and_std(source_lab)
    target_mean, target_std = _get_mean_and_std(target_lab)

    # Avoid division by zero
    source_std = np.clip(source_std, 0.001, None)
    
    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result_lab, lab_to_bgr)


def _get_mean_and_std(image: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    mean, std = cv2.meanStdDev(image)
    return mean.flatten(), std.flatten()
