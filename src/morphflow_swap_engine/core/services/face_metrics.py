from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from morphflow_swap_engine.core.entities.detected_face import DetectedFace


@dataclass(frozen=True)
class FaceMetrics:
    width: float
    height: float
    area_ratio: float
    centrality: float
    normalized_size: float


def compute_face_metrics(face: DetectedFace, frame_size: tuple[int, int]) -> FaceMetrics:
    """Compute reusable geometry metrics for a detection."""
    frame_width, frame_height = frame_size
    frame_area = max(frame_width * frame_height, 1)

    bbox = face.bounding_box
    width = max(float(bbox[2] - bbox[0]), 0.0)
    height = max(float(bbox[3] - bbox[1]), 0.0)
    area_ratio = (width * height) / float(frame_area)

    img_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
    face_center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
    max_distance = max(float(np.linalg.norm(img_center)), 1e-6)
    distance = float(np.linalg.norm(face_center - img_center))
    centrality = max(0.0, 1.0 - (distance / max_distance))

    return FaceMetrics(
        width=width,
        height=height,
        area_ratio=area_ratio,
        centrality=centrality,
        normalized_size=min(1.0, sqrt(area_ratio)),
    )
