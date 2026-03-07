from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectedFace:
    """A face detected in a single frame."""

    # [x1, y1, x2, y2]
    bounding_box: NDArray[np.float32]
    # (5, 2) keypoints
    landmark_5: NDArray[np.float32]
    score: float
    frame_index: int = -1
    track_id: int = -1
    embedding: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
