from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .detected_face import DetectedFace


@dataclass
class TrackedFaceSequence:
    """An identity-consistent sequence of DetectedFace instances across frames."""

    track_id: int
    faces: List[DetectedFace] = field(default_factory=list)
    missed_frames: int = 0
    stability_score: float = 0.0
    last_seen_frame: int = -1
    is_active: bool = True
