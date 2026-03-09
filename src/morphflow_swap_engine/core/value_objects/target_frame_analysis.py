from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetFrameAnalysis:
    """Summary of detections for one sampled target frame."""

    frame_index: int
    frame_width: int
    frame_height: int
    raw_face_count: int
    filtered_face_count: int
    primary_face_box: tuple[float, float, float, float] | None = None
    primary_confidence: float | None = None
    primary_face_size_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "raw_face_count": self.raw_face_count,
            "filtered_face_count": self.filtered_face_count,
            "primary_face_box": list(self.primary_face_box) if self.primary_face_box else None,
            "primary_confidence": self.primary_confidence,
            "primary_face_size_ratio": self.primary_face_size_ratio,
            "warnings": list(self.warnings),
        }
