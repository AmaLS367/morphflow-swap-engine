from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .target_frame_analysis import TargetFrameAnalysis


@dataclass
class TargetVideoAnalysis:
    """Aggregate analysis over sampled target video frames."""

    frame_count: int
    sampled_frame_indices: list[int]
    sampled_frames: list[TargetFrameAnalysis] = field(default_factory=list)
    average_primary_face_size_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "sampled_frame_count": len(self.sampled_frames),
            "average_primary_face_size_ratio": self.average_primary_face_size_ratio,
            "warnings": list(self.warnings),
            "sampled_frames": [frame.to_dict() for frame in self.sampled_frames],
        }
