from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from morphflow_swap_engine.core.entities.detected_face import DetectedFace


@dataclass
class ReferenceFaceAnalysis:
    """Structured analysis of the reference image detection result."""

    image_width: int
    image_height: int
    raw_face_count: int
    filtered_face_count: int
    primary_face: DetectedFace | None = None
    primary_face_index: int | None = None
    primary_face_box: tuple[float, float, float, float] | None = None
    primary_confidence: float | None = None
    primary_face_size_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "raw_face_count": self.raw_face_count,
            "filtered_face_count": self.filtered_face_count,
            "primary_face_index": self.primary_face_index,
            "primary_face_box": list(self.primary_face_box) if self.primary_face_box else None,
            "primary_confidence": self.primary_confidence,
            "primary_face_size_ratio": self.primary_face_size_ratio,
            "warnings": list(self.warnings),
        }
