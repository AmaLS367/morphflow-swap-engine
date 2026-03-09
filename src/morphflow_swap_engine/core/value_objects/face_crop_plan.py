from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FaceCropPlan:
    """Configuration and derived metadata for an aligned face crop."""

    purpose: str
    output_size: int
    template_name: str
    configured_margin_ratio: float
    effective_margin_ratio: float
    face_coverage_ratio: float
    frame_width: int
    frame_height: int
    face_area_ratio: float
    track_id: int | None = None
    small_face_mode: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "output_size": self.output_size,
            "template_name": self.template_name,
            "configured_margin_ratio": self.configured_margin_ratio,
            "effective_margin_ratio": self.effective_margin_ratio,
            "face_coverage_ratio": self.face_coverage_ratio,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "face_area_ratio": self.face_area_ratio,
            "track_id": self.track_id,
            "small_face_mode": self.small_face_mode,
            "notes": list(self.notes),
        }
