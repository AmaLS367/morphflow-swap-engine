from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..entities.detected_face import DetectedFace
from ..entities.tracked_face_sequence import TrackedFaceSequence


class IFaceTracker(ABC):
    """Maintains identity continuity of faces across video frames."""

    @abstractmethod
    def update(self, detections: List[DetectedFace], frame_index: int) -> List[TrackedFaceSequence]:
        """Assign track IDs to detections and return updated TrackedFaceSequences."""
        ...

    @abstractmethod
    def get_tracks(self, include_inactive: bool = True) -> List[TrackedFaceSequence]:
        """Return tracked sequences collected so far."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all active tracks (call between unrelated clips)."""
        ...
