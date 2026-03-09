from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from morphflow_swap_engine.core.entities.detected_face import DetectedFace


class IFaceDetector(ABC):
    """Detects faces in a single image frame."""

    @abstractmethod
    def detect(self, frame: Any, score_threshold: float = 0.5) -> List[DetectedFace]:
        """Return a list of DetectedFace for all faces found in frame."""
        ...

    def detect_batch(self, frames: Sequence[Any], score_threshold: float = 0.5) -> List[List[DetectedFace]]:
        """Return one detection list per frame using a default sequential fallback."""
        return [self.detect(frame, score_threshold=score_threshold) for frame in frames]
