from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from morphflow_swap_engine.core.entities.detected_face import DetectedFace


class IFaceDetector(ABC):
    """Detects faces in a single image frame."""

    @abstractmethod
    def detect(self, frame: Any, score_threshold: float = 0.5) -> List[DetectedFace]:
        """Return a list of DetectedFace for all faces found in frame."""
        ...
