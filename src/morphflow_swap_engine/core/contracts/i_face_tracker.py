from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class IFaceTracker(ABC):
    """Maintains identity continuity of faces across video frames."""

    @abstractmethod
    def update(self, detections: List[Any], frame_index: int) -> List[Any]:
        """Assign track IDs to detections and return updated TrackedFaceSequences."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all active tracks (call between unrelated clips)."""
        ...
