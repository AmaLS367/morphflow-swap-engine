from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ITemporalStabilizer(ABC):
    """Reduces flicker and identity drift across video frames."""

    @abstractmethod
    def stabilize(self, current_frame: Any, track_id: int) -> Any:
        """Return a temporally stabilized version of current_frame."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all frame history (call between unrelated clips)."""
        ...
