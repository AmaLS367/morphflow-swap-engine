from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IFaceAligner(ABC):
    """Crops and aligns a detected face into a canonical patch."""

    @abstractmethod
    def align(self, frame: Any, face: Any, crop_plan: Any) -> Any:
        """Return a deterministic aligned crop plus transform metadata."""
        ...
