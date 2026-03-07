from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class IFaceAligner(ABC):
    """Crops and aligns a detected face into a canonical patch."""

    @abstractmethod
    def align(self, frame: Any, face: Any) -> Tuple[Any, Any]:
        """Return (aligned_crop, inverse_affine_matrix) for the given face."""
        ...
