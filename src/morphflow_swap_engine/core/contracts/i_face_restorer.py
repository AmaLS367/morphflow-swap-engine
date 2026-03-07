from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IFaceRestorer(ABC):
    """Enhances and restores a swapped face crop (optional stage)."""

    @abstractmethod
    def restore(self, swapped_crop: Any) -> Any:
        """Return a restored/enhanced version of swapped_crop."""
        ...
