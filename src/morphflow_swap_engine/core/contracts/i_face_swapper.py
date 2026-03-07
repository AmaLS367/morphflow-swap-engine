from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IFaceSwapper(ABC):
    """Swaps the identity of an aligned face crop."""

    @abstractmethod
    def swap(self, source_embedding: Any, target_crop: Any) -> Any:
        """Return a swapped face crop of the same spatial size as target_crop."""
        ...

    @abstractmethod
    def swap_batch(self, source_embedding: Any, target_crops: Any) -> Any:
        """Return a batch of swapped face crops."""
        ...
