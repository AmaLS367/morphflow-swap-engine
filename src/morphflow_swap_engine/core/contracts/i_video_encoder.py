from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from ..entities.target_video_asset import TargetVideoAsset


class IVideoEncoder(ABC):
    """Encodes processed frames into an output video file."""

    @abstractmethod
    def encode(
        self,
        frames: Iterable[Any],
        source_asset: TargetVideoAsset,
        output_path: Path,
    ) -> Path:
        """Write frames to output_path and return it."""
        ...
