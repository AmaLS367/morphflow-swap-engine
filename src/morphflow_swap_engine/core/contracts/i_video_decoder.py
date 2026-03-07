from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

from ..entities.target_video_asset import TargetVideoAsset


class IVideoDecoder(ABC):
    """Decodes a video file into raw frames."""

    @abstractmethod
    def probe(self, asset: TargetVideoAsset) -> TargetVideoAsset:
        """Return a fully populated TargetVideoAsset (fps, frame_count, etc.)."""
        ...

    @abstractmethod
    def frames(self, asset: TargetVideoAsset) -> Generator[Any, None, None]:
        """Yield decoded frames (numpy uint8 arrays) in presentation order."""
        ...
