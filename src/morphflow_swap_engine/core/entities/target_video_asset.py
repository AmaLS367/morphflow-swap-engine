from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TargetVideoAsset:
    """Describes the video (or image) to be processed."""

    asset_path: Path
    fps: float
    frame_count: int
    width: int
    height: int
