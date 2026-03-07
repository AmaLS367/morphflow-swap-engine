from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .reference_face_asset import ReferenceFaceAsset
from .target_video_asset import TargetVideoAsset


@dataclass
class SwapRequest:
    """Input specification for a face swap job."""

    reference_faces: List[ReferenceFaceAsset]
    target_asset: TargetVideoAsset
    profile_name: str = "balanced"
    output_path: Path = field(default_factory=Path)
