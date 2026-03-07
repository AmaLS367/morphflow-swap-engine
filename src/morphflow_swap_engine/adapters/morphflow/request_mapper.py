from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...core.entities.reference_face_asset import ReferenceFaceAsset
from ...core.entities.swap_request import SwapRequest
from ...core.entities.target_video_asset import TargetVideoAsset

import numpy as np


def map_request(payload: Dict[str, Any]) -> SwapRequest:
    """Convert a raw MorphFlow API payload dict into a SwapRequest.

    Expected payload keys:
        source_face_path (str): path to the reference face image/video
        target_path (str): path to the target video or image
        profile (str, optional): engine profile name, default "balanced"
        output_path (str, optional): desired output path
        label (str, optional): human-readable label for the reference face
    """
    source_path = Path(payload["source_face_path"])
    target_path = Path(payload["target_path"])

    reference = ReferenceFaceAsset(
        asset_path=source_path,
        embedding=np.empty(512, dtype=np.float32),
        label=payload.get("label", ""),
    )

    target = TargetVideoAsset(
        asset_path=target_path,
        fps=0.0,
        frame_count=0,
        width=0,
        height=0,
    )

    return SwapRequest(
        reference_faces=[reference],
        target_asset=target,
        profile_name=payload.get("profile", "balanced"),
        output_path=payload.get("output_path", ""),
    )
