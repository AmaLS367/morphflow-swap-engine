from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2

from morphflow_swap_engine.core.contracts.i_video_encoder import IVideoEncoder
from morphflow_swap_engine.core.entities.target_video_asset import TargetVideoAsset


class OpenCVVideoEncoder(IVideoEncoder):
    """Basic video encoder using OpenCV."""

    def encode(
        self,
        frames: Iterable[Any],
        source_asset: TargetVideoAsset,
        output_path: Path,
    ) -> Path:
        """Write frames to output_path and return it."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use mp4v codec by default for generic MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # We need width and height. If not present in source_asset, we take it from the first frame.
        writer = None
        
        for frame in frames:
            if writer is None:
                height, width = frame.shape[:2]
                fps = source_asset.fps if source_asset.fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
            writer.write(frame)
            
        if writer is not None:
            writer.release()
            
        return output_path
