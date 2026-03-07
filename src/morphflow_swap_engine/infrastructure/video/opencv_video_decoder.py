from __future__ import annotations

import os
from typing import Any, Generator

import cv2

from morphflow_swap_engine.core.contracts.i_video_decoder import IVideoDecoder
from morphflow_swap_engine.core.entities.target_video_asset import TargetVideoAsset


class OpenCVVideoDecoder(IVideoDecoder):
    """Basic video decoder using OpenCV."""

    def probe(self, asset: TargetVideoAsset) -> TargetVideoAsset:
        if not os.path.exists(asset.asset_path):
            raise FileNotFoundError(f"Video not found: {asset.asset_path}")

        cap = cv2.VideoCapture(asset.asset_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {asset.asset_path}")

        asset.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        asset.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asset.fps = float(cap.get(cv2.CAP_PROP_FPS))
        asset.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        return asset

    def frames(self, asset: TargetVideoAsset) -> Generator[Any, None, None]:
        cap = cv2.VideoCapture(asset.asset_path)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

        cap.release()
