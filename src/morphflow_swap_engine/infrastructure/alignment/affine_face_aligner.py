from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from morphflow_swap_engine.core.contracts.i_face_aligner import IFaceAligner
from morphflow_swap_engine.core.entities.detected_face import DetectedFace

# Standard ArcFace normalized template (usually used for 112x112 or 512x512 with InsightFace)
ARCFACE_TEMPLATE = np.array([
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.50022500, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089]
], dtype=np.float32)

# FFHQ 512 template (often used for high res swappers like Ghost)
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465]
], dtype=np.float32)

# ArcFace 128 template (used for inswapper_128)
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453]
], dtype=np.float32)


class AffineFaceAligner(IFaceAligner):
    """Aligns faces using affine transformation based on 5 landmarks."""

    def __init__(self, crop_size: int = 512, template: str = "ffhq"):
        self.crop_size = crop_size
        if template == "arcface_112":
            self.base_template = ARCFACE_TEMPLATE
        elif template == "arcface_128":
            self.base_template = ARCFACE_128_TEMPLATE
        elif template == "ffhq":
            self.base_template = FFHQ_TEMPLATE
        else:
            self.base_template = ARCFACE_TEMPLATE
            
        self.target_points = self.base_template * self.crop_size

    def align(self, frame: np.ndarray[Any, Any], face: DetectedFace) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Align and crop the face from the frame.
        
        Returns:
            Tuple containing the cropped face image and the inverse affine matrix.
        """
        source_points = face.landmark_5
        
        # Estimate the affine transformation matrix
        affine_matrix = cv2.estimateAffinePartial2D(
            source_points, 
            self.target_points, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=100
        )[0]
        
        # Warp the image to get the cropped face
        crop = cv2.warpAffine(
            frame, 
            affine_matrix, 
            (self.crop_size, self.crop_size),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA
        )
        
        # Calculate the inverse matrix for pasting back later
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        
        return crop, inverse_matrix
