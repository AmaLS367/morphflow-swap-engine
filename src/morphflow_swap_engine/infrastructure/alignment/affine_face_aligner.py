from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from morphflow_swap_engine.core.contracts.i_face_aligner import IFaceAligner
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.value_objects.face_alignment_result import FaceAlignmentResult
from morphflow_swap_engine.core.value_objects.face_crop_plan import FaceCropPlan

ARCFACE_TEMPLATE = np.array([
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.50022500, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089]
], dtype=np.float32)

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
    """Aligns faces with affine warping driven by an explicit crop plan."""

    TEMPLATE_MAP = {
        "arcface_112": ARCFACE_TEMPLATE,
        "arcface_128": ARCFACE_128_TEMPLATE,
        "ffhq": FFHQ_TEMPLATE,
    }

    def align(
        self,
        frame: np.ndarray[Any, Any],
        face: DetectedFace,
        crop_plan: FaceCropPlan,
    ) -> FaceAlignmentResult:
        source_points = face.landmark_5

        if source_points.shape != (5, 2):
            raise ValueError("Face alignment requires 5-point landmarks.")

        target_points = self._target_points(crop_plan)
        affine_matrix = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.LMEDS,
        )[0]

        if affine_matrix is None:
            raise ValueError("Failed to estimate affine transform for face alignment.")

        face_width = float(face.bounding_box[2] - face.bounding_box[0])
        face_height = float(face.bounding_box[3] - face.bounding_box[1])
        face_extent = max(face_width, face_height, 1.0)
        scale_factor_estimate = float(crop_plan.output_size / face_extent)
        interpolation_flag = cv2.INTER_CUBIC if scale_factor_estimate > 1.0 else cv2.INTER_AREA
        interpolation_name = "cubic" if interpolation_flag == cv2.INTER_CUBIC else "area"

        crop = cv2.warpAffine(
            frame,
            affine_matrix,
            (crop_plan.output_size, crop_plan.output_size),
            borderMode=cv2.BORDER_REPLICATE,
            flags=interpolation_flag,
        )
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)

        return FaceAlignmentResult(
            crop=crop.astype(np.uint8, copy=False),
            affine_matrix=affine_matrix.astype(np.float32),
            inverse_affine_matrix=inverse_matrix.astype(np.float32),
            crop_plan=crop_plan,
            interpolation=interpolation_name,
            scale_factor_estimate=scale_factor_estimate,
        )

    def _target_points(self, crop_plan: FaceCropPlan) -> np.ndarray[Any, Any]:
        base_template = self.TEMPLATE_MAP.get(crop_plan.template_name, FFHQ_TEMPLATE)
        center = np.full((5, 2), 0.5, dtype=np.float32)
        scaled_template = center + ((base_template - center) * crop_plan.face_coverage_ratio)
        return scaled_template * crop_plan.output_size
