from __future__ import annotations

from typing import List

from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.value_objects.reference_face_analysis import ReferenceFaceAnalysis

from .face_metrics import compute_face_metrics
from .primary_face_selector import PrimaryFaceSelector


class ReferenceFaceAnalyzer:
    """Builds a structured analysis record for the reference image."""

    def __init__(
        self,
        selector: PrimaryFaceSelector,
        low_confidence_threshold: float = 0.55,
        min_primary_face_ratio: float = 0.015,
    ):
        self.selector = selector
        self.low_confidence_threshold = low_confidence_threshold
        self.min_primary_face_ratio = min_primary_face_ratio

    def analyze(
        self,
        frame_size: tuple[int, int],
        raw_faces: List[DetectedFace],
        filtered_faces: List[DetectedFace],
    ) -> ReferenceFaceAnalysis:
        frame_width, frame_height = frame_size
        warnings: list[str] = []

        if not raw_faces:
            warnings.append("reference_no_faces_detected")
        elif not filtered_faces:
            warnings.append("reference_all_faces_filtered")

        if len(filtered_faces) > 1:
            warnings.append("reference_multiple_faces_detected")

        primary_face = self.selector.select(filtered_faces, frame_size)
        primary_face_index: int | None = None
        primary_face_box: tuple[float, float, float, float] | None = None
        primary_confidence: float | None = None
        primary_face_size_ratio: float | None = None

        if primary_face is None:
            warnings.append("reference_primary_face_not_selected")
        else:
            primary_face_index = next(
                (index for index, face in enumerate(filtered_faces) if face is primary_face),
                None,
            )
            primary_face_box = (
                float(primary_face.bounding_box[0]),
                float(primary_face.bounding_box[1]),
                float(primary_face.bounding_box[2]),
                float(primary_face.bounding_box[3]),
            )
            primary_confidence = float(primary_face.score)
            primary_face_size_ratio = compute_face_metrics(primary_face, frame_size).area_ratio

            if primary_confidence < self.low_confidence_threshold:
                warnings.append("reference_primary_face_low_confidence")
            if primary_face_size_ratio < self.min_primary_face_ratio:
                warnings.append("reference_primary_face_small")

        return ReferenceFaceAnalysis(
            image_width=frame_width,
            image_height=frame_height,
            raw_face_count=len(raw_faces),
            filtered_face_count=len(filtered_faces),
            primary_face=primary_face,
            primary_face_index=primary_face_index,
            primary_face_box=primary_face_box,
            primary_confidence=primary_confidence,
            primary_face_size_ratio=primary_face_size_ratio,
            warnings=warnings,
        )
