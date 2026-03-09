from __future__ import annotations

from typing import List

from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.value_objects.target_frame_analysis import TargetFrameAnalysis
from morphflow_swap_engine.core.value_objects.target_video_analysis import TargetVideoAnalysis

from .face_metrics import compute_face_metrics
from .primary_face_selector import PrimaryFaceSelector


class TargetVideoAnalyzer:
    """Analyzes sampled target frames before tracking takes over."""

    def __init__(
        self,
        selector: PrimaryFaceSelector,
        sample_count: int = 5,
        low_confidence_threshold: float = 0.55,
        min_primary_face_ratio: float = 0.005,
    ):
        self.selector = selector
        self.sample_count = sample_count
        self.low_confidence_threshold = low_confidence_threshold
        self.min_primary_face_ratio = min_primary_face_ratio

    def sample_frame_indices(self, frame_count: int) -> list[int]:
        if frame_count <= 0:
            return []

        sample_count = min(self.sample_count, frame_count)
        if sample_count <= 1:
            return [0]

        indices = {
            int(round(position * (frame_count - 1) / float(sample_count - 1)))
            for position in range(sample_count)
        }
        return sorted(indices)

    def analyze_frame(
        self,
        frame_index: int,
        frame_size: tuple[int, int],
        raw_faces: List[DetectedFace],
        filtered_faces: List[DetectedFace],
    ) -> TargetFrameAnalysis:
        frame_width, frame_height = frame_size
        warnings: list[str] = []

        if not raw_faces:
            warnings.append("target_no_faces_detected")
        elif not filtered_faces:
            warnings.append("target_all_faces_filtered")

        if len(filtered_faces) > 1:
            warnings.append("target_multiple_faces_detected")

        primary_face = self.selector.select(filtered_faces, frame_size)
        primary_face_box: tuple[float, float, float, float] | None = None
        primary_confidence: float | None = None
        primary_face_size_ratio: float | None = None

        if primary_face is None:
            warnings.append("target_primary_face_not_selected")
        else:
            primary_face_box = (
                float(primary_face.bounding_box[0]),
                float(primary_face.bounding_box[1]),
                float(primary_face.bounding_box[2]),
                float(primary_face.bounding_box[3]),
            )
            primary_confidence = float(primary_face.score)
            primary_face_size_ratio = compute_face_metrics(primary_face, frame_size).area_ratio

            if primary_confidence < self.low_confidence_threshold:
                warnings.append("target_primary_face_low_confidence")
            if primary_face_size_ratio < self.min_primary_face_ratio:
                warnings.append("target_primary_face_small")

        return TargetFrameAnalysis(
            frame_index=frame_index,
            frame_width=frame_width,
            frame_height=frame_height,
            raw_face_count=len(raw_faces),
            filtered_face_count=len(filtered_faces),
            primary_face_box=primary_face_box,
            primary_confidence=primary_confidence,
            primary_face_size_ratio=primary_face_size_ratio,
            warnings=warnings,
        )

    def summarize(self, frame_count: int, sampled_frames: List[TargetFrameAnalysis]) -> TargetVideoAnalysis:
        warnings: list[str] = []
        if not sampled_frames:
            warnings.append("target_no_sampled_frames_analyzed")

        no_face_frames = [frame.frame_index for frame in sampled_frames if frame.filtered_face_count == 0]
        if no_face_frames:
            warnings.append("target_sample_frames_without_faces")

        multiple_face_frames = [frame.frame_index for frame in sampled_frames if frame.filtered_face_count > 1]
        if multiple_face_frames:
            warnings.append("target_sample_frames_with_multiple_faces")

        primary_face_ratios = [
            frame.primary_face_size_ratio
            for frame in sampled_frames
            if frame.primary_face_size_ratio is not None
        ]
        average_primary_face_size_ratio = (
            sum(primary_face_ratios) / len(primary_face_ratios) if primary_face_ratios else None
        )

        for frame in sampled_frames:
            warnings.extend(frame.warnings)

        return TargetVideoAnalysis(
            frame_count=frame_count,
            sampled_frame_indices=[frame.frame_index for frame in sampled_frames],
            sampled_frames=sampled_frames,
            average_primary_face_size_ratio=average_primary_face_size_ratio,
            warnings=sorted(set(warnings)),
        )
