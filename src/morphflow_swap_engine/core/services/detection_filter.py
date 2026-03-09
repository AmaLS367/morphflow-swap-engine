from __future__ import annotations

from typing import List

from morphflow_swap_engine.core.entities.detected_face import DetectedFace

from .face_metrics import compute_face_metrics


class FaceDetectionFilter:
    """Removes detections that are too weak to be useful downstream."""

    def __init__(
        self,
        min_face_size: int = 8,
        min_face_ratio: float = 0.0002,
        min_centrality: float = 0.0,
    ):
        self.min_face_size = min_face_size
        self.min_face_ratio = min_face_ratio
        self.min_centrality = min_centrality

    def filter_faces(
        self,
        faces: List[DetectedFace],
        frame_size: tuple[int, int],
    ) -> List[DetectedFace]:
        filtered: list[DetectedFace] = []
        for face in faces:
            metrics = compute_face_metrics(face, frame_size)
            face.face_area_ratio = metrics.area_ratio
            face.centrality = metrics.centrality
            if metrics.width < self.min_face_size or metrics.height < self.min_face_size:
                continue
            if metrics.area_ratio < self.min_face_ratio:
                continue
            if metrics.centrality < self.min_centrality:
                continue
            filtered.append(face)
        return filtered
