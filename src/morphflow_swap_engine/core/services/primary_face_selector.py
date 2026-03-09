from __future__ import annotations

from typing import List

from morphflow_swap_engine.core.entities.detected_face import DetectedFace

from .face_metrics import compute_face_metrics


class PrimaryFaceSelector:
    """Selects the best candidate using size, confidence, and centrality."""

    def __init__(
        self,
        size_weight: float = 0.5,
        confidence_weight: float = 0.3,
        centrality_weight: float = 0.2,
    ):
        self.size_weight = size_weight
        self.confidence_weight = confidence_weight
        self.centrality_weight = centrality_weight

    def select(self, faces: List[DetectedFace], frame_size: tuple[int, int]) -> DetectedFace | None:
        if not faces:
            return None
        return max(faces, key=lambda face: self.score_face(face, frame_size))

    def score_face(self, face: DetectedFace, frame_size: tuple[int, int]) -> float:
        metrics = compute_face_metrics(face, frame_size)
        return (
            self.size_weight * metrics.normalized_size
            + self.confidence_weight * float(face.score)
            + self.centrality_weight * metrics.centrality
        )
