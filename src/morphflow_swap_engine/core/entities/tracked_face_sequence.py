from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from .detected_face import DetectedFace


@dataclass
class TrackedFaceSequence:
    """An identity-consistent sequence of DetectedFace instances across frames."""

    track_id: int
    faces: List[DetectedFace] = field(default_factory=list)
    missed_frames: int = 0
    total_missed_frames: int = 0
    stability_score: float = 0.0
    last_seen_frame: int = -1
    is_active: bool = True
    average_confidence: float = 0.0
    average_face_area_ratio: float = 0.0
    average_centrality: float = 0.0
    embedding_centroid: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    @property
    def first_frame_index(self) -> int:
        if not self.faces:
            return -1
        return self.faces[0].frame_index

    @property
    def frame_count(self) -> int:
        return len(self.faces)

    def recompute_aggregates(self) -> None:
        if not self.faces:
            self.average_confidence = 0.0
            self.average_face_area_ratio = 0.0
            self.average_centrality = 0.0
            self.embedding_centroid = np.empty(0, dtype=np.float32)
            self.stability_score = 0.0
            return

        self.average_confidence = float(np.mean([face.score for face in self.faces]))
        self.average_face_area_ratio = float(np.mean([face.face_area_ratio for face in self.faces]))
        self.average_centrality = float(np.mean([face.centrality for face in self.faces]))

        embeddings = [face.embedding for face in self.faces if face.embedding.size > 0]
        if embeddings:
            centroid = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(centroid))
            self.embedding_centroid = centroid / norm if norm > 0 else centroid
        else:
            self.embedding_centroid = np.empty(0, dtype=np.float32)

        total_frames = self.frame_count + self.total_missed_frames
        self.stability_score = float(self.frame_count / total_frames) if total_frames > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_count": self.frame_count,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_seen_frame,
            "missed_frames": self.missed_frames,
            "total_missed_frames": self.total_missed_frames,
            "stability_score": self.stability_score,
            "average_confidence": self.average_confidence,
            "average_face_area_ratio": self.average_face_area_ratio,
            "average_centrality": self.average_centrality,
            "is_active": self.is_active,
        }
