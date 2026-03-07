from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from morphflow_swap_engine.core.entities.tracked_face_sequence import TrackedFaceSequence


class TrackScorer:
    """Scores face tracks to find the primary target face."""

    def __init__(
        self,
        weight_persistence: float = 1.0,
        weight_size: float = 1.0,
        weight_confidence: float = 0.5,
        weight_centrality: float = 0.5,
        weight_stability: float = 1.5,
    ):
        self.weight_persistence = weight_persistence
        self.weight_size = weight_size
        self.weight_confidence = weight_confidence
        self.weight_centrality = weight_centrality
        self.weight_stability = weight_stability

    def find_best_track(
        self,
        tracks: List[TrackedFaceSequence],
        frame_size: Optional[Tuple[int, int]] = None
    ) -> Optional[TrackedFaceSequence]:
        """Calculate scores for all tracks and return the one with the highest total score."""
        if not tracks:
            return None

        scored_tracks = []
        for track in tracks:
            if not track.faces:
                continue
            
            score = self.calculate_score(track, frame_size)
            scored_tracks.append((track, score))

        if not scored_tracks:
            return None

        # Return track with highest score
        return max(scored_tracks, key=lambda x: x[1])[0]

    def calculate_score(
        self,
        track: TrackedFaceSequence,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> float:
        """Calculate a composite score for a single track."""
        # 1. Persistence (track length relative to total potential frames)
        persistence = len(track.faces)
        stability = track.stability_score if track.stability_score > 0 else 0.0
        
        # 2. Average Size
        sizes = []
        for face in track.faces:
            bbox = face.bounding_box
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            sizes.append(width * height)
        avg_size = np.mean(sizes) if sizes else 0.0

        # 3. Average Confidence
        confidences = [face.score for face in track.faces]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # 4. Centrality (if frame_size provided)
        centrality = 0.0
        if frame_size:
            img_center = np.array([frame_size[0] / 2, frame_size[1] / 2])
            dist_from_center = []
            for face in track.faces:
                bbox = face.bounding_box
                face_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                dist = np.linalg.norm(face_center - img_center)
                # Normalize by diagonal
                max_dist = np.linalg.norm(img_center)
                dist_from_center.append(1.0 - (dist / max_dist))
            centrality = np.mean(dist_from_center) if dist_from_center else 0.0

        # Composite score
        # Note: We should ideally normalize these components to a similar range
        # For now, we use raw values with weights as a first pass
        total_score = (
            self.weight_persistence * persistence +
            self.weight_size * (avg_size / 10000.0) +  # rough normalization for pixels
            self.weight_confidence * avg_confidence +
            self.weight_centrality * centrality +
            self.weight_stability * stability
        )
        
        return float(total_score)
