from __future__ import annotations

from typing import List, Optional, Tuple

from morphflow_swap_engine.core.entities.tracked_face_sequence import TrackedFaceSequence


class TrackScorer:
    """Scores face tracks to find the primary target face."""

    def __init__(
        self,
        weight_persistence: float = 0.30,
        weight_size: float = 0.20,
        weight_confidence: float = 0.20,
        weight_centrality: float = 0.10,
        weight_stability: float = 0.20,
    ):
        self.weight_persistence = weight_persistence
        self.weight_size = weight_size
        self.weight_confidence = weight_confidence
        self.weight_centrality = weight_centrality
        self.weight_stability = weight_stability

    def find_best_track(
        self,
        tracks: List[TrackedFaceSequence],
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[TrackedFaceSequence]:
        """Calculate scores for all tracks and return the one with the highest total score."""
        del frame_size
        if not tracks:
            return None

        scored_tracks: list[tuple[TrackedFaceSequence, float]] = []
        candidate_tracks = [track for track in tracks if track.faces]
        if not candidate_tracks:
            return None

        max_persistence = max(track.frame_count for track in candidate_tracks)
        max_size_ratio = max(track.average_face_area_ratio for track in candidate_tracks)

        for track in candidate_tracks:
            score = self.calculate_score(
                track,
                normalizers=(float(max_persistence), float(max_size_ratio)),
            )
            scored_tracks.append((track, score))

        if not scored_tracks:
            return None

        # Return track with highest score
        return max(scored_tracks, key=lambda x: x[1])[0]

    def calculate_score(
        self,
        track: TrackedFaceSequence,
        frame_size: Optional[Tuple[int, int]] = None,
        normalizers: tuple[float, float] | None = None,
    ) -> float:
        """Calculate a composite score for a single track."""
        del frame_size
        if normalizers is None:
            max_persistence = float(max(track.frame_count, 1))
            max_size_ratio = max(track.average_face_area_ratio, 1e-6)
        else:
            max_persistence, max_size_ratio = normalizers

        persistence = track.frame_count / max(max_persistence, 1.0)
        size = track.average_face_area_ratio / max(max_size_ratio, 1e-6)
        confidence = track.average_confidence
        centrality = track.average_centrality
        stability = track.stability_score

        total_score = (
            self.weight_persistence * persistence +
            self.weight_size * size +
            self.weight_confidence * confidence +
            self.weight_centrality * centrality +
            self.weight_stability * stability
        )

        return float(total_score)
