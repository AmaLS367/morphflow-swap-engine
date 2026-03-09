from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from morphflow_swap_engine.core.contracts.i_face_tracker import IFaceTracker
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.entities.tracked_face_sequence import TrackedFaceSequence


class IOUFaceTracker(IFaceTracker):
    """Hybrid IOU + embedding tracker for short-gap target face continuity."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 5,
        embedding_similarity_threshold: float = 0.35,
        reid_window_frames: int = 8,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.reid_window_frames = reid_window_frames
        self.tracks: Dict[int, TrackedFaceSequence] = {}
        self.next_track_id = 1

    def update(self, detections: List[DetectedFace], frame_index: int) -> List[TrackedFaceSequence]:
        """Match detections with tracks using IOU first, then short-gap embedding re-id."""
        if not self.tracks:
            for det in detections:
                self._create_track(det, frame_index)
            return list(self.tracks.values())

        match_candidates: list[tuple[int, float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                if not track.faces:
                    continue

                frame_gap = frame_index - track.last_seen_frame
                if frame_gap <= 0 or frame_gap > self.reid_window_frames:
                    continue

                iou = self._calculate_iou(det.bounding_box, track.faces[-1].bounding_box)
                if track.is_active and iou >= self.iou_threshold:
                    match_candidates.append((0, iou, det_idx, track_id))
                    continue

                similarity = self._embedding_similarity(det, track)
                if similarity >= self.embedding_similarity_threshold:
                    match_candidates.append((1, similarity, det_idx, track_id))

        match_candidates.sort(key=lambda item: (item[0], -item[1]))
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        final_matches: list[tuple[int, int]] = []

        for _priority, _score, det_idx, track_id in match_candidates:
            if det_idx in used_detections or track_id in used_tracks:
                continue
            final_matches.append((det_idx, track_id))
            used_detections.add(det_idx)
            used_tracks.add(track_id)

        for det_idx, track_id in final_matches:
            self._update_track(self.tracks[track_id], detections[det_idx], frame_index)

        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self._create_track(det, frame_index)

        for track_id, track in self.tracks.items():
            if track_id in used_tracks or track.last_seen_frame >= frame_index:
                continue
            frame_gap = frame_index - track.last_seen_frame
            if frame_gap > track.missed_frames:
                track.total_missed_frames += frame_gap - track.missed_frames
            track.missed_frames = frame_gap
            if track.missed_frames > self.max_lost_frames:
                track.is_active = False
            track.recompute_aggregates()

        return list(self.tracks.values())

    def get_tracks(self, include_inactive: bool = True) -> List[TrackedFaceSequence]:
        if include_inactive:
            return list(self.tracks.values())
        return [track for track in self.tracks.values() if track.is_active]

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1

    def _create_track(self, det: DetectedFace, frame_index: int) -> None:
        det.track_id = self.next_track_id
        det.frame_index = frame_index
        track = TrackedFaceSequence(
            track_id=self.next_track_id,
            faces=[det],
            last_seen_frame=frame_index,
            is_active=True,
        )
        track.recompute_aggregates()
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def _update_track(self, track: TrackedFaceSequence, det: DetectedFace, frame_index: int) -> None:
        det.track_id = track.track_id
        det.frame_index = frame_index
        track.faces.append(det)
        track.last_seen_frame = frame_index
        track.missed_frames = 0
        track.is_active = True
        track.recompute_aggregates()

    def _calculate_iou(self, bbox1: np.ndarray[Any, Any], bbox2: np.ndarray[Any, Any]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return float(intersection / union)

    def _embedding_similarity(self, det: DetectedFace, track: TrackedFaceSequence) -> float:
        if det.embedding.size == 0 or track.embedding_centroid.size == 0:
            return 0.0
        det_embedding = det.embedding.astype(np.float32)
        det_norm = float(np.linalg.norm(det_embedding))
        if det_norm == 0:
            return 0.0
        det_embedding = det_embedding / det_norm
        similarity = float(np.dot(det_embedding, track.embedding_centroid))
        return max(0.0, similarity)
