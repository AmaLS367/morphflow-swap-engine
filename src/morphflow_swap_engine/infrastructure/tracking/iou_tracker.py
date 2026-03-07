from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from morphflow_swap_engine.core.contracts.i_face_tracker import IFaceTracker
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.entities.tracked_face_sequence import TrackedFaceSequence


class IOUFaceTracker(IFaceTracker):
    """Simple IOU-based face tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.active_tracks: Dict[int, TrackedFaceSequence] = {}
        self.lost_frames: Dict[int, int] = {}
        self.next_track_id = 1

    def update(self, detections: List[DetectedFace], frame_index: int) -> List[TrackedFaceSequence]:
        """Match detections with existing tracks using IOU."""
        if not self.active_tracks:
            # Initialize new tracks for all detections
            for det in detections:
                det.track_id = self.next_track_id
                det.frame_index = frame_index
                seq = TrackedFaceSequence(track_id=self.next_track_id, faces=[det])
                self.active_tracks[self.next_track_id] = seq
                self.lost_frames[self.next_track_id] = 0
                self.next_track_id += 1
            return list(self.active_tracks.values())

        # Match detections to existing tracks
        track_ids = list(self.active_tracks.keys())
        last_faces = [self.active_tracks[tid].faces[-1] for tid in track_ids]
        
        matches: List[tuple[int, int, float]] = []
        for det_idx, det in enumerate(detections):
            best_iou = -1.0
            best_track_idx = -1
            for track_idx, last_face in enumerate(last_faces):
                iou = self._calculate_iou(det.bounding_box, last_face.bounding_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx != -1:
                matches.append((det_idx, best_track_idx, best_iou))

        # Sort matches by IOU and keep unique pairs
        matches.sort(key=lambda x: x[2], reverse=True)
        used_det = set()
        used_track = set()
        
        final_matches = []
        for det_idx, track_idx, iou in matches:
            if det_idx not in used_det and track_idx not in used_track:
                final_matches.append((det_idx, track_idx))
                used_det.add(det_idx)
                used_track.add(track_idx)

        # Update matched tracks
        for det_idx, track_idx in final_matches:
            tid = track_ids[track_idx]
            det = detections[det_idx]
            det.track_id = tid
            det.frame_index = frame_index
            self.active_tracks[tid].faces.append(det)
            self.lost_frames[tid] = 0

        # Handle unmatched detections (new tracks)
        for det_idx, det in enumerate(detections):
            if det_idx not in used_det:
                det.track_id = self.next_track_id
                det.frame_index = frame_index
                seq = TrackedFaceSequence(track_id=self.next_track_id, faces=[det])
                self.active_tracks[self.next_track_id] = seq
                self.lost_frames[self.next_track_id] = 0
                self.next_track_id += 1

        # Handle unmatched tracks (lost)
        lost_track_ids = [tid for i, tid in enumerate(track_ids) if i not in used_track]
        for tid in lost_track_ids:
            self.lost_frames[tid] += 1
            if self.lost_frames[tid] > self.max_lost_frames:
                # Still keep track data but mark it for final results if needed
                # For now just stop updating it
                pass

        return list(self.active_tracks.values())

    def reset(self) -> None:
        self.active_tracks.clear()
        self.lost_frames.clear()
        self.next_track_id = 1

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
