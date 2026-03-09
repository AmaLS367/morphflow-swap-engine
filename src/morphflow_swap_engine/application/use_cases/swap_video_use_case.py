from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from morphflow_swap_engine.config.schema import EngineConfig
from morphflow_swap_engine.core.services.detection_filter import FaceDetectionFilter
from morphflow_swap_engine.core.services.reference_face_analyzer import ReferenceFaceAnalyzer
from morphflow_swap_engine.core.services.target_video_analyzer import TargetVideoAnalyzer
from morphflow_swap_engine.core.services.track_scorer import TrackScorer
from morphflow_swap_engine.core.contracts.i_artifact_store import IArtifactStore
from morphflow_swap_engine.core.contracts.i_face_aligner import IFaceAligner
from morphflow_swap_engine.core.contracts.i_face_detector import IFaceDetector
from morphflow_swap_engine.core.contracts.i_face_restorer import IFaceRestorer
from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper
from morphflow_swap_engine.core.contracts.i_face_tracker import IFaceTracker
from morphflow_swap_engine.core.contracts.i_temporal_stabilizer import ITemporalStabilizer
from morphflow_swap_engine.core.contracts.i_video_decoder import IVideoDecoder
from morphflow_swap_engine.core.contracts.i_video_encoder import IVideoEncoder
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.entities.swap_request import SwapRequest
from morphflow_swap_engine.core.entities.swap_result import SwapResult
from morphflow_swap_engine.core.value_objects.runtime_report import RuntimeReport
from morphflow_swap_engine.core.value_objects.stage_artifact import StageArtifact
from morphflow_swap_engine.infrastructure.restoration import apply_color_transfer


class SwapVideoUseCase:
    """Orchestrates the end-to-step face swap pipeline on a video."""

    def __init__(
        self,
        config: EngineConfig,
        decoder: IVideoDecoder,
        encoder: IVideoEncoder,
        detector: IFaceDetector,
        tracker: IFaceTracker,
        track_scorer: TrackScorer,
        detection_filter: FaceDetectionFilter,
        reference_analyzer: ReferenceFaceAnalyzer,
        target_analyzer: TargetVideoAnalyzer,
        aligner: IFaceAligner,
        swapper: IFaceSwapper,
        restorer: Optional[IFaceRestorer] = None,
        temporal_stabilizer: Optional[ITemporalStabilizer] = None,
        artifact_store: Optional[IArtifactStore] = None,
    ):
        self.config = config
        self.decoder = decoder
        self.encoder = encoder
        self.detector = detector
        self.tracker = tracker
        self.track_scorer = track_scorer
        self.detection_filter = detection_filter
        self.reference_analyzer = reference_analyzer
        self.target_analyzer = target_analyzer
        self.aligner = aligner
        self.swapper = swapper
        self.restorer = restorer
        self.temporal_stabilizer = temporal_stabilizer
        self.artifact_store = artifact_store

    def execute(self, request: SwapRequest) -> SwapResult:
        start_time = time.time()
        frames_processed = 0
        success = False
        error_message = ""
        artifacts: list[StageArtifact] = []
        warnings: list[str] = []
        stage_timings: dict[str, float] = {}

        job_artifact_dir = Path(self.config.artifact_dir) / str(int(start_time))
        metadata_dir = job_artifact_dir / "metadata"
        logs_dir = job_artifact_dir / "logs"
        stage_root_dir = job_artifact_dir / "artifacts"
        if self.config.save_artifacts and self.artifact_store:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            stage_root_dir.mkdir(parents=True, exist_ok=True)

        def save_artifact(stage_name: str, name: str, data: Any) -> None:
            if not (self.config.save_artifacts and self.artifact_store):
                return
            artifact = self.artifact_store.save(name, data, stage_root_dir / stage_name)
            artifacts.append(artifact)

        def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
            import json

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def sample_track_frame_indices(track_faces: list[DetectedFace], max_samples: int = 5) -> set[int]:
            if not track_faces:
                return set()

            sample_count = min(max_samples, len(track_faces))
            if sample_count <= 1:
                return {track_faces[0].frame_index}

            sampled_indices = {
                int(round(position * (len(track_faces) - 1) / float(sample_count - 1)))
                for position in range(sample_count)
            }
            return {track_faces[index].frame_index for index in sampled_indices}

        try:
            # 1. Probe video
            detect_track_started = time.time()
            asset = self.decoder.probe(request.target_asset)
            self.tracker.reset()
            sampled_frame_indices = set(self.target_analyzer.sample_frame_indices(asset.frame_count))
            sampled_target_frames = []

            # 2. Extract reference embeddings
            if not request.reference_faces:
                raise ValueError("No reference faces provided.")

            ref_path = request.reference_faces[0].asset_path
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                raise ValueError(f"Could not read reference image: {ref_path}")

            ref_faces = self.detector.detect(ref_img, score_threshold=self.config.detector_score_threshold)
            filtered_ref_faces = self.detection_filter.filter_faces(
                ref_faces,
                (ref_img.shape[1], ref_img.shape[0]),
            )
            reference_analysis = self.reference_analyzer.analyze(
                (ref_img.shape[1], ref_img.shape[0]),
                ref_faces,
                filtered_ref_faces,
            )
            warnings.extend(reference_analysis.warnings)

            if reference_analysis.primary_face is None:
                raise ValueError("No usable face detected in reference image.")

            source_embedding = reference_analysis.primary_face.embedding

            save_artifact("01_detection", "source_face.jpg", ref_img)
            save_artifact("01_detection", "reference_analysis", reference_analysis.to_dict())

            # 3. Two-pass or One-pass? 
            # Pass 1: Detection & Tracking
            detection_batch_size = max(1, self.config.detector_batch_size)
            detection_frames: list[np.ndarray[Any, Any]] = []
            detection_indices: list[int] = []

            def flush_detection_batch() -> None:
                if not detection_frames:
                    return

                raw_batches = self.detector.detect_batch(
                    detection_frames,
                    score_threshold=self.config.detector_score_threshold,
                )
                for frame, frame_index, raw_detections in zip(detection_frames, detection_indices, raw_batches):
                    filtered_detections = self.detection_filter.filter_faces(
                        raw_detections,
                        (frame.shape[1], frame.shape[0]),
                    )
                    self.tracker.update(filtered_detections, frame_index)

                    if frame_index in sampled_frame_indices:
                        sampled_target_frames.append(
                            self.target_analyzer.analyze_frame(
                                frame_index=frame_index,
                                frame_size=(frame.shape[1], frame.shape[0]),
                                raw_faces=raw_detections,
                                filtered_faces=filtered_detections,
                            )
                        )

                detection_frames.clear()
                detection_indices.clear()

            for frame_idx, frame in enumerate(self.decoder.frames(asset)):
                detection_frames.append(frame)
                detection_indices.append(frame_idx)
                if len(detection_frames) >= detection_batch_size:
                    flush_detection_batch()

            flush_detection_batch()

            target_analysis = self.target_analyzer.summarize(asset.frame_count, sampled_target_frames)
            warnings.extend(target_analysis.warnings)
            save_artifact("01_detection", "target_analysis", target_analysis.to_dict())

            stage_timings["detection_tracking_seconds"] = time.time() - detect_track_started
            tracks = self.tracker.get_tracks(include_inactive=True)
            best_track = self.track_scorer.find_best_track(tracks, (asset.width, asset.height))

            if not best_track:
                raise ValueError("No target face track found in video.")

            target_track_id = best_track.track_id
            selected_track_sample_indices = sample_track_frame_indices(best_track.faces, max_samples=5)
            tracking_manifest = [track.to_dict() for track in sorted(tracks, key=lambda item: item.track_id)]

            save_artifact("02_tracking", "track_manifest", tracking_manifest)
            save_artifact("02_tracking", "selected_track", best_track.to_dict())

            # Pass 2: Swapping & Reconstruction
            output_path = Path(request.output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)

            reconstruction_started = time.time()

            def process_frames() -> Generator[np.ndarray[Any, Any], None, None]:
                nonlocal frames_processed
                # Reset tracker/temporal for pass 2
                if self.temporal_stabilizer:
                    self.temporal_stabilizer.reset()
                saved_tracking_sample_indices: set[int] = set()

                batch_size = self.config.batch_size
                batch_frames: list[np.ndarray[Any, Any]] = []
                batch_faces: list[Optional[DetectedFace]] = []
                batch_indices: list[int] = []

                def flush_batch() -> list[np.ndarray[Any, Any]]:
                    if not batch_frames:
                        return []

                    # 1. Prepare batch crops and matrices
                    crops: list[np.ndarray[Any, Any]] = []
                    inv_matrices: list[np.ndarray[Any, Any]] = []
                    valid_indices: list[int] = []

                    for i, face in enumerate(batch_faces):
                        if face is not None:
                            crop, inv_matrix = self.aligner.align(batch_frames[i], face)
                            crops.append(crop)
                            inv_matrices.append(inv_matrix)
                            valid_indices.append(i)

                    if crops:
                        # 2. Batched Swap
                        swapped_crops = self.swapper.swap_batch(source_embedding, crops)

                        # 3. Post-process each swapped crop
                        for i, swapped_crop in enumerate(swapped_crops):
                            orig_idx = valid_indices[i]
                            frame_idx = batch_indices[orig_idx]
                            orig_crop = crops[i]
                            inv_matrix = inv_matrices[i]

                            # Apply color transfer
                            swapped_crop = apply_color_transfer(orig_crop, swapped_crop)
                            if frame_idx % 60 == 0:
                                save_artifact("04_swap", f"frame_{frame_idx}_swap.jpg", swapped_crop)

                            # Restore
                            if self.config.enable_restoration and self.restorer:
                                swapped_crop = self.restorer.restore(swapped_crop)
                                if frame_idx % 60 == 0:
                                    save_artifact("05_restore", f"frame_{frame_idx}_restore.jpg", swapped_crop)

                            # Temporal Stabilize
                            if self.config.enable_temporal_stabilization and self.temporal_stabilizer:
                                swapped_crop = self.temporal_stabilizer.stabilize(swapped_crop, target_track_id)
                                if frame_idx % 60 == 0:
                                    save_artifact("06_temporal", f"frame_{frame_idx}_temporal.jpg", swapped_crop)

                            # Paste back
                            mask_height, mask_width = swapped_crop.shape[:2]
                            mask = np.ones((mask_height, mask_width, 1), dtype=np.float32)
                            margin = int(min(mask_height, mask_width) * 0.1)
                            if margin > 0:
                                mask[:margin, :] *= np.linspace(0, 1, margin).reshape(-1, 1, 1)
                                mask[-margin:, :] *= np.linspace(1, 0, margin).reshape(-1, 1, 1)
                                mask[:, :margin] *= np.linspace(0, 1, margin).reshape(1, -1, 1)
                                mask[:, -margin:] *= np.linspace(1, 0, margin).reshape(1, -1, 1)
                                blur_size = (margin * 2) | 1
                                mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                            if len(mask.shape) == 2:
                                mask = np.expand_dims(mask, axis=-1)

                            paste_w, paste_h = batch_frames[orig_idx].shape[1], batch_frames[orig_idx].shape[0]
                            inv_mask = cv2.warpAffine(mask, inv_matrix, (paste_w, paste_h))
                            if len(inv_mask.shape) == 2:
                                inv_mask = np.expand_dims(inv_mask, axis=-1)

                            inv_crop = cv2.warpAffine(swapped_crop, inv_matrix, (paste_w, paste_h), borderMode=cv2.BORDER_REPLICATE)

                            batch_frames[orig_idx] = (batch_frames[orig_idx] * (1 - inv_mask) + inv_crop * inv_mask).astype(np.uint8)

                            if frame_idx % 60 == 0:
                                save_artifact("07_reconstruction", f"frame_{frame_idx}_frame.jpg", batch_frames[orig_idx])

                    results = list(batch_frames)
                    batch_frames.clear()
                    batch_faces.clear()
                    batch_indices.clear()
                    return results

                for frame_idx, frame in enumerate(self.decoder.frames(asset)):
                    target_face = None
                    for face in best_track.faces:
                        if face.frame_index == frame_idx:
                            target_face = face
                            break

                    if (
                        target_face is not None
                        and frame_idx in selected_track_sample_indices
                        and frame_idx not in saved_tracking_sample_indices
                    ):
                        sample_crop, _ = self.aligner.align(frame, target_face)
                        save_artifact(
                            "02_tracking",
                            f"track_{target_track_id}_frame_{frame_idx}_crop.jpg",
                            sample_crop,
                        )
                        saved_tracking_sample_indices.add(frame_idx)

                    batch_frames.append(frame)
                    batch_faces.append(target_face)
                    batch_indices.append(frame_idx)

                    if len(batch_frames) >= batch_size:
                        for processed_frame in flush_batch():
                            frames_processed += 1
                            yield processed_frame

                # Final flush
                for processed_frame in flush_batch():
                    frames_processed += 1
                    yield processed_frame

            # Encode
            self.encoder.encode(process_frames(), asset, output_path)
            stage_timings["reconstruction_seconds"] = time.time() - reconstruction_started
            success = True

        except Exception as e:
            error_message = str(e)
            success = False
            if self.config.save_artifacts:
                (logs_dir / "run.log").write_text(error_message, encoding="utf-8")

        duration = time.time() - start_time
        runtime_report = RuntimeReport(
            total_frames=frames_processed,
            total_duration_seconds=duration,
            avg_fps=(frames_processed / duration) if duration > 0 else 0.0,
            artifacts=artifacts,
            warnings=warnings,
        )
        if self.config.save_artifacts:
            write_json(
                metadata_dir / "artifact_manifest.json",
                [
                    {
                        "stage_name": artifact.stage_name,
                        "artifact_path": str(artifact.artifact_path),
                        "metadata": artifact.metadata,
                    }
                    for artifact in artifacts
                ],
            )
            write_json(
                metadata_dir / "runtime_report.json",
                {
                    "profile": self.config.profile,
                    "success": success,
                    "error_message": error_message,
                    "used_modules": {
                        "detector": type(self.detector).__name__,
                        "tracker": type(self.tracker).__name__,
                        "aligner": type(self.aligner).__name__,
                        "swapper": type(self.swapper).__name__,
                        "restorer": type(self.restorer).__name__ if self.restorer else None,
                        "temporal": type(self.temporal_stabilizer).__name__ if self.temporal_stabilizer else None,
                    },
                    "stage_timings": stage_timings,
                    "total_frames": runtime_report.total_frames,
                    "total_duration_seconds": runtime_report.total_duration_seconds,
                    "avg_fps": runtime_report.avg_fps,
                    "warnings": runtime_report.warnings,
                },
            )
            if not error_message:
                (logs_dir / "run.log").write_text("run completed successfully", encoding="utf-8")

        return SwapResult(
            output_path=Path(request.output_path),
            frames_processed=frames_processed,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
        )
