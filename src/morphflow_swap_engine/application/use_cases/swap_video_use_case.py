from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from morphflow_swap_engine.config.schema import EngineConfig
from morphflow_swap_engine.core.contracts.i_artifact_store import IArtifactStore
from morphflow_swap_engine.core.contracts.i_face_aligner import IFaceAligner
from morphflow_swap_engine.core.contracts.i_face_detector import IFaceDetector
from morphflow_swap_engine.core.contracts.i_face_restorer import IFaceRestorer
from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper
from morphflow_swap_engine.core.contracts.i_face_tracker import IFaceTracker
from morphflow_swap_engine.core.contracts.i_temporal_stabilizer import ITemporalStabilizer
from morphflow_swap_engine.core.contracts.i_video_decoder import IVideoDecoder
from morphflow_swap_engine.core.contracts.i_video_encoder import IVideoEncoder
from morphflow_swap_engine.core.entities.swap_request import SwapRequest
from morphflow_swap_engine.core.entities.swap_result import SwapResult
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
        
        # Prepare artifact directories
        job_artifact_dir = Path(self.config.artifact_dir) / str(int(start_time))
        if self.config.save_artifacts and self.artifact_store:
            job_artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Probe video
            asset = self.decoder.probe(request.target_asset)
            
            # 2. Extract reference embeddings
            if not request.reference_faces:
                raise ValueError("No reference faces provided.")
            
            ref_path = request.reference_faces[0].asset_path
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                raise ValueError(f"Could not read reference image: {ref_path}")
                
            ref_faces = self.detector.detect(ref_img, score_threshold=self.config.detector_score_threshold)
            if not ref_faces:
                raise ValueError("No face detected in reference image.")
                
            # Pick largest/highest score face as reference
            ref_faces.sort(key=lambda f: f.score, reverse=True)
            source_embedding = ref_faces[0].embedding
            
            if self.config.save_artifacts and self.artifact_store:
                self.artifact_store.save("source_face.jpg", ref_img, job_artifact_dir / "00_reference")

            # 3. Two-pass or One-pass? 
            # Pass 1: Detection & Tracking
            frame_idx = 0
            for frame in self.decoder.frames(asset):
                detections = self.detector.detect(frame, score_threshold=self.config.detector_score_threshold)
                self.tracker.update(detections, frame_idx)
                frame_idx += 1

            tracks = list(self.tracker.active_tracks.values())
            best_track = self.track_scorer.find_best_track(tracks, (asset.width, asset.height))
            
            if not best_track:
                raise ValueError("No target face track found in video.")
                
            target_track_id = best_track.track_id
            
            if self.config.save_artifacts and self.artifact_store:
                self.artifact_store.save("track_info.json", {
                    "track_id": target_track_id,
                    "frame_count": len(best_track.faces),
                }, job_artifact_dir / "02_tracking")
            
            # Pass 2: Swapping & Reconstruction
            output_path = Path(request.output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
            def process_frames():
                nonlocal frames_processed
                # Reset tracker/temporal for pass 2
                if self.temporal_stabilizer:
                    self.temporal_stabilizer.reset()
                
                batch_size = self.config.batch_size
                batch_frames = []
                batch_faces = []
                batch_indices = []

                def flush_batch():
                    if not batch_frames:
                        return []
                    
                    # 1. Prepare batch crops and matrices
                    crops = []
                    inv_matrices = []
                    valid_indices = []
                    
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
                            
                            # Restore
                            if self.config.enable_restoration and self.restorer:
                                swapped_crop = self.restorer.restore(swapped_crop)
                            
                            # Temporal Stabilize
                            if self.config.enable_temporal_stabilization and self.temporal_stabilizer:
                                swapped_crop = self.temporal_stabilizer.stabilize(swapped_crop, target_track_id)
                                
                            # Paste back
                            mask_height, mask_width = swapped_crop.shape[:2]
                            mask = np.ones((mask_height, mask_width, 1), dtype=np.float32)
                            margin = int(min(mask_height, mask_width) * 0.1)
                            mask[:margin, :] *= np.linspace(0, 1, margin).reshape(-1, 1, 1)
                            mask[-margin:, :] *= np.linspace(1, 0, margin).reshape(-1, 1, 1)
                            mask[:, :margin] *= np.linspace(0, 1, margin).reshape(1, -1, 1)
                            mask[:, -margin:] *= np.linspace(1, 0, margin).reshape(1, -1, 1)
                            blur_size = (margin * 2) | 1
                            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                            if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=-1)
                            
                            paste_w, paste_h = batch_frames[orig_idx].shape[1], batch_frames[orig_idx].shape[0]
                            inv_mask = cv2.warpAffine(mask, inv_matrix, (paste_w, paste_h))
                            if len(inv_mask.shape) == 2: inv_mask = np.expand_dims(inv_mask, axis=-1)
                            
                            inv_crop = cv2.warpAffine(swapped_crop, inv_matrix, (paste_w, paste_h), borderMode=cv2.BORDER_REPLICATE)
                            
                            batch_frames[orig_idx] = (batch_frames[orig_idx] * (1 - inv_mask) + inv_crop * inv_mask).astype(np.uint8)

                            if self.config.save_artifacts and self.artifact_store and frame_idx % 30 == 0:
                                self.artifact_store.save(f"frame_{frame_idx}_final_crop.jpg", swapped_crop, job_artifact_dir / "07_final")

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
            success = True

        except Exception as e:
            error_message = str(e)
            success = False

        duration = time.time() - start_time
        return SwapResult(
            output_path=request.output_path,
            frames_processed=frames_processed,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
        )
