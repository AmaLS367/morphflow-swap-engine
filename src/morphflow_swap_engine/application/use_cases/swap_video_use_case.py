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
from morphflow_swap_engine.core.services.track_scorer import TrackScorer


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
                    
                for frame_idx, frame in enumerate(self.decoder.frames(asset)):
                    # Get the face for this frame from the best track
                    target_face = None
                    for face in best_track.faces:
                        if face.frame_index == frame_idx:
                            target_face = face
                            break
                            
                    if target_face is not None:
                        # 1. Align & Crop
                        crop, inv_matrix = self.aligner.align(frame, target_face)
                        if self.config.save_artifacts and self.artifact_store and frame_idx % 30 == 0:
                            self.artifact_store.save(f"frame_{frame_idx}_crop.jpg", crop, job_artifact_dir / "03_alignment")
                        
                        # 2. Swap
                        swapped_crop = self.swapper.swap(source_embedding, crop)
                        if self.config.save_artifacts and self.artifact_store and frame_idx % 30 == 0:
                            self.artifact_store.save(f"frame_{frame_idx}_swapped.jpg", swapped_crop, job_artifact_dir / "04_swap")
                        
                        # 3. Restore (optional)
                        if self.config.enable_restoration and self.restorer:
                            swapped_crop = self.restorer.restore(swapped_crop)
                            if self.config.save_artifacts and self.artifact_store and frame_idx % 30 == 0:
                                self.artifact_store.save(f"frame_{frame_idx}_restored.jpg", swapped_crop, job_artifact_dir / "05_restore")
                            
                        # 4. Temporal Stabilize (optional)
                        if self.config.enable_temporal_stabilization and self.temporal_stabilizer:
                            swapped_crop = self.temporal_stabilizer.stabilize(swapped_crop, target_track_id)
                            if self.config.save_artifacts and self.artifact_store and frame_idx % 30 == 0:
                                self.artifact_store.save(f"frame_{frame_idx}_stabilized.jpg", swapped_crop, job_artifact_dir / "06_temporal")

                            
                        # 5. Paste back
                        # Create a soft mask based on the crop size to avoid box artifacts.
                        # We use a box mask with blurred edges.
                        mask_height, mask_width = swapped_crop.shape[:2]
                        mask = np.ones((mask_height, mask_width, 1), dtype=np.float32)
                        
                        # Add some padding to the mask to avoid sharp edges at the very boundary
                        # (The crop might already have replicate padding, but a soft transition is better)
                        margin = int(min(mask_height, mask_width) * 0.1)
                        mask[:margin, :] *= np.linspace(0, 1, margin).reshape(-1, 1, 1)
                        mask[-margin:, :] *= np.linspace(1, 0, margin).reshape(-1, 1, 1)
                        mask[:, :margin] *= np.linspace(0, 1, margin).reshape(1, -1, 1)
                        mask[:, -margin:] *= np.linspace(1, 0, margin).reshape(1, -1, 1)
                        
                        # Further blur the mask for a smoother transition
                        blur_size = (margin * 2) | 1 # Ensure odd
                        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                        if len(mask.shape) == 2:
                            mask = np.expand_dims(mask, axis=-1)
                        
                        paste_width = frame.shape[1]
                        paste_height = frame.shape[0]
                        
                        # Warp mask and swapped crop back to original frame coordinates
                        inv_mask = cv2.warpAffine(mask, inv_matrix, (paste_width, paste_height))
                        inv_mask = np.clip(inv_mask, 0, 1)
                        if len(inv_mask.shape) == 2:
                            inv_mask = np.expand_dims(inv_mask, axis=-1)
                            
                        inv_crop = cv2.warpAffine(swapped_crop, inv_matrix, (paste_width, paste_height), borderMode=cv2.BORDER_REPLICATE)
                        
                        # Blend the swapped crop with the original frame using the mask
                        frame = (frame * (1 - inv_mask) + inv_crop * inv_mask).astype(np.uint8)

                    frames_processed += 1
                    yield frame
            
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
