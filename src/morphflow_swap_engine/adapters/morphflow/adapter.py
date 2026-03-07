from __future__ import annotations

from typing import Any, Dict

from .feature_flag import use_new_engine
from .request_mapper import map_request
from .response_mapper import map_response
from ...core.entities.swap_result import SwapResult

from ...config.loader import load_config
from ...core.services.track_scorer import TrackScorer
from ...infrastructure.alignment.affine_face_aligner import AffineFaceAligner
from ...infrastructure.detection.insightface_detector import InsightFaceDetector
from ...infrastructure.restoration.codeformer_restorer import CodeFormerRestorer
from ...infrastructure.swapping.ghost_swapper import GhostSwapper
from ...infrastructure.temporal.film_stabilizer import FilmStabilizer
from ...infrastructure.tracking.iou_tracker import IOUFaceTracker
from ...infrastructure.video.opencv_video_decoder import OpenCVVideoDecoder
from ...infrastructure.video.opencv_video_encoder import OpenCVVideoEncoder
from ...application.use_cases.swap_video_use_case import SwapVideoUseCase


class MorphFlowAdapter:
    """Entry point that routes MorphFlow API calls to the correct engine.

    When the MORPHFLOW_NEW_ENGINE feature flag is off, the adapter raises
    NotImplementedError so the caller falls back to the legacy FaceFusion path.
    When the flag is on, the adapter delegates to the new engine pipeline.
    """

    def __init__(self) -> None:
        self.config = load_config()
        self.decoder = OpenCVVideoDecoder()
        self.encoder = OpenCVVideoEncoder()
        self.detector = InsightFaceDetector()
        self.tracker = IOUFaceTracker()
        self.track_scorer = TrackScorer()
        self.aligner = AffineFaceAligner(crop_size=512, template="ffhq")
        
        # Paths would typically come from config, using defaults for now
        self.swapper = GhostSwapper(model_path="models/ghost_1_256.onnx", batch_size=self.config.batch_size, use_fp16=self.config.use_fp16)
        self.restorer = CodeFormerRestorer(model_path="models/codeformer.onnx", use_fp16=self.config.use_fp16) if self.config.enable_restoration else None
        self.temporal_stabilizer = FilmStabilizer(model_path="models/film.onnx", use_fp16=self.config.use_fp16) if self.config.enable_temporal_stabilization else None

        self.use_case = SwapVideoUseCase(
            config=self.config,
            decoder=self.decoder,
            encoder=self.encoder,
            detector=self.detector,
            tracker=self.tracker,
            track_scorer=self.track_scorer,
            aligner=self.aligner,
            swapper=self.swapper,
            restorer=self.restorer,
            temporal_stabilizer=self.temporal_stabilizer
        )

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a swap request payload and return a response dict."""
        if not use_new_engine():
            raise NotImplementedError(
                "New engine is disabled. Set MORPHFLOW_NEW_ENGINE=1 to enable."
            )

        request = map_request(payload)
        
        # Set profile if requested
        if request.profile_name:
            self.config.profile = request.profile_name

        try:
            result = self.use_case.execute(request)
        except Exception as e:
            result = SwapResult(
                output_path=request.output_path,
                frames_processed=0,
                duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )

        return map_response(result)
