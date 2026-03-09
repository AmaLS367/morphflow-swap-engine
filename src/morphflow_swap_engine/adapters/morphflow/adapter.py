from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .request_mapper import map_request
from .response_mapper import map_response
from ...core.entities.swap_result import SwapResult

from ...config import apply_profile, load_config
from ...core.services.detection_filter import FaceDetectionFilter
from ...core.services.primary_face_selector import PrimaryFaceSelector
from ...core.services.reference_face_analyzer import ReferenceFaceAnalyzer
from ...core.services.target_video_analyzer import TargetVideoAnalyzer
from ...core.services.track_scorer import TrackScorer
from ...infrastructure.alignment.affine_face_aligner import AffineFaceAligner
from ...infrastructure.detection.insightface_detector import InsightFaceDetector
from ...infrastructure.restoration.codeformer_restorer import CodeFormerRestorer
from ...infrastructure.swapping.ghost_swapper import GhostSwapper
from ...infrastructure.swapping.simswap_swapper import SimSwapSwapper
from ...infrastructure.temporal.film_stabilizer import FilmStabilizer
from ...infrastructure.tracking.iou_tracker import IOUFaceTracker
from ...infrastructure.video.opencv_video_decoder import OpenCVVideoDecoder
from ...infrastructure.video.opencv_video_encoder import OpenCVVideoEncoder
from ...infrastructure.diagnostics.local_artifact_store import LocalArtifactStore
from ...application.use_cases.swap_video_use_case import SwapVideoUseCase
from ...config.schema import EngineConfig
from ...core.contracts.i_face_restorer import IFaceRestorer
from ...core.contracts.i_face_swapper import IFaceSwapper
from ...core.contracts.i_temporal_stabilizer import ITemporalStabilizer


class MorphFlowAdapter:
    """Entry point that routes MorphFlow API calls into the swap engine."""

    MODEL_PATHS = {
        "ghost_512": "models/ghost_512.onnx",
        "simswap_512": "models/simswap_512.onnx",
        "codeformer": "models/codeformer.onnx",
        "film": "models/film.onnx",
    }

    def __init__(self) -> None:
        self.artifact_store = LocalArtifactStore()
        self.primary_face_selector = PrimaryFaceSelector()

    def _build_swapper(self, config: EngineConfig) -> IFaceSwapper:
        if config.swap_model_key == "ghost_512":
            return GhostSwapper(
                model_path=self.MODEL_PATHS["ghost_512"],
                execution_providers=config.execution_providers,
                batch_size=config.batch_size,
                use_fp16=config.use_fp16,
            )
        if config.swap_model_key == "simswap_512":
            return SimSwapSwapper(
                model_path=self.MODEL_PATHS["simswap_512"],
                execution_providers=config.execution_providers,
                batch_size=config.batch_size,
                use_fp16=config.use_fp16,
            )
        raise ValueError(f"Unsupported swap model key: {config.swap_model_key}")

    def _build_restorer(self, config: EngineConfig) -> IFaceRestorer | None:
        if not config.enable_restoration:
            return None
        return CodeFormerRestorer(
            model_path=self.MODEL_PATHS["codeformer"],
            execution_providers=config.execution_providers,
            use_fp16=config.use_fp16,
        )

    def _build_temporal(self, config: EngineConfig) -> ITemporalStabilizer | None:
        if not config.enable_temporal_stabilization:
            return None
        return FilmStabilizer(
            model_path=self.MODEL_PATHS["film"],
            execution_providers=config.execution_providers,
            use_fp16=config.use_fp16,
        )

    def _build_use_case(self, config: EngineConfig) -> SwapVideoUseCase:
        return SwapVideoUseCase(
            config=config,
            decoder=OpenCVVideoDecoder(),
            encoder=OpenCVVideoEncoder(),
            detector=InsightFaceDetector(providers=config.execution_providers),
            tracker=IOUFaceTracker(iou_threshold=config.detector_iou_threshold),
            track_scorer=TrackScorer(),
            detection_filter=FaceDetectionFilter(
                min_face_size=config.detector_min_face_size,
                min_face_ratio=config.detector_min_face_ratio,
                min_centrality=config.detector_min_centrality,
            ),
            reference_analyzer=ReferenceFaceAnalyzer(selector=self.primary_face_selector),
            target_analyzer=TargetVideoAnalyzer(
                selector=self.primary_face_selector,
                sample_count=config.target_analysis_sample_count,
            ),
            aligner=AffineFaceAligner(crop_size=512, template="ffhq"),
            swapper=self._build_swapper(config),
            restorer=self._build_restorer(config),
            temporal_stabilizer=self._build_temporal(config),
            artifact_store=self.artifact_store,
        )

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a swap request payload and return a response dict."""
        request = map_request(payload)
        config_path = payload.get("config_path")
        ini_path = Path(config_path) if config_path else None
        config = apply_profile(load_config(ini_path), request.profile_name)
        use_case = self._build_use_case(config)

        try:
            result = use_case.execute(request)
        except Exception as e:
            result = SwapResult(
                output_path=request.output_path,
                frames_processed=0,
                duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )

        return map_response(result)
