from __future__ import annotations

import json
import shutil
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from morphflow_swap_engine.application.use_cases.swap_video_use_case import SwapVideoUseCase
from morphflow_swap_engine.config.loader import load_config
from morphflow_swap_engine.config.schema import EngineConfig
from morphflow_swap_engine.core.contracts.i_face_detector import IFaceDetector
from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper
from morphflow_swap_engine.core.contracts.i_video_decoder import IVideoDecoder
from morphflow_swap_engine.core.contracts.i_video_encoder import IVideoEncoder
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.entities.reference_face_asset import ReferenceFaceAsset
from morphflow_swap_engine.core.entities.swap_request import SwapRequest
from morphflow_swap_engine.core.entities.target_video_asset import TargetVideoAsset
from morphflow_swap_engine.core.services.detection_filter import FaceDetectionFilter
from morphflow_swap_engine.core.services.face_crop_strategy import FaceCropStrategy
from morphflow_swap_engine.core.services.primary_face_selector import PrimaryFaceSelector
from morphflow_swap_engine.core.services.reference_face_analyzer import ReferenceFaceAnalyzer
from morphflow_swap_engine.core.services.target_video_analyzer import TargetVideoAnalyzer
from morphflow_swap_engine.core.services.track_scorer import TrackScorer
from morphflow_swap_engine.infrastructure.alignment.affine_face_aligner import AffineFaceAligner
from morphflow_swap_engine.infrastructure.diagnostics.local_artifact_store import LocalArtifactStore
from morphflow_swap_engine.infrastructure.tracking.iou_tracker import IOUFaceTracker


def _embedding(values: list[float]) -> np.ndarray[Any, Any]:
    return np.array(values, dtype=np.float32)


def _gradient_frame(size: int = 128) -> np.ndarray[Any, Any]:
    base = np.linspace(0, 255, size, dtype=np.uint8)
    grid_x = np.tile(base, (size, 1))
    grid_y = np.tile(base.reshape(-1, 1), (1, size))
    red = grid_x
    green = grid_y
    blue = ((grid_x.astype(np.uint16) + grid_y.astype(np.uint16)) // 2).astype(np.uint8)
    return np.dstack([blue, green, red])


def _face(
    bbox: tuple[float, float, float, float],
    *,
    score: float = 0.9,
    face_area_ratio: float | None = None,
    embedding: np.ndarray[Any, Any] | None = None,
    frame_index: int = -1,
    track_id: int = -1,
) -> DetectedFace:
    x1, y1, x2, y2 = bbox
    if face_area_ratio is None:
        face_area_ratio = 0.0
    return DetectedFace(
        bounding_box=np.array([x1, y1, x2, y2], dtype=np.float32),
        landmark_5=np.array(
            [
                [x1 + 0.30 * (x2 - x1), y1 + 0.35 * (y2 - y1)],
                [x1 + 0.70 * (x2 - x1), y1 + 0.35 * (y2 - y1)],
                [x1 + 0.50 * (x2 - x1), y1 + 0.55 * (y2 - y1)],
                [x1 + 0.35 * (x2 - x1), y1 + 0.78 * (y2 - y1)],
                [x1 + 0.65 * (x2 - x1), y1 + 0.78 * (y2 - y1)],
            ],
            dtype=np.float32,
        ),
        score=score,
        face_area_ratio=face_area_ratio,
        embedding=embedding if embedding is not None else _embedding([1.0, 0.0, 0.0]),
        frame_index=frame_index,
        track_id=track_id,
    )


class _FakeDecoder(IVideoDecoder):
    def __init__(self, frames: list[np.ndarray[Any, Any]]) -> None:
        self._frames = frames

    def probe(self, asset: TargetVideoAsset) -> TargetVideoAsset:
        asset.width = int(self._frames[0].shape[1])
        asset.height = int(self._frames[0].shape[0])
        asset.fps = 24.0
        asset.frame_count = len(self._frames)
        return asset

    def frames(self, asset: TargetVideoAsset) -> Generator[np.ndarray[Any, Any], None, None]:
        del asset
        for frame in self._frames:
            yield frame.copy()


class _FakeEncoder(IVideoEncoder):
    def __init__(self) -> None:
        self.frames: list[np.ndarray[Any, Any]] = []

    def encode(
        self,
        frames: Iterable[np.ndarray[Any, Any]],
        source_asset: TargetVideoAsset,
        output_path: Path,
    ) -> Path:
        del source_asset
        self.frames = list(frames)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")
        return output_path


class _FakeDetector(IFaceDetector):
    def __init__(
        self,
        reference_faces: list[DetectedFace],
        target_batches: list[list[list[DetectedFace]]],
    ) -> None:
        self.reference_faces = reference_faces
        self.target_batches = target_batches
        self.batch_index = 0

    def detect(self, frame: Any, score_threshold: float = 0.5) -> list[DetectedFace]:
        del frame, score_threshold
        return list(self.reference_faces)

    def detect_batch(self, frames: Any, score_threshold: float = 0.5) -> list[list[DetectedFace]]:
        del frames, score_threshold
        batch = self.target_batches[self.batch_index]
        self.batch_index += 1
        return [[face for face in frame_faces] for frame_faces in batch]


class _PassthroughSwapper(IFaceSwapper):
    def swap(self, source_embedding: Any, target_crop: Any) -> Any:
        del source_embedding
        return target_crop

    def swap_batch(self, source_embedding: Any, target_crops: Any) -> Any:
        del source_embedding
        return list(target_crops)


def _make_test_dir() -> Path:
    root = Path.cwd() / "src" / "morphflow_swap_engine" / "tests" / ".tmp_phase4"
    root.mkdir(parents=True, exist_ok=True)
    test_dir = root / str(uuid4())
    test_dir.mkdir()
    return test_dir


def test_affine_face_aligner_produces_deterministic_crop_size() -> None:
    frame = _gradient_frame()
    face = _face((32.0, 24.0, 96.0, 104.0), face_area_ratio=0.31)
    crop_strategy = FaceCropStrategy(
        output_size=512,
        template_name="ffhq",
        margin_ratio=0.12,
        small_face_threshold_ratio=0.035,
    )
    crop_plan = crop_strategy.build_target_plan((frame.shape[1], frame.shape[0]), face)
    aligner = AffineFaceAligner()

    first = aligner.align(frame, face, crop_plan)
    second = aligner.align(frame, face, crop_plan)

    assert first.crop.shape == (512, 512, 3)
    assert np.array_equal(first.crop, second.crop)
    assert np.allclose(first.affine_matrix, second.affine_matrix)
    assert np.allclose(first.inverse_affine_matrix, second.inverse_affine_matrix)


def test_configurable_margin_really_changes_crop_behavior() -> None:
    frame = _gradient_frame()
    face = _face((32.0, 24.0, 96.0, 104.0), face_area_ratio=0.31)
    aligner = AffineFaceAligner()

    tight_plan = FaceCropStrategy(512, "ffhq", 0.05, 0.035).build_target_plan((128, 128), face)
    wide_plan = FaceCropStrategy(512, "ffhq", 0.20, 0.035).build_target_plan((128, 128), face)

    tight = aligner.align(frame, face, tight_plan)
    wide = aligner.align(frame, face, wide_plan)

    assert tight.crop_plan.effective_margin_ratio == 0.05
    assert wide.crop_plan.effective_margin_ratio == 0.20
    assert tight.crop_plan.face_coverage_ratio > wide.crop_plan.face_coverage_ratio
    assert not np.array_equal(tight.crop, wide.crop)


def test_crop_strategy_switches_to_crop_to_swap_for_small_faces() -> None:
    crop_strategy = FaceCropStrategy(
        output_size=512,
        template_name="ffhq",
        margin_ratio=0.12,
        small_face_threshold_ratio=0.035,
    )
    large_face = _face((24.0, 24.0, 104.0, 104.0), face_area_ratio=0.39)
    small_face = _face((56.0, 56.0, 70.0, 70.0), face_area_ratio=0.012)

    large_plan = crop_strategy.build_target_plan((128, 128), large_face)
    small_plan = crop_strategy.build_target_plan((128, 128), small_face)

    assert large_plan.small_face_mode is False
    assert small_plan.small_face_mode is True
    assert small_plan.effective_margin_ratio < large_plan.effective_margin_ratio
    assert small_plan.face_coverage_ratio > large_plan.face_coverage_ratio
    assert "small_face_crop_to_swap" in small_plan.notes


def test_load_config_reads_phase4_alignment_settings() -> None:
    tmp_path = _make_test_dir()
    try:
        ini_path = tmp_path / "engine.ini"
        ini_path.write_text(
            "\n".join(
                [
                    "[engine]",
                    "alignment_crop_size = 384",
                    "alignment_template = arcface_128",
                    "alignment_margin_ratio = 0.18",
                    "alignment_small_face_threshold_ratio = 0.02",
                ]
            ),
            encoding="utf-8",
        )

        config = load_config(ini_path)

        assert config.alignment_crop_size == 384
        assert config.alignment_template == "arcface_128"
        assert config.alignment_margin_ratio == 0.18
        assert config.alignment_small_face_threshold_ratio == 0.02
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_swap_video_use_case_writes_phase4_alignment_artifacts_and_keeps_pipeline_working() -> None:
    tmp_path = _make_test_dir()
    try:
        reference_path = tmp_path / "reference.jpg"
        reference_frame = _gradient_frame()
        cv2.imwrite(str(reference_path), reference_frame)

        frames = [_gradient_frame() for _ in range(3)]
        ref_face = _face((28.0, 20.0, 100.0, 112.0), face_area_ratio=0.40)
        track_faces = [
            _face((28.0, 20.0, 100.0, 112.0), face_area_ratio=0.40, frame_index=0, track_id=1),
            _face((48.0, 48.0, 64.0, 64.0), face_area_ratio=0.015, frame_index=1, track_id=1),
            _face((30.0, 22.0, 102.0, 114.0), face_area_ratio=0.41, frame_index=2, track_id=1),
        ]
        detector = _FakeDetector(
            reference_faces=[ref_face],
            target_batches=[[[track_faces[0]], [track_faces[1]], [track_faces[2]]]],
        )
        encoder = _FakeEncoder()
        config = EngineConfig(
            artifact_dir=str(tmp_path / "debug"),
            save_artifacts=True,
            enable_restoration=False,
            enable_temporal_stabilization=False,
            detector_batch_size=8,
            target_analysis_sample_count=3,
            batch_size=2,
        )

        use_case = SwapVideoUseCase(
            config=config,
            decoder=_FakeDecoder(frames),
            encoder=encoder,
            detector=detector,
            tracker=IOUFaceTracker(),
            track_scorer=TrackScorer(),
            detection_filter=FaceDetectionFilter(),
            reference_analyzer=ReferenceFaceAnalyzer(selector=PrimaryFaceSelector()),
            target_analyzer=TargetVideoAnalyzer(selector=PrimaryFaceSelector(), sample_count=3),
            crop_strategy=FaceCropStrategy(
                output_size=config.alignment_crop_size,
                template_name=config.alignment_template,
                margin_ratio=config.alignment_margin_ratio,
                small_face_threshold_ratio=config.alignment_small_face_threshold_ratio,
            ),
            aligner=AffineFaceAligner(),
            swapper=_PassthroughSwapper(),
            artifact_store=LocalArtifactStore(),
        )

        request = SwapRequest(
            reference_faces=[
                ReferenceFaceAsset(
                    asset_path=reference_path,
                    embedding=np.empty(0, dtype=np.float32),
                )
            ],
            target_asset=TargetVideoAsset(
                asset_path=tmp_path / "target.mp4",
                fps=0.0,
                frame_count=0,
                width=0,
                height=0,
            ),
            output_path=tmp_path / "output.mp4",
        )

        result = use_case.execute(request)

        assert result.success is True
        assert len(encoder.frames) == 3

        debug_root = next((tmp_path / "debug").iterdir())
        alignment_dir = debug_root / "artifacts" / "03_alignment"
        summary = json.loads((alignment_dir / "alignment_summary.json").read_text(encoding="utf-8"))

        assert (alignment_dir / "reference_aligned_crop.jpg").exists()
        assert len(list(alignment_dir.glob("track_1_frame_*_aligned.jpg"))) == 3
        assert summary["reference"]["crop_plan"]["purpose"] == "reference"
        assert len(summary["selected_target_samples"]) == 3
        assert any(sample["crop_plan"]["small_face_mode"] for sample in summary["selected_target_samples"])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
