from __future__ import annotations

from collections.abc import Generator, Iterable
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from morphflow_swap_engine.config.schema import EngineConfig
from morphflow_swap_engine.core.contracts.i_face_aligner import IFaceAligner
from morphflow_swap_engine.core.contracts.i_face_detector import IFaceDetector
from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper
from morphflow_swap_engine.core.contracts.i_video_decoder import IVideoDecoder
from morphflow_swap_engine.core.contracts.i_video_encoder import IVideoEncoder
from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.entities.reference_face_asset import ReferenceFaceAsset
from morphflow_swap_engine.core.entities.swap_request import SwapRequest
from morphflow_swap_engine.core.entities.target_video_asset import TargetVideoAsset
from morphflow_swap_engine.core.entities.tracked_face_sequence import TrackedFaceSequence
from morphflow_swap_engine.core.services.detection_filter import FaceDetectionFilter
from morphflow_swap_engine.core.services.face_crop_strategy import FaceCropStrategy
from morphflow_swap_engine.core.services.primary_face_selector import PrimaryFaceSelector
from morphflow_swap_engine.core.services.reference_face_analyzer import ReferenceFaceAnalyzer
from morphflow_swap_engine.core.services.target_video_analyzer import TargetVideoAnalyzer
from morphflow_swap_engine.core.services.track_scorer import TrackScorer
from morphflow_swap_engine.core.value_objects.face_alignment_result import FaceAlignmentResult
from morphflow_swap_engine.infrastructure.diagnostics.local_artifact_store import LocalArtifactStore
from morphflow_swap_engine.infrastructure.tracking.iou_tracker import IOUFaceTracker
from morphflow_swap_engine.application.use_cases.swap_video_use_case import SwapVideoUseCase


def _embedding(values: list[float]) -> np.ndarray[Any, Any]:
    return np.array(values, dtype=np.float32)


def _face(
    bbox: tuple[float, float, float, float],
    score: float,
    embedding: np.ndarray[Any, Any],
    *,
    face_area_ratio: float = 0.1,
    centrality: float = 0.8,
) -> DetectedFace:
    x1, y1, x2, y2 = bbox
    return DetectedFace(
        bounding_box=np.array([x1, y1, x2, y2], dtype=np.float32),
        landmark_5=np.array(
            [
                [x1 + 2.0, y1 + 2.0],
                [x2 - 2.0, y1 + 2.0],
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                [x1 + 3.0, y2 - 3.0],
                [x2 - 3.0, y2 - 3.0],
            ],
            dtype=np.float32,
        ),
        score=score,
        face_area_ratio=face_area_ratio,
        centrality=centrality,
        embedding=embedding,
    )


def _track(
    track_id: int,
    face_count: int,
    *,
    score: float,
    area_ratio: float,
    centrality: float,
    total_missed_frames: int,
) -> TrackedFaceSequence:
    faces = [
        _face(
            (10.0 + index, 10.0, 40.0 + index, 40.0),
            score,
            _embedding([1.0, 0.0, 0.0]),
            face_area_ratio=area_ratio,
            centrality=centrality,
        )
        for index in range(face_count)
    ]
    for index, face in enumerate(faces):
        face.frame_index = index
        face.track_id = track_id
    track = TrackedFaceSequence(
        track_id=track_id,
        faces=faces,
        last_seen_frame=face_count - 1,
        total_missed_frames=total_missed_frames,
    )
    track.recompute_aggregates()
    return track


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


class _IdentityAligner(IFaceAligner):
    def align(
        self,
        frame: np.ndarray[Any, Any],
        face: DetectedFace,
        crop_plan: Any,
    ) -> FaceAlignmentResult:
        del face
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return FaceAlignmentResult(
            crop=frame.copy(),
            affine_matrix=identity,
            inverse_affine_matrix=identity,
            crop_plan=crop_plan,
            interpolation="area",
            scale_factor_estimate=1.0,
        )


class _PassthroughSwapper(IFaceSwapper):
    def swap(self, source_embedding: Any, target_crop: Any) -> Any:
        del source_embedding
        return target_crop

    def swap_batch(self, source_embedding: Any, target_crops: Any) -> Any:
        del source_embedding
        return list(target_crops)


def _make_test_dir() -> Path:
    root = Path.cwd() / "src" / "morphflow_swap_engine" / "tests" / ".tmp_runtime"
    root.mkdir(parents=True, exist_ok=True)
    test_dir = root / str(uuid4())
    test_dir.mkdir()
    return test_dir


def _crop_strategy() -> FaceCropStrategy:
    return FaceCropStrategy(
        output_size=512,
        template_name="ffhq",
        margin_ratio=0.12,
        small_face_threshold_ratio=0.035,
    )


def test_tracker_links_adjacent_detections_into_one_track() -> None:
    tracker = IOUFaceTracker()

    first = _face((10.0, 10.0, 50.0, 50.0), 0.9, _embedding([1.0, 0.0, 0.0]))
    second = _face((12.0, 10.0, 52.0, 50.0), 0.92, _embedding([1.0, 0.0, 0.0]))

    tracker.update([first], frame_index=0)
    tracks = tracker.update([second], frame_index=1)

    assert len(tracks) == 1
    assert tracks[0].frame_count == 2
    assert tracks[0].track_id == 1
    assert second.track_id == 1


def test_tracker_recovers_after_short_gap_using_embedding_similarity() -> None:
    tracker = IOUFaceTracker(max_lost_frames=1, reid_window_frames=8, embedding_similarity_threshold=0.35)
    first = _face((10.0, 10.0, 50.0, 50.0), 0.9, _embedding([1.0, 0.0, 0.0]))
    recovered = _face((60.0, 10.0, 100.0, 50.0), 0.91, _embedding([1.0, 0.0, 0.0]))

    tracker.update([first], frame_index=0)
    tracker.update([], frame_index=1)
    tracks = tracker.update([recovered], frame_index=2)

    assert len(tracks) == 1
    assert recovered.track_id == 1
    assert tracks[0].total_missed_frames == 1
    assert tracks[0].missed_frames == 0
    assert tracks[0].is_active is True


def test_tracker_creates_new_track_for_identity_switch() -> None:
    tracker = IOUFaceTracker(embedding_similarity_threshold=0.8)
    first = _face((10.0, 10.0, 50.0, 50.0), 0.9, _embedding([1.0, 0.0, 0.0]))
    switched = _face((70.0, 10.0, 110.0, 50.0), 0.9, _embedding([0.0, 1.0, 0.0]))

    tracker.update([first], frame_index=0)
    tracks = tracker.update([switched], frame_index=1)

    assert len(tracks) == 2
    assert switched.track_id == 2


def test_tracker_does_not_match_when_embedding_similarity_is_too_low() -> None:
    tracker = IOUFaceTracker(embedding_similarity_threshold=0.5)
    first = _face((10.0, 10.0, 50.0, 50.0), 0.9, _embedding([1.0, 0.0, 0.0]))
    other = _face((55.0, 10.0, 95.0, 50.0), 0.9, _embedding([0.0, 1.0, 0.0]))

    tracker.update([first], frame_index=0)
    tracker.update([], frame_index=1)
    tracks = tracker.update([other], frame_index=2)

    assert len(tracks) == 2
    assert other.track_id == 2


def test_get_tracks_include_inactive_returns_finished_tracks() -> None:
    tracker = IOUFaceTracker(max_lost_frames=1, reid_window_frames=2)
    first = _face((10.0, 10.0, 50.0, 50.0), 0.9, _embedding([1.0, 0.0, 0.0]))

    tracker.update([first], frame_index=0)
    tracker.update([], frame_index=1)
    tracker.update([], frame_index=2)

    all_tracks = tracker.get_tracks(include_inactive=True)
    active_tracks = tracker.get_tracks(include_inactive=False)

    assert len(all_tracks) == 1
    assert all_tracks[0].is_active is False
    assert active_tracks == []


def test_track_scorer_prefers_more_stable_track() -> None:
    scorer = TrackScorer()
    stable = _track(1, 5, score=0.8, area_ratio=0.09, centrality=0.8, total_missed_frames=0)
    unstable = _track(2, 5, score=0.8, area_ratio=0.09, centrality=0.8, total_missed_frames=4)

    selected = scorer.find_best_track([stable, unstable])

    assert selected is stable


def test_track_scorer_prevents_size_from_dominating_other_signals() -> None:
    scorer = TrackScorer()
    large_but_weak = _track(1, 4, score=0.55, area_ratio=0.20, centrality=0.35, total_missed_frames=5)
    smaller_but_strong = _track(2, 4, score=0.9, area_ratio=0.09, centrality=0.9, total_missed_frames=0)

    selected = scorer.find_best_track([large_but_weak, smaller_but_strong])

    assert selected is smaller_but_strong


def test_track_scorer_accounts_for_confidence_and_centrality() -> None:
    scorer = TrackScorer()
    weaker = _track(1, 4, score=0.7, area_ratio=0.08, centrality=0.4, total_missed_frames=0)
    stronger = _track(2, 4, score=0.9, area_ratio=0.08, centrality=0.9, total_missed_frames=0)

    selected = scorer.find_best_track([weaker, stronger])

    assert selected is stronger


def test_swap_video_use_case_writes_tracking_manifest_and_sampled_crops() -> None:
    tmp_path = _make_test_dir()
    try:
        reference_path = tmp_path / "reference.jpg"
        cv2.imwrite(str(reference_path), np.zeros((64, 64, 3), dtype=np.uint8))

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        ref_face = _face((12.0, 12.0, 48.0, 48.0), 0.95, _embedding([1.0, 0.0, 0.0]))
        track_a = [
            _face((10.0, 10.0, 42.0, 42.0), 0.9, _embedding([1.0, 0.0, 0.0]), face_area_ratio=0.25, centrality=0.8),
            _face((12.0, 10.0, 44.0, 42.0), 0.9, _embedding([1.0, 0.0, 0.0]), face_area_ratio=0.25, centrality=0.8),
            _face((14.0, 10.0, 46.0, 42.0), 0.9, _embedding([1.0, 0.0, 0.0]), face_area_ratio=0.25, centrality=0.8),
        ]
        detector = _FakeDetector(
            reference_faces=[ref_face],
            target_batches=[
                [[track_a[0], _face((45.0, 12.0, 60.0, 27.0), 0.7, _embedding([0.0, 1.0, 0.0]), face_area_ratio=0.05, centrality=0.3)], [track_a[1]], [track_a[2]]],
            ],
        )
        encoder = _FakeEncoder()
        config = EngineConfig(
            artifact_dir=str(tmp_path / "debug"),
            save_artifacts=True,
            enable_restoration=False,
            enable_temporal_stabilization=False,
            detector_batch_size=10,
            target_analysis_sample_count=3,
            tracker_iou_threshold=0.3,
            tracker_max_lost_frames=5,
            tracker_embedding_similarity_threshold=0.35,
            tracker_reid_window_frames=8,
        )

        use_case = SwapVideoUseCase(
            config=config,
            decoder=_FakeDecoder(frames),
            encoder=encoder,
            detector=detector,
            tracker=IOUFaceTracker(
                iou_threshold=0.3,
                max_lost_frames=5,
                embedding_similarity_threshold=0.35,
                reid_window_frames=8,
            ),
            track_scorer=TrackScorer(),
            detection_filter=FaceDetectionFilter(),
            reference_analyzer=ReferenceFaceAnalyzer(selector=PrimaryFaceSelector()),
            target_analyzer=TargetVideoAnalyzer(selector=PrimaryFaceSelector(), sample_count=3),
            crop_strategy=_crop_strategy(),
            aligner=_IdentityAligner(),
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
        tracking_dir = debug_root / "artifacts" / "02_tracking"
        alignment_dir = debug_root / "artifacts" / "03_alignment"
        assert (tracking_dir / "track_manifest.json").exists()
        assert (tracking_dir / "selected_track.json").exists()
        assert (alignment_dir / "reference_aligned_crop.jpg").exists()
        assert (alignment_dir / "alignment_summary.json").exists()
        assert len(list(alignment_dir.glob("track_1_frame_*_aligned.jpg"))) == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_swap_video_use_case_fails_when_no_usable_track_is_found() -> None:
    tmp_path = _make_test_dir()
    try:
        reference_path = tmp_path / "reference.jpg"
        cv2.imwrite(str(reference_path), np.zeros((64, 64, 3), dtype=np.uint8))

        config = EngineConfig(
            artifact_dir=str(tmp_path / "debug"),
            save_artifacts=False,
            enable_restoration=False,
            enable_temporal_stabilization=False,
        )
        use_case = SwapVideoUseCase(
            config=config,
            decoder=_FakeDecoder([np.zeros((64, 64, 3), dtype=np.uint8)]),
            encoder=_FakeEncoder(),
            detector=_FakeDetector(
                reference_faces=[_face((12.0, 12.0, 48.0, 48.0), 0.95, _embedding([1.0, 0.0, 0.0]))],
                target_batches=[[[]]],
            ),
            tracker=IOUFaceTracker(),
            track_scorer=TrackScorer(),
            detection_filter=FaceDetectionFilter(),
            reference_analyzer=ReferenceFaceAnalyzer(selector=PrimaryFaceSelector()),
            target_analyzer=TargetVideoAnalyzer(selector=PrimaryFaceSelector(), sample_count=1),
            crop_strategy=_crop_strategy(),
            aligner=_IdentityAligner(),
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

        assert result.success is False
        assert "No target face track found in video." in result.error_message
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
