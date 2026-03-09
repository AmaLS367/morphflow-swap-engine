from __future__ import annotations

import numpy as np

from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.services.detection_filter import FaceDetectionFilter
from morphflow_swap_engine.core.services.primary_face_selector import PrimaryFaceSelector
from morphflow_swap_engine.core.services.reference_face_analyzer import ReferenceFaceAnalyzer
from morphflow_swap_engine.core.services.target_video_analyzer import TargetVideoAnalyzer
from morphflow_swap_engine.infrastructure.detection.insightface_detector import InsightFaceDetector


def _face(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
) -> DetectedFace:
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
        embedding=np.ones(512, dtype=np.float32),
    )


def test_insightface_detector_applies_threshold_filtering() -> None:
    detector = InsightFaceDetector()
    detector._is_prepared = True  # type: ignore[attr-defined]

    class _FakeFace:
        def __init__(self, det_score: float) -> None:
            self.det_score = det_score
            self.bbox = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
            self.kps = np.zeros((5, 2), dtype=np.float32)
            self.embedding = np.ones(512, dtype=np.float32)

    class _FakeApp:
        def get(self, frame: np.ndarray) -> list[_FakeFace]:
            return [_FakeFace(0.9), _FakeFace(0.45)]

    detector.app = _FakeApp()

    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), score_threshold=0.5)

    assert len(detections) == 1
    assert detections[0].score == 0.9


def test_face_detection_filter_removes_small_and_edge_faces() -> None:
    filter_service = FaceDetectionFilter(min_face_size=8, min_face_ratio=0.0002, min_centrality=0.25)
    frame_size = (100, 100)

    kept = _face(25.0, 25.0, 65.0, 65.0, 0.9)
    tiny = _face(1.0, 1.0, 6.0, 6.0, 0.95)
    edge = _face(0.0, 0.0, 20.0, 20.0, 0.8)

    filtered = filter_service.filter_faces([kept, tiny, edge], frame_size)

    assert filtered == [kept]


def test_primary_face_selector_prefers_size_confidence_and_centrality() -> None:
    selector = PrimaryFaceSelector()
    frame_size = (100, 100)

    high_score_edge = _face(0.0, 0.0, 18.0, 18.0, 0.99)
    large_center = _face(25.0, 25.0, 70.0, 70.0, 0.85)
    medium_center = _face(30.0, 30.0, 55.0, 55.0, 0.95)

    selected = selector.select([high_score_edge, large_center, medium_center], frame_size)

    assert selected is large_center


def test_reference_face_analyzer_reports_primary_face_and_warnings() -> None:
    analyzer = ReferenceFaceAnalyzer(selector=PrimaryFaceSelector())
    frame_size = (100, 100)

    primary = _face(25.0, 25.0, 70.0, 70.0, 0.8)
    alternate = _face(75.0, 10.0, 90.0, 25.0, 0.7)

    analysis = analyzer.analyze(frame_size, [primary, alternate], [primary, alternate])

    assert analysis.primary_face is primary
    assert analysis.primary_face_index == 0
    assert analysis.primary_face_box == (25.0, 25.0, 70.0, 70.0)
    assert analysis.primary_face_size_ratio is not None
    assert "reference_multiple_faces_detected" in analysis.warnings


def test_target_video_analyzer_summarizes_sampled_frames() -> None:
    analyzer = TargetVideoAnalyzer(selector=PrimaryFaceSelector(), sample_count=3)
    frame_size = (100, 100)

    frame0 = analyzer.analyze_frame(0, frame_size, [], [])
    frame5_face = _face(25.0, 25.0, 65.0, 65.0, 0.85)
    frame5 = analyzer.analyze_frame(5, frame_size, [frame5_face], [frame5_face])
    frame9_face_a = _face(20.0, 20.0, 55.0, 55.0, 0.7)
    frame9_face_b = _face(60.0, 20.0, 88.0, 48.0, 0.75)
    frame9 = analyzer.analyze_frame(9, frame_size, [frame9_face_a, frame9_face_b], [frame9_face_a, frame9_face_b])

    summary = analyzer.summarize(frame_count=10, sampled_frames=[frame0, frame5, frame9])

    assert analyzer.sample_frame_indices(10) == [0, 4, 9]
    assert summary.sampled_frame_indices == [0, 5, 9]
    assert summary.average_primary_face_size_ratio is not None
    assert "target_sample_frames_without_faces" in summary.warnings
    assert "target_sample_frames_with_multiple_faces" in summary.warnings
