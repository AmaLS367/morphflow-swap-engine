from __future__ import annotations

from morphflow_swap_engine.core.entities.detected_face import DetectedFace
from morphflow_swap_engine.core.value_objects.face_crop_plan import FaceCropPlan


class FaceCropStrategy:
    """Derives deterministic crop plans for reference and tracked target faces."""

    def __init__(
        self,
        output_size: int,
        template_name: str,
        margin_ratio: float,
        small_face_threshold_ratio: float,
    ) -> None:
        self.output_size = output_size
        self.template_name = template_name
        self.margin_ratio = max(0.0, min(margin_ratio, 0.3))
        self.small_face_threshold_ratio = max(0.0, small_face_threshold_ratio)

    def build_reference_plan(
        self,
        frame_size: tuple[int, int],
        face: DetectedFace,
    ) -> FaceCropPlan:
        return self._build_plan("reference", frame_size, face)

    def build_target_plan(
        self,
        frame_size: tuple[int, int],
        face: DetectedFace,
    ) -> FaceCropPlan:
        track_id = face.track_id if face.track_id >= 0 else None
        return self._build_plan("target", frame_size, face, track_id=track_id)

    def _build_plan(
        self,
        purpose: str,
        frame_size: tuple[int, int],
        face: DetectedFace,
        track_id: int | None = None,
    ) -> FaceCropPlan:
        frame_width, frame_height = frame_size
        face_area_ratio = self._resolve_face_area_ratio(frame_size, face)
        effective_margin_ratio = self.margin_ratio
        notes: list[str] = []

        if self.small_face_threshold_ratio > 0.0 and 0.0 < face_area_ratio < self.small_face_threshold_ratio:
            ratio = max(face_area_ratio / self.small_face_threshold_ratio, 0.2)
            effective_margin_ratio *= ratio
            notes.append("small_face_crop_to_swap")

        face_coverage_ratio = max(0.2, 1.0 - (2.0 * effective_margin_ratio))

        return FaceCropPlan(
            purpose=purpose,
            output_size=self.output_size,
            template_name=self.template_name,
            configured_margin_ratio=self.margin_ratio,
            effective_margin_ratio=effective_margin_ratio,
            face_coverage_ratio=face_coverage_ratio,
            frame_width=frame_width,
            frame_height=frame_height,
            face_area_ratio=face_area_ratio,
            track_id=track_id,
            small_face_mode=effective_margin_ratio < self.margin_ratio,
            notes=tuple(notes),
        )

    @staticmethod
    def _resolve_face_area_ratio(frame_size: tuple[int, int], face: DetectedFace) -> float:
        if face.face_area_ratio > 0.0:
            return face.face_area_ratio

        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            return 0.0

        x1, y1, x2, y2 = [float(value) for value in face.bounding_box]
        face_width = max(0.0, x2 - x1)
        face_height = max(0.0, y2 - y1)
        return float((face_width * face_height) / float(frame_width * frame_height))
