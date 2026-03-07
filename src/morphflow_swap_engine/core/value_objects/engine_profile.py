from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineProfile:
    """Immutable runtime profile controlling quality/speed trade-offs."""

    name: str
    enable_restoration: bool
    enable_temporal_stabilization: bool
    swap_model_key: str
    detector_score_threshold: float
    use_fp16: bool
