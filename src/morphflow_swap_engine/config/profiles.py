from __future__ import annotations

from ..core.value_objects.engine_profile import EngineProfile

BALANCED = EngineProfile(
    name="balanced",
    enable_restoration=True,
    enable_temporal_stabilization=True,
    swap_model_key="hyperswap_1a_256",
    detector_score_threshold=0.5,
    use_fp16=True,
)

HIGH_QUALITY = EngineProfile(
    name="high_quality",
    enable_restoration=True,
    enable_temporal_stabilization=True,
    swap_model_key="hyperswap_1c_256",
    detector_score_threshold=0.4,
    use_fp16=False,
)

THROUGHPUT_MAX = EngineProfile(
    name="throughput_max",
    enable_restoration=False,
    enable_temporal_stabilization=False,
    swap_model_key="inswapper_128_fp16",
    detector_score_threshold=0.6,
    use_fp16=True,
)

PROFILES = {
    BALANCED.name: BALANCED,
    HIGH_QUALITY.name: HIGH_QUALITY,
    THROUGHPUT_MAX.name: THROUGHPUT_MAX,
}
