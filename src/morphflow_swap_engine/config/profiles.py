from __future__ import annotations

from dataclasses import replace

from .schema import EngineConfig
from ..core.value_objects.engine_profile import EngineProfile

BALANCED = EngineProfile(
    name="balanced",
    enable_restoration=True,
    enable_temporal_stabilization=True,
    swap_model_key="ghost_512",
    detector_score_threshold=0.5,
    use_fp16=True,
)

HIGH_QUALITY = EngineProfile(
    name="high_quality",
    enable_restoration=True,
    enable_temporal_stabilization=True,
    swap_model_key="ghost_512",
    detector_score_threshold=0.4,
    use_fp16=False,
)

THROUGHPUT_MAX = EngineProfile(
    name="throughput_max",
    enable_restoration=False,
    enable_temporal_stabilization=False,
    swap_model_key="simswap_512",
    detector_score_threshold=0.6,
    use_fp16=True,
)

PROFILES = {
    BALANCED.name: BALANCED,
    HIGH_QUALITY.name: HIGH_QUALITY,
    THROUGHPUT_MAX.name: THROUGHPUT_MAX,
}

MODEL_KEY_ALIASES = {
    "ghost_1_256": "ghost_512",
    "hyperswap_1a_256": "ghost_512",
    "hyperswap_1c_256": "ghost_512",
    "inswapper_128_fp16": "simswap_512",
}


def apply_profile(config: EngineConfig, profile_name: str | None = None) -> EngineConfig:
    resolved_name = profile_name or config.profile
    if resolved_name not in PROFILES:
        raise ValueError(f"Unknown engine profile: {resolved_name}")

    profile = PROFILES[resolved_name]
    return replace(
        config,
        profile=profile.name,
        swap_model_key=MODEL_KEY_ALIASES.get(profile.swap_model_key, profile.swap_model_key),
        detector_score_threshold=profile.detector_score_threshold,
        enable_restoration=profile.enable_restoration,
        enable_temporal_stabilization=profile.enable_temporal_stabilization,
        use_fp16=profile.use_fp16,
    )
