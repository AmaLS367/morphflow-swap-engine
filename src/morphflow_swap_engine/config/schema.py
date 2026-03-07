from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EngineConfig:
    """Top-level configuration for the MorphFlow swap engine."""

    # Active profile name: "balanced" | "high_quality" | "throughput_max"
    profile: str = "balanced"

    # Face detection
    detector_score_threshold: float = 0.5
    detector_iou_threshold: float = 0.4

    # Swap model
    swap_model_key: str = "hyperswap_1a_256"

    # Optional stages (can be toggled per profile)
    enable_restoration: bool = True
    enable_temporal_stabilization: bool = True

    # Inference
    use_fp16: bool = True
    execution_providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    batch_size: int = 1

    # I/O
    output_dir: str = "output"
    artifact_dir: str = "artifacts"
    save_artifacts: bool = False
