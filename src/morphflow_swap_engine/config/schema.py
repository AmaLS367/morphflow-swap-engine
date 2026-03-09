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
    detector_batch_size: int = 8
    detector_min_face_size: int = 8
    detector_min_face_ratio: float = 0.0002
    detector_min_centrality: float = 0.0
    target_analysis_sample_count: int = 5

    # Swap model
    swap_model_key: str = "ghost_512"

    # Optional stages (can be toggled per profile)
    enable_restoration: bool = True
    enable_temporal_stabilization: bool = True

    # Inference
    use_fp16: bool = True
    execution_providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    batch_size: int = 1

    # I/O
    output_dir: str = "output"
    artifact_dir: str = "storage/debug"
    save_artifacts: bool = False
