from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .stage_artifact import StageArtifact


@dataclass
class RuntimeReport:
    """Aggregated metrics emitted by a completed engine run."""

    total_frames: int
    total_duration_seconds: float
    avg_fps: float
    artifacts: List[StageArtifact] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
