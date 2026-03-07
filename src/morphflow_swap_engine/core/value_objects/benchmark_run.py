from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .benchmark_case import BenchmarkCase


@dataclass
class BenchmarkRun:
    """Result of executing a BenchmarkCase."""

    case: BenchmarkCase
    engine_profile: str
    psnr: float
    ssim: float
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
