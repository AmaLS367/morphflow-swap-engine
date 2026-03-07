from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SwapResult:
    """Outcome of a completed face swap job."""

    output_path: Path
    frames_processed: int
    duration_seconds: float
    success: bool
    error_message: str = ""
