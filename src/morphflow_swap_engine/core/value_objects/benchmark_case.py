from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkCase:
    """Specification for a single benchmark run case."""

    case_id: str
    source_face_path: Path
    target_video_path: Path
    expected_output_path: Path
    notes: str = ""
