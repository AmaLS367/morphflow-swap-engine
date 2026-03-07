from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class StageArtifact:
    """Debug artifact produced by a pipeline stage."""

    stage_name: str
    artifact_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
