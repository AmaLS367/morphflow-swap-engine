from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..value_objects.stage_artifact import StageArtifact


class IArtifactStore(ABC):
    """Persists debug artifacts produced by pipeline stages."""

    @abstractmethod
    def save(self, name: str, data: Any, base_dir: Path) -> StageArtifact:
        """Persist data and return a StageArtifact describing the saved file."""
        ...
