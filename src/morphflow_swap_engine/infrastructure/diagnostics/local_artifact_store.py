from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from morphflow_swap_engine.core.contracts.i_artifact_store import IArtifactStore
from morphflow_swap_engine.core.value_objects.stage_artifact import StageArtifact


class LocalArtifactStore(IArtifactStore):
    """Saves debug artifacts to the local file system."""

    def save(self, name: str, data: Any, base_dir: Path) -> StageArtifact:
        """Persist data (images, JSON, numpy arrays) and return metadata."""
        base_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = base_dir / name
        metadata = {}

        if isinstance(data, np.ndarray):
            if len(data.shape) in (2, 3) and data.dtype == np.uint8:
                # Save image
                cv2.imwrite(str(artifact_path), data)
                metadata["type"] = "image"
            else:
                # Save numpy array
                np.save(str(artifact_path), data)
                metadata["type"] = "numpy"
                artifact_path = artifact_path.with_suffix(".npy")
        elif isinstance(data, (dict, list)):
            # Save JSON
            with open(artifact_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            metadata["type"] = "json"
            artifact_path = artifact_path.with_suffix(".json")
        elif isinstance(data, str):
            # Save text
            with open(artifact_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(data)
            metadata["type"] = "text"
            artifact_path = artifact_path.with_suffix(".txt")
        else:
            # Fallback to string representation
            with open(artifact_path.with_suffix(".log"), "w", encoding="utf-8") as f:
                f.write(str(data))
            metadata["type"] = "log"
            artifact_path = artifact_path.with_suffix(".log")

        metadata["size_bytes"] = artifact_path.stat().st_size
        
        return StageArtifact(
            stage_name=base_dir.name,
            artifact_path=artifact_path,
            metadata=metadata
        )
