from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReferenceFaceAsset:
    """A reference face used as the identity source for swapping."""

    asset_path: Path
    embedding: NDArray[np.float32]
    label: str = ""
    normed_embedding: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
