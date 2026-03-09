from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .face_crop_plan import FaceCropPlan


@dataclass(frozen=True)
class FaceAlignmentResult:
    """Aligned crop plus transform metadata for reintegration and diagnostics."""

    crop: NDArray[np.uint8]
    affine_matrix: NDArray[np.float32]
    inverse_affine_matrix: NDArray[np.float32]
    crop_plan: FaceCropPlan
    interpolation: str
    scale_factor_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_shape": [int(value) for value in self.crop.shape],
            "affine_matrix": self.affine_matrix.tolist(),
            "inverse_affine_matrix": self.inverse_affine_matrix.tolist(),
            "crop_plan": self.crop_plan.to_dict(),
            "interpolation": self.interpolation,
            "scale_factor_estimate": self.scale_factor_estimate,
        }
