from __future__ import annotations

from .onnx_swapper import OnnxSwapper


class SimSwapSwapper(OnnxSwapper):
    """Specialized SimSwap++ swapper."""

    def __init__(
        self,
        model_path: str,
        **kwargs
    ):
        # SimSwap++ often uses specific normalization (mean=0.5, std=0.5)
        # but it can vary. We use defaults from OnnxSwapper.
        super().__init__(model_path=model_path, **kwargs)
