from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from morphflow_swap_engine.core.contracts.i_temporal_stabilizer import ITemporalStabilizer


class FilmStabilizer(ITemporalStabilizer):
    """Temporal stabilization using FILM (Frame Interpolation for Large Motion).
    
    This helps reduce flicker by interpolating between the previous stabilized frame
    and the current frame.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        blend_alpha: float = 0.5,
        execution_providers: Optional[List[str]] = None,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.blend_alpha = blend_alpha
        self.execution_providers = execution_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.use_fp16 = use_fp16
        
        self.session: Optional[Any] = None
        self._history: Dict[int, np.ndarray[Any, Any]] = {}
        
        # If no model path is provided, we fall back to EMA (Exponential Moving Average) blending
        self.use_ema_fallback = model_path is None

    def load(self) -> None:
        if self.use_ema_fallback:
            return
            
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for FilmStabilizer")
            
        if self.session is None and self.model_path:
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=self.execution_providers
            )

    def _normalize(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize to [0, 1] RGB CHW."""
        img = image[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        return img.astype(np.float32) / 255.0

    def _denormalize(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Denormalize back to BGR HWC."""
        img = np.clip(image, 0.0, 1.0) * 255.0
        img = img.transpose((1, 2, 0))
        img = img[:, :, ::-1]
        return img.astype(np.uint8)

    def stabilize(self, current_frame: np.ndarray[Any, Any], track_id: int) -> np.ndarray[Any, Any]:
        """Stabilize the current frame using history of the given track_id."""
        if self.session is None and not self.use_ema_fallback:
            self.load()

        if track_id not in self._history:
            self._history[track_id] = current_frame
            return current_frame

        prev_frame = self._history[track_id]

        if self.use_ema_fallback or self.session is None:
            # Fallback: Simple alpha blending
            stabilized = cv2.addWeighted(
                current_frame, self.blend_alpha, 
                prev_frame, 1.0 - self.blend_alpha, 0
            )
        else:
            # FILM Inference
            # Usually FILM takes x0 (prev), x1 (curr), and time (e.g., 0.5 for midpoint)
            x0 = np.expand_dims(self._normalize(prev_frame), axis=0)
            x1 = np.expand_dims(self._normalize(current_frame), axis=0)
            time_input = np.array([[self.blend_alpha]], dtype=np.float32)
            
            inputs = {
                self.session.get_inputs()[0].name: x0,
                self.session.get_inputs()[1].name: x1,
                self.session.get_inputs()[2].name: time_input,
            }
            
            outputs = self.session.run(None, inputs)
            stabilized_norm = outputs[0][0]
            stabilized = self._denormalize(stabilized_norm)

        self._history[track_id] = stabilized
        return stabilized

    def reset(self) -> None:
        """Clear the history."""
        self._history.clear()
