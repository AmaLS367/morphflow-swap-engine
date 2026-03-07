from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from morphflow_swap_engine.core.contracts.i_face_restorer import IFaceRestorer


class CodeFormerRestorer(IFaceRestorer):
    """Face restoration using CodeFormer.
    
    Usually expects 512x512 aligned crops and outputs 512x512 enhanced crops.
    """

    def __init__(
        self,
        model_path: str,
        fidelity_weight: float = 0.7,
        execution_providers: Optional[List[str]] = None,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.fidelity_weight = fidelity_weight
        self.execution_providers = execution_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.use_fp16 = use_fp16
        self.session: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def load(self) -> None:
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for CodeFormerRestorer")
            
        if self.session is None:
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=self.execution_providers
            )
            self._input_names = [inp.name for inp in self.session.get_inputs()]
            self._output_names = [out.name for out in self.session.get_outputs()]

    def _normalize(self, crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize BGR image to expected model input.
        CodeFormer usually expects RGB [0, 1] or [-1, 1].
        Assuming [0, 1] for ONNX version often.
        """
        crop = crop[:, :, ::-1] # BGR to RGB
        crop = crop.transpose((2, 0, 1)) # (3, H, W)
        crop = crop.astype(np.float32) / 255.0
        
        # Standard normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        crop = (crop - mean) / std
        return crop

    def _denormalize(self, crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Denormalize back to BGR."""
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        
        crop = (crop * std) + mean
        crop = np.clip(crop, 0.0, 1.0) * 255.0
        
        crop = crop.transpose((1, 2, 0)) # CHW to HWC
        crop = crop[:, :, ::-1] # RGB to BGR
        return crop.astype(np.uint8)

    def restore(self, swapped_crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Restore and enhance the swapped face crop."""
        if self.session is None:
            self.load()

        # Some CodeFormer ONNX models require resize to 512x512 if not already
        original_shape = swapped_crop.shape[:2]
        if original_shape != (512, 512):
            crop = cv2.resize(swapped_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            crop = swapped_crop

        norm_crop = self._normalize(crop)
        norm_crop = np.expand_dims(norm_crop, axis=0) # Add batch dimension

        # Run inference
        # CodeFormer might take a fidelity weight as input `w`
        inputs = {self._input_names[0]: norm_crop}
        
        if len(self._input_names) > 1 and "w" in self._input_names[1].lower():
            # Pass fidelity weight if the model expects it
            inputs[self._input_names[1]] = np.array([self.fidelity_weight], dtype=np.float64).astype(np.float32)

        outputs = self.session.run(self._output_names, inputs)
        
        restored_crop = outputs[0][0]
        result = self._denormalize(restored_crop)
        
        if original_shape != (512, 512):
            result = cv2.resize(result, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

        return result
