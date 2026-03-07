from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper


class OnnxSwapper(IFaceSwapper):
    """Generic Swapper implementation for ONNX models (SimSwap++, etc.)."""

    def __init__(
        self,
        model_path: str,
        execution_providers: Optional[List[str]] = None,
        batch_size: int = 1,
        use_fp16: bool = True,
        input_mean: float = 0.5,
        input_std: float = 0.5,
    ):
        self.model_path = model_path
        self.execution_providers = execution_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.input_mean = input_mean
        self.input_std = input_std
        
        self.session: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def load(self) -> None:
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for OnnxSwapper")
            
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
        crop = crop[:, :, ::-1].transpose((2, 0, 1)) # BGR to RGB, HWC to CHW
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.input_mean) / self.input_std
        return crop

    def _denormalize(self, crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        crop = (crop * self.input_std) + self.input_mean
        crop = np.clip(crop, 0.0, 1.0) * 255.0
        crop = crop.transpose((1, 2, 0))[:, :, ::-1] # CHW to HWC, RGB to BGR
        return crop.astype(np.uint8)

    def swap(self, source_embedding: Any, target_crop: Any) -> Any:
        if self.session is None:
            self.load()

        emb = np.array(source_embedding, dtype=np.float32)
        if len(emb.shape) == 1:
            emb = np.expand_dims(emb, axis=0)

        norm_crop = np.expand_dims(self._normalize(target_crop), axis=0)

        inputs = {
            self._input_names[0]: norm_crop,
            self._input_names[1]: emb
        }
        
        outputs = self.session.run(self._output_names, inputs)
        return self._denormalize(outputs[0][0])
