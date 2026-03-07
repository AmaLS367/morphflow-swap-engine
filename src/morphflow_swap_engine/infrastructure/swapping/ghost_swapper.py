from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from morphflow_swap_engine.core.contracts.i_face_swapper import IFaceSwapper


class GhostSwapper(IFaceSwapper):
    """Ghost Swapper implementation using ONNXRuntime.
    
    Expects 512x512 aligned crops.
    """

    def __init__(
        self,
        model_path: str,
        execution_providers: Optional[List[str]] = None,
        batch_size: int = 1,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.execution_providers = execution_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.session: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def load(self) -> None:
        """Load the ONNX model into memory."""
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for GhostSwapper")
            
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

    def _normalize_crop(self, crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize the crop for the Ghost model.
        Usually Ghost models expect input in range [-1, 1] or [0, 1].
        Assuming [0, 1] normalization with specific mean/std or simple /255.0.
        """
        # OpenCV standard is BGR, ONNX models often expect RGB
        # HWC to CHW
        crop = crop[:, :, ::-1] # BGR to RGB
        crop = crop.transpose((2, 0, 1)) # (3, H, W)
        crop = crop.astype(np.float32) / 255.0
        
        # Standard mean/std normalization for Ghost
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        crop = (crop - mean) / std
        
        return crop

    def _denormalize_crop(self, crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Denormalize the model output back to an image."""
        # Ghost output is often CHW and normalized
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        
        crop = (crop * std) + mean
        crop = np.clip(crop, 0.0, 1.0) * 255.0
        
        # CHW to HWC
        crop = crop.transpose((1, 2, 0))
        # RGB to BGR
        crop = crop[:, :, ::-1]
        return crop.astype(np.uint8)

    def swap(self, source_embedding: Any, target_crop: Any) -> Any:
        """Swap the identity of the target crop."""
        return self.swap_batch(source_embedding, [target_crop])[0]

    def swap_batch(self, source_embedding: Any, target_crops: List[np.ndarray[Any, Any]]) -> List[np.ndarray[Any, Any]]:
        """Swap identity for a batch of crops."""
        if self.session is None:
            self.load()

        emb = np.array(source_embedding, dtype=np.float32)
        if len(emb.shape) == 1:
            emb = np.expand_dims(emb, axis=0)
        
        # Ghost models often expect embedding to match batch size
        if emb.shape[0] == 1 and len(target_crops) > 1:
            emb = np.repeat(emb, len(target_crops), axis=0)

        norm_crops = np.stack([self._normalize_crop(c) for c in target_crops])

        inputs = {
            self._input_names[0]: norm_crops,
            self._input_names[1]: emb
        }
        
        outputs = self.session.run(self._output_names, inputs)
        
        return [self._denormalize_crop(out) for out in outputs[0]]
