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
        """Swap the identity of the target crop.
        
        Args:
            source_embedding: Identity embedding of the source face.
            target_crop: The aligned target face crop.
            
        Returns:
            The swapped face crop.
        """
        if self.session is None:
            self.load()

        # Ensure embedding is correct shape. Some Ghost variants need specific shapes.
        # Assuming embedding is (1, 512)
        emb = np.array(source_embedding, dtype=np.float32)
        if len(emb.shape) == 1:
            emb = np.expand_dims(emb, axis=0)

        # Normalize target crop
        norm_crop = self._normalize_crop(target_crop)
        norm_crop = np.expand_dims(norm_crop, axis=0) # Add batch dimension

        if self.use_fp16:
            # Note: For FP16 inference, the model itself should be cast to FP16,
            # or we cast inputs if the graph expects FP16.
            pass

        # Run inference
        # The input names for Ghost are typically "target" and "source" 
        # but we use _input_names to be safer.
        # Usually: index 0 is target image, index 1 is source embedding
        inputs = {
            self._input_names[0]: norm_crop,
            self._input_names[1]: emb
        }
        
        outputs = self.session.run(self._output_names, inputs)
        
        # Output is usually at index 0, with shape (1, 3, H, W)
        swapped_crop = outputs[0][0]
        
        # Denormalize
        return self._denormalize_crop(swapped_crop)
