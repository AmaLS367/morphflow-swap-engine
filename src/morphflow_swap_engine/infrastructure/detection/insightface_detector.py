from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import insightface
from insightface.app import FaceAnalysis

from morphflow_swap_engine.core.contracts.i_face_detector import IFaceDetector
from morphflow_swap_engine.core.entities.detected_face import DetectedFace


class InsightFaceDetector(IFaceDetector):
    """Implementation of IFaceDetector using InsightFace buffalo_l (SCRFD)."""

    def __init__(
        self,
        name: str = "buffalo_l",
        root: str = "~/.insightface",
        providers: Optional[List[str]] = None,
        det_size: tuple[int, int] = (640, 640),
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        self.app = FaceAnalysis(name=name, root=root, providers=providers)
        self.det_size = det_size
        self._is_prepared = False

    def _prepare(self) -> None:
        if not self._is_prepared:
            # ctx_id=0 for first GPU
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            self._is_prepared = True

    def detect(self, frame: np.ndarray[Any, Any], score_threshold: float = 0.5) -> List[DetectedFace]:
        """Detect faces in a frame using InsightFace."""
        self._prepare()
        
        # insightface expects BGR image (opencv default)
        faces = self.app.get(frame)
        
        results = []
        for face in faces:
            if face.det_score < score_threshold:
                continue
                
            detected_face = DetectedFace(
                bounding_box=face.bbox,
                landmark_5=face.kps,
                score=float(face.det_score),
                embedding=face.embedding if hasattr(face, "embedding") else np.empty(0, dtype=np.float32)
            )
            results.append(detected_face)
            
        return results
