from __future__ import annotations

from typing import Any, Dict

from .feature_flag import use_new_engine
from .request_mapper import map_request
from .response_mapper import map_response
from ...core.entities.swap_result import SwapResult


class MorphFlowAdapter:
    """Entry point that routes MorphFlow API calls to the correct engine.

    When the MORPHFLOW_NEW_ENGINE feature flag is off, the adapter raises
    NotImplementedError so the caller falls back to the legacy FaceFusion path.
    When the flag is on, the adapter delegates to the new engine pipeline
    (to be wired up in Phase 6).
    """

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a swap request payload and return a response dict."""
        if not use_new_engine():
            raise NotImplementedError(
                "New engine is disabled. Set MORPHFLOW_NEW_ENGINE=1 to enable."
            )

        request = map_request(payload)

        # Pipeline execution will be injected here in Phase 6.
        # For now return a stub result so the adapter layer is importable.
        result = SwapResult(
            output_path=request.target_asset.asset_path,
            frames_processed=0,
            duration_seconds=0.0,
            success=False,
            error_message="Pipeline not yet implemented (Phase 6).",
        )

        return map_response(result)
