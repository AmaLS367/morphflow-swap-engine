from __future__ import annotations

from typing import Any, Dict

from ...core.entities.swap_result import SwapResult


def map_response(result: SwapResult) -> Dict[str, Any]:
    """Convert a SwapResult into a MorphFlow API response dict."""
    return {
        "success": result.success,
        "output_path": str(result.output_path),
        "frames_processed": result.frames_processed,
        "duration_seconds": result.duration_seconds,
        "error_message": result.error_message,
    }
