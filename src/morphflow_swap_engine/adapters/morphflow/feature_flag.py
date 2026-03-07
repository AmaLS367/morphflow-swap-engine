from __future__ import annotations

import os


def use_new_engine() -> bool:
    """Return True when the new MorphFlow engine should handle the request.

    Controlled by the MORPHFLOW_NEW_ENGINE env var.
    Set to "1" or "true" to enable; anything else keeps the legacy path.
    """
    value = os.environ.get("MORPHFLOW_NEW_ENGINE", "0").strip().lower()
    return value in ("1", "true", "yes")
