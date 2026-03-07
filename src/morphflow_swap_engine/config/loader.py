from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional

from .schema import EngineConfig


def load_config(ini_path: Optional[Path] = None) -> EngineConfig:
    """Load EngineConfig from an INI file, falling back to defaults.

    If ini_path is None or the file does not exist, the default EngineConfig
    is returned unchanged.
    """
    cfg = EngineConfig()

    if ini_path is None or not ini_path.exists():
        return cfg

    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")

    section = "engine"
    if not parser.has_section(section):
        return cfg

    def _get(key: str, fallback: str) -> str:
        return parser.get(section, key, fallback=fallback)

    cfg.profile = _get("profile", cfg.profile)
    cfg.detector_score_threshold = float(_get("detector_score_threshold", str(cfg.detector_score_threshold)))
    cfg.detector_iou_threshold = float(_get("detector_iou_threshold", str(cfg.detector_iou_threshold)))
    cfg.swap_model_key = _get("swap_model_key", cfg.swap_model_key)
    cfg.enable_restoration = _get("enable_restoration", str(cfg.enable_restoration)).lower() == "true"
    cfg.enable_temporal_stabilization = _get("enable_temporal_stabilization", str(cfg.enable_temporal_stabilization)).lower() == "true"
    cfg.use_fp16 = _get("use_fp16", str(cfg.use_fp16)).lower() == "true"
    cfg.batch_size = int(_get("batch_size", str(cfg.batch_size)))
    cfg.output_dir = _get("output_dir", cfg.output_dir)
    cfg.artifact_dir = _get("artifact_dir", cfg.artifact_dir)
    cfg.save_artifacts = _get("save_artifacts", str(cfg.save_artifacts)).lower() == "true"

    raw_providers = _get("execution_providers", "")
    if raw_providers:
        cfg.execution_providers = [p.strip() for p in raw_providers.split(",") if p.strip()]

    return cfg
