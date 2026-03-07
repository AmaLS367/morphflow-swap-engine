"""Smoke tests — verify the package and all sub-modules are importable."""
from __future__ import annotations


def test_package_importable() -> None:
    import morphflow_swap_engine

    assert morphflow_swap_engine.__version__ == "0.1.0"


def test_config_importable() -> None:
    from morphflow_swap_engine.config import BALANCED, HIGH_QUALITY, THROUGHPUT_MAX, EngineConfig, load_config

    assert BALANCED.name == "balanced"
    assert HIGH_QUALITY.name == "high_quality"
    assert THROUGHPUT_MAX.name == "throughput_max"
    assert isinstance(load_config(), EngineConfig)


def test_contracts_importable() -> None:
    from morphflow_swap_engine.core.contracts import (
        IArtifactStore,
        IBenchmarkRunner,
        IFaceAligner,
        IFaceDetector,
        IFaceRestorer,
        IFaceSwapper,
        IFaceTracker,
        ITemporalStabilizer,
        IVideoDecoder,
        IVideoEncoder,
    )

    for contract in (
        IArtifactStore,
        IBenchmarkRunner,
        IFaceAligner,
        IFaceDetector,
        IFaceRestorer,
        IFaceSwapper,
        IFaceTracker,
        ITemporalStabilizer,
        IVideoDecoder,
        IVideoEncoder,
    ):
        assert hasattr(contract, "__abstractmethods__")


def test_adapter_importable() -> None:
    from morphflow_swap_engine.adapters.morphflow import MorphFlowAdapter, use_new_engine

    assert use_new_engine() is True
    assert MorphFlowAdapter is not None
