# MorphFlow Fork Changelog vs FaceFusion

## Comparison Basis

- Upstream repository: `https://github.com/facefusion/facefusion`
- Upstream HEAD at comparison time: `c7976ec9d4de0a6bc5b499fc8ba3df6b850349cb`
- Local baseline tag: `baseline-facefusion-old`
- Local baseline tag target: `c7976ec9d4de0a6bc5b499fc8ba3df6b850349cb`
- Current fork HEAD: `4adea76`

This means the local baseline tag and the current upstream HEAD point to the same FaceFusion commit. The changelog below therefore describes the exact delta between this fork and the current upstream repository at comparison time.

## Summary

- Files changed versus upstream: `110`
- Insertions versus upstream: `2990`
- Deletions versus upstream: `264`
- Primary fork direction: preserve the legacy `facefusion/` runtime while building a new `src/morphflow_swap_engine/` package beside it

## What Exists In This Fork That Does Not Exist Upstream

### 1. New modular engine package

The fork adds a new package under `src/morphflow_swap_engine/` with a clean split between:

- `core`
- `application`
- `infrastructure`
- `adapters`
- `config`
- `tests`

This package does not exist in upstream FaceFusion.

### 2. New domain model and contracts

The fork adds typed engine-facing abstractions that upstream does not provide in this shape:

- entities such as `SwapRequest`, `SwapResult`, `DetectedFace`, `TrackedFaceSequence`
- contracts such as `IFaceDetector`, `IFaceTracker`, `IFaceSwapper`, `IFaceRestorer`, `ITemporalStabilizer`, `IVideoDecoder`, `IVideoEncoder`, `IArtifactStore`
- value objects such as `EngineProfile`, `RuntimeReport`, `StageArtifact`

This is the architectural base for swapping components without rewriting the whole engine.

### 3. MorphFlow adapter layer

The fork adds an explicit integration surface under `src/morphflow_swap_engine/adapters/morphflow/`:

- `feature_flag.py`
- `request_mapper.py`
- `response_mapper.py`
- `adapter.py`

This adapter is the bridge back into MorphFlow and is not present upstream.

### 4. Modernized swap-engine pipeline components

The fork adds new infrastructure modules for the planned replacement stack:

- detection: `InsightFaceDetector`
- tracking: `IOUFaceTracker`
- alignment: `AffineFaceAligner`
- swapping: `GhostSwapper`, `OnnxSwapper`, `SimSwapSwapper`
- restoration: `CodeFormerRestorer`, `apply_color_transfer`
- temporal: `FilmStabilizer`
- video: OpenCV decoder and encoder
- diagnostics: `LocalArtifactStore`

Upstream FaceFusion has its own runtime path, but not this separate modular implementation under `src/`.

### 5. New orchestration use case

The fork adds `SwapVideoUseCase`, which wires together:

- reference extraction
- detection and tracking
- track scoring
- alignment
- batched swapping
- optional restoration
- optional temporal stabilization
- reconstruction
- debug artifact output

This orchestration layer does not exist upstream in this form.

### 6. Runtime profile system for the new engine

The fork adds an explicit profile layer for the new package:

- `balanced`
- `high_quality`
- `throughput_max`

The adapter now applies these profiles to actual runtime component selection and stage toggles.

### 7. Diagnostics and artifact reporting for the new engine

The fork adds a separate debug artifact mechanism for the new engine:

- artifact storage under `storage/debug`
- artifact manifest output
- runtime report output
- stage-level artifacts for detection, tracking, swap, restoration, temporal, and reconstruction

This diagnostics surface is part of the migration plan and is not part of upstream in this structure.

## Fork-Level Repository Changes vs Upstream

### 1. Rebrand from FaceFusion to MorphFlow surface

The fork renames the project-facing entrypoints:

- `facefusion.py` -> `morphflow_swap_engine.py`
- `facefusion.ini` -> `morphflow_swap_engine.ini`

It also rewrites repository-facing metadata and docs around the MorphFlow identity.

### 2. Project roadmap added

The fork adds [MORPHFLOW_SWAP_ENGINE_PLAN.md](./MORPHFLOW_SWAP_ENGINE_PLAN.md), which defines:

- migration phases
- target stack
- commit sequencing
- acceptance direction for the new engine

There is no equivalent project-specific migration plan in upstream FaceFusion.

### 3. Packaging and tooling moved to `pyproject.toml`

The fork adds `pyproject.toml` and removes legacy top-level files:

- removed: `requirements.txt`
- removed: `mypy.ini`
- removed: `.flake8`
- removed: `.coveragerc`
- added: `pyproject.toml`

This is one of the clearest repo-level divergences from upstream.

### 4. CI path changed

The fork modifies `.github/workflows/ci.yml` to validate:

- editable install from `pyproject.toml`
- `ruff`
- `mypy`
- smoke tests for the new engine package

This diverges from upstream CI because the fork now validates both the preserved legacy path and the new engine package.

## Legacy FaceFusion Surface Changed In The Fork

The fork intentionally keeps `facefusion/` but edits selected files to support the migration and rebrand. Compared with upstream, local modifications exist in files such as:

- `facefusion/audio.py`
- `facefusion/face_masker.py`
- `facefusion/memory.py`
- `facefusion/metadata.py`
- `facefusion/program.py`
- several UI files under `facefusion/uis/`

These are not a full rewrite. They are selective compatibility and branding changes while the legacy runtime remains available as fallback.

## What Is Still Inherited From Upstream

- The `facefusion/` package remains in the fork.
- Most of the legacy runtime behavior still comes from upstream FaceFusion.
- The new engine is additive, not a wholesale replacement yet.
- The fork is still in migration mode rather than final-state replacement mode.

## Commit Timeline In The Fork Since Upstream Baseline

1. `a2b7aa8` `docs: rebrand repository as morphflow swap engine`
2. `487abe2` `refactor: rename runtime entrypoint and default config`
3. `6d22a49` `chore: remove upstream repo assets and stale branding hooks`
4. `c6729b6` `chore: complete upstream branding hook cleanup`
5. `881b98a` `chore: update morphflow project metadata`
6. `c442b86` `chore: modernize development tooling and ci`
7. `a6fb31e` `test: repoint cli smoke coverage to morphflow swap engine`
8. `291e401` `add morphflow-swap-engine package skeleton`
9. `e92eb3d` `chore: update dependency management and remove legacy files`
10. `c4be60b` `feat: add core entities`
11. `e6909d4` `feat: add core value objects`
12. `e8f464b` `feat: add core contracts`
13. `3852f5c` `feat: add config layer (schema, profiles, loader)`
14. `680986a` `feat: add morphflow adapter (request/response mappers, feature flag)`
15. `24dd021` `test: add engine import smoke tests and extend CI with mypy + new test step`
16. `b940530` `feat: add detector - InsightFace buffalo_l behind IFaceDetector`
17. `d880f23` `feat: add target face tracking and selection strategy`
18. `2458606` `feat: implement core infrastructure modules (alignment, swapping, restoration, temporal, video)`
19. `3af7718` `feat: implement SwapVideoUseCase and wire it in MorphFlowAdapter`
20. `dccdbea` `feat: implement Phase 10 - Diagnostics and Artifact Store`
21. `776a6c8` `feat: implement Color Transfer for better visual blending`
22. `94f1805` `feat: implement Phase 5 - OnnxSwapper for fallback models like SimSwap++`
23. `92e278e` `feat: implement Phase 9 - Batched Inference for performance optimization`
24. `263591d` `feat: implement SimSwap++ swapper as requested in Phase 5`
25. `bf6ddef` `feat: apply runtime profiles in morphflow adapter`
26. `d5efbcc` `feat: add stable face track state and scoring`
27. `4adea76` `fix: add pipeline diagnostics and typed runtime plumbing`

## Practical Reading

If someone asks "what does this fork have that upstream FaceFusion does not?", the shortest accurate answer is:

1. A new modular swap-engine package under `src/morphflow_swap_engine/`
2. A MorphFlow-specific adapter and feature-flagged migration path
3. Modernized packaging and CI around `pyproject.toml`
4. A staged roadmap and diagnostics path for replacing the legacy face swap stack

