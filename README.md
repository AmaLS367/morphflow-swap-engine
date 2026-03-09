# morphflow-swap-engine

`morphflow-swap-engine` is the active MorphFlow face swap runtime.

This repository no longer carries the preserved legacy `facefusion/` baseline. The
live product surface is `src/morphflow_swap_engine/`, and the migration roadmap in
`MORPHFLOW_SWAP_ENGINE_PLAN.md` is now used as architecture and sequencing context.

## What exists today

The current engine already wires an end-to-end video pipeline:

1. probe target video
2. detect the reference face and extract its embedding
3. detect faces across target frames
4. track faces and select the dominant target track
5. align face crops
6. run batched face swap
7. optionally run restoration
8. optionally run temporal stabilization
9. paste the swapped crop back into the frame
10. encode the processed video

The default runtime is assembled through the MorphFlow adapter and currently uses:

- detector: InsightFace `buffalo_l`
- tracker: IOU-based face tracker
- aligner: affine 5-point alignment with FFHQ-style 512 crop
- swapper: Ghost 512, with SimSwap as fallback profile option
- restorer: CodeFormer
- temporal stabilizer: FILM wrapper
- video I/O: OpenCV decoder and encoder
- diagnostics: local artifact store under `storage/debug/`

## Repository layout

```text
src/morphflow_swap_engine/
    core/            domain entities, value objects, contracts, pure services
    application/     use cases and orchestration
    infrastructure/  model wrappers, video I/O, tracking, diagnostics
    adapters/        CLI and MorphFlow-facing adapter surface
    config/          config schema, INI loader, runtime profiles
    tests/           package smoke tests and CLI parsing tests
```

## Current limitations

The repository is beyond the skeleton phase, but it is not feature-complete relative
to the full roadmap.

- The pipeline is real, but test coverage is still shallow and focused on imports and CLI parsing.
- `use_fp16` and batching are exposed in config, but the 5090 optimization phase is not fully realized yet.
- Video encoding is currently OpenCV-based and does not preserve source audio.
- The MorphFlow feature flag is effectively always on in the current adapter surface.
- Model files are expected to exist locally under `models/`; the repo does not fetch them.

## Installation

```powershell
pip install -e .[dev]
```

Python `>=3.10` is required. Packaging and tool configuration live in `pyproject.toml`.

## Running the engine

Use the local runner:

```powershell
python run_morphflow_swap_engine.py --source-face C:\path\source.jpg --target C:\path\target.mp4
python run_morphflow_swap_engine.py --source-face C:\path\source.jpg --target C:\path\target.mp4 --profile high_quality --config morphflow_swap_engine.ini --json
```

After installation, the console script is also available:

```powershell
morphflow-swap-engine --source-face C:\path\source.jpg --target C:\path\target.mp4
```

Available profiles at the moment:

- `balanced`
- `high_quality`
- `throughput_max`

The default config template is `morphflow_swap_engine.ini`.

## Validation

Use the repository validation path:

```powershell
ruff check .
mypy src/morphflow_swap_engine
pytest src/morphflow_swap_engine/tests -q
```

## Notes for contributors

- Treat `src/morphflow_swap_engine/` as the source of truth.
- Preserve the clean architecture split: `core` is pure, `application` orchestrates, `infrastructure` wraps external tools, `adapters` expose entrypoints.
- When docs disagree with code, trust the live repo state over stale prose.
