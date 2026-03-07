# morphflow-swap-engine

`morphflow-swap-engine` is MorphFlow's standalone face swap engine.

## Current status

- The active runtime lives under `src/morphflow_swap_engine/`.
- The migration roadmap lives in `MORPHFLOW_SWAP_ENGINE_PLAN.md`.
- The repository exposes only the MorphFlow engine surface.

## Architecture

The codebase follows the modular structure defined in the migration plan:

```text
src/morphflow_swap_engine/
    core/
    application/
    infrastructure/
    adapters/
    config/
    tests/
```

## Running the engine

Use the MorphFlow CLI:

```powershell
python run_morphflow_swap_engine.py --source-face C:\path\source.jpg --target C:\path\target.mp4
python run_morphflow_swap_engine.py --source-face C:\path\source.jpg --target C:\path\target.mp4 --profile high_quality --config morphflow_swap_engine.ini --json
```

After `pip install -e .`, the console script `morphflow-swap-engine` is also available.

The default config template is [`morphflow_swap_engine.ini`](./morphflow_swap_engine.ini).

## Development notes

- `pyproject.toml` is the packaging source of truth.
- Validation should target the MorphFlow package and its tests.
- Model wrappers expect ONNX assets in `models/`.
