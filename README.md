# morphflow-swap-engine

`morphflow-swap-engine` is the working repository for MorphFlow's next face swap engine.

This repository is intentionally starting from a preserved legacy baseline so the team can compare the old pipeline against the new engine as it is built. The goal of this phase is cleanup and adaptation, not a full rewrite of the runtime.

## Current status

- The current runtime is still the inherited legacy engine internals under `facefusion/`.
- The repository surface is being reworked to match the MorphFlow project identity.
- The detailed execution roadmap lives in `MORPHFLOW_SWAP_ENGINE_PLAN.md`.
- The untouched upstream baseline is tagged as `baseline-facefusion-old`.

## Project direction

This repository exists to support the phased migration described in `MORPHFLOW_SWAP_ENGINE_PLAN.md`.

Immediate priorities:

1. clean the inherited repository surface
2. preserve a runnable comparison baseline
3. prepare a stable base for the new modular swap engine

## Legacy baseline

The current baseline remains valuable for comparison runs and for reusing pieces of the old pipeline during migration.

- Internal package namespace is still `facefusion` in this phase.
- The baseline CLI and config surface are still inherited and will be renamed during the cleanup commits that follow.
- The license remains unchanged.

## Running the current baseline

Use the renamed MorphFlow entrypoint for the preserved legacy baseline:

```powershell
python morphflow_swap_engine.py --help
python install.py --help
```

The default config template is `morphflow_swap_engine.ini`.

## Development notes

- Use `requirements.txt` for runtime dependencies.
- Cleanup-phase tooling is being modernized with a lighter CI path and updated local lint/type-check commands.
- Heavy benchmark and model replacement work starts only after this repository cleanup phase is complete.

## Next step after cleanup

The next implementation phase is to create the `morphflow_swap_engine` foundation described in Phase 1 of `MORPHFLOW_SWAP_ENGINE_PLAN.md`, while keeping the legacy runtime callable for comparison.
