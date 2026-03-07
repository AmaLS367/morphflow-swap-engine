# MorphFlow Swap Engine Plan

## Goal
Build a new `morphflow-swap-engine` as a strong replacement for the current weak face swap core.

The new engine must:
- stop depending on weak default FaceFusion behavior as the main quality path
- use stronger face detection, tracking, swapping, restoration, and temporal stabilization
- be designed for a server with RTX 5090, 32 GB VRAM, 31 GB RAM
- be modular so agents can replace individual technologies without rewriting the whole engine
- integrate back into MorphFlow through a stable internal adapter/API layer

This plan is written for execution by Codex, Cursor, Claude Code, and other agents.

---

## Product direction

We are not doing a cosmetic refactor.

We are building a new engine with these priorities:
1. stronger face swap quality on real videos
2. more stable target face selection across frames
3. less flicker and less identity drift
4. better behavior on small and imperfect faces than the current stack
5. efficient use of RTX 5090 resources
6. clean observability, benchmarking, and debug artifacts
7. production-ready integration path back into MorphFlow

---

## Strategic decision

We are not patching random pieces of the existing weak swap flow forever.

We will:
1. fork the current repository
2. create a dedicated engine module named `morphflow-swap-engine`
3. keep the current API/pipeline shell only where useful
4. replace the core CV stack step by step with stronger components
5. validate progress through benchmark-first execution, not vibes

---

## High-level execution phases

### Phase 0. Fork and freeze baseline
Objective: create a clean execution base.

Tasks:
- Fork current MorphFlow repository.
- Create new default branch for engine work, for example:
  - `main` remains stable
  - `feature/swap-engine-foundation`
- Tag current state as baseline, for example:
  - `baseline-facefusion-old`
- Preserve current pipeline for comparison runs.
- Do not delete the old implementation immediately.

Deliverables:
- forked repo
- baseline tag
- clean branch strategy
- README section explaining baseline vs new engine

Acceptance criteria:
- project still runs at baseline state
- old pipeline can still be invoked for comparison

---

### Phase 1. Engine foundation and architecture skeleton
Objective: create the new modular engine skeleton before hard swapping technologies.

Create a new package, for example:

```text
src/morphflow_swap_engine/
    core/
        entities/
        value_objects/
        contracts/
        services/
    application/
        use_cases/
        orchestrators/
        benchmark/
    infrastructure/
        detection/
        tracking/
        alignment/
        swapping/
        restoration/
        temporal/
        video/
        diagnostics/
        runtime/
    adapters/
        morphflow/
        cli/
    config/
    tests/
```

Rules:
- one module = one responsibility
- no cyclic dependencies
- contracts first, concrete implementations second
- all external model/tool wrappers live in infrastructure
- old MorphFlow pipeline should call the new engine through an adapter, not by reaching into internals

Core contracts to define first:
- `IFaceDetector`
- `IFaceTracker`
- `IFaceAligner`
- `IFaceSwapper`
- `IFaceRestorer`
- `ITemporalStabilizer`
- `IVideoDecoder`
- `IVideoEncoder`
- `IArtifactStore`
- `IBenchmarkRunner`

Core entities/value objects to define:
- `ReferenceFaceAsset`
- `TargetVideoAsset`
- `DetectedFace`
- `TrackedFaceSequence`
- `SwapRequest`
- `SwapResult`
- `StageArtifact`
- `BenchmarkCase`
- `BenchmarkRun`
- `EngineProfile`
- `RuntimeReport`

Deliverables:
- empty but working package skeleton
- typed contracts
- config layer
- adapter entrypoint from current MorphFlow code

Acceptance criteria:
- package imports cleanly
- type checker passes on foundation layer
- no hard-coded model logic inside application/core layers

---

### Phase 2. Benchmark harness first
Objective: stop guessing and create a fast repeatable comparison harness.

This phase is mandatory before deep model replacement.

Build a benchmark subsystem that can run multiple engine profiles on the same inputs and save comparable outputs.

Create benchmark dataset structure:

```text
datasets/
    references/
        ref_01.jpg
        ref_02.jpg
        ref_03.jpg
    targets/
        target_01.mp4
        target_02.mp4
        target_03.mp4
    cases/
        benchmark_cases.json
```

Each benchmark case should define:
- reference image path
- target video path
- clip range or full video mode
- expected target face count scenario
- notes

Benchmark runner must support:
- running only first N seconds of video
- running multiple engine profiles on the same case
- saving stage artifacts per run
- saving command/config snapshot
- saving timing and VRAM statistics if available
- saving final outputs in deterministic folder layout

Suggested output structure:

```text
storage/benchmarks/
    <run_id>/
        case.json
        profile.json
        metrics.json
        logs/
        artifacts/
            01_detection/
            02_tracking/
            03_swap/
            04_restore/
            05_temporal/
            final/
```

Metrics to record initially:
- total runtime
- per-stage runtime
- processed frames count
- detected face count consistency
- tracked face continuity
- swap output existence
- restoration stage enabled/disabled
- temporal stage enabled/disabled
- warnings

Do not over-engineer metrics yet.
Main purpose is quick visual and engineering comparison.

Deliverables:
- benchmark CLI
- benchmark config schema
- benchmark output structure
- 3-5 benchmark cases

Acceptance criteria:
- one command can benchmark multiple profiles on the same case
- artifacts are easy to inspect
- runs are reproducible

---

### Phase 3. Stronger detection stack
Objective: replace weak precheck-style face detection with a stronger modern detector.

Current weakness:
- old rough precheck logic is too noisy and too weak for production engine decisions

Plan:
- add a stronger detector implementation based on the stack chosen in the architecture plan
- keep detector behind `IFaceDetector`
- implement confidence thresholds and sane filtering
- support batch frame detection
- support selecting top faces by confidence, size, and centrality

Detector requirements:
- detect faces on reference image
- detect faces across sampled target frames
- support returning landmarks if available
- support face crop extraction
- support confidence score

Filtering requirements:
- prefer primary face based on configurable policy:
  - largest consistent face
  - center-most consistent face
  - tracked identity continuity
- ignore implausible tiny detections when a dominant face exists
- reduce false positives in reference analysis

Reference analysis should report:
- image size
- face count
- primary face box
- confidence
- face size ratio
- warnings

Target analysis should report:
- sampled frames
- detections per frame
- face size ratios
- continuity hints
- warnings

Deliverables:
- new detector module
- filtering policy module
- reference analyzer
- target analyzer

Acceptance criteria:
- detector beats current precheck behavior on benchmark cases
- false positives are reduced
- debug artifacts show selected boxes and rejected boxes

---

### Phase 4. Tracking layer
Objective: stop treating each frame like a disconnected universe.

This phase is critical.

Build a `IFaceTracker` implementation that:
- links detections across frames
- finds the dominant target face track
- keeps a stable identity selection through the clip
- exports tracked crops and metadata

Tracking outputs should include:
- track id
- box per frame
- landmarks per frame if available
- detection confidence history
- track length
- missing frame count
- stability score

Selection policy:
- choose the best primary track based on:
  - persistence across frames
  - average face size
  - average confidence
  - center bias
- fall back to largest stable track if needed

Artifacts to save:
- preview frames with tracking boxes
- JSON track summaries
- extracted face crops for the chosen track

Deliverables:
- tracking module
- track scoring module
- target face selection policy

Acceptance criteria:
- same main face is selected consistently across benchmark clip
- track summary explains why a target was chosen
- engine no longer relies on naive `face-selector-mode one` style logic

---

### Phase 5. Alignment and crop pipeline
Objective: feed the swapper cleaner inputs.

Implement alignment/crop stage:
- align reference face using detected landmarks
- align tracked target crops consistently
- normalize crop sizes to profile-specific expectations
- support optional expanded crop margin

Why this matters:
- the swapper performs better when the target and reference are normalized
- small faces in full-frame video need crop amplification

Requirements:
- crop around tracked face with configurable margin
- preserve enough forehead/jaw context
- avoid over-tight crops
- support crop-to-swap workflow for videos where face is too small in full frame

Deliverables:
- face alignment module
- crop strategy module
- crop preview artifacts

Acceptance criteria:
- benchmark artifacts clearly show normalized aligned reference and target crops
- crop mode improves usable face size for swap stage

---

### Phase 6. Swapper replacement layer
Objective: replace the weak default swap core with stronger swappers under one contract.

Important rule:
- do not hard-code belief that one model is the winner
- implement swappers behind `IFaceSwapper`
- benchmark them through the same harness

Profiles to support initially:
- `balanced`
- `high_quality`
- `aggressive`

Swapper adapter responsibilities:
- load selected model/backend
- receive aligned reference and target crops or sequences
- run swap
- return swapped frames/crops
- expose runtime metadata

Each swapper module should have:
- config schema
- warmup/load logic
- inference logic
- memory-safe batch logic
- output artifact hooks

Minimum implementation plan:
1. integrate the stack selected by the architecture proposal as primary candidate
2. keep one fallback swapper profile for comparison
3. add profile toggles in config/CLI

Artifacts to save:
- raw swapped crops/frames before restoration
- per-batch logs
- runtime stats

Deliverables:
- primary swapper implementation
- fallback swapper implementation or stub-ready adapter
- profile registry

Acceptance criteria:
- benchmark can run multiple swappers on the same case
- outputs are directly comparable
- engine can switch swapper without architecture changes

---

### Phase 7. Restoration layer
Objective: improve face detail only after the swap stage is isolated and measurable.

Build `IFaceRestorer` layer with ON/OFF support.

Rules:
- restoration must be optional
- benchmark must support swap-only vs swap+restore comparison
- restoration must not be forced during first-line debugging

Profiles to support:
- `off`
- `standard`
- `high_quality`

Responsibilities:
- restore swapped face details
- preserve identity as much as possible
- avoid over-smoothing or reverting toward target identity

Artifacts:
- restored crops/frames
- before/after restore comparison crops

Deliverables:
- restoration adapter
- restoration profile config
- benchmark toggle support

Acceptance criteria:
- restore ON/OFF can be compared quickly
- restoration stage can be disabled globally for debugging

---

### Phase 8. Temporal stabilization layer
Objective: reduce flicker, frame-to-frame identity drift, and unstable face texture.

Build `ITemporalStabilizer` as a dedicated stage.

Responsibilities:
- operate after swap or after restore depending on selected profile
- smooth frame-to-frame variation
- reduce temporal identity jitter
- reduce texture flicker

Requirements:
- keep module isolated
- support disable/enable by profile
- save stage artifacts for review

Artifacts:
- pre-temporal clip
- post-temporal clip
- frame diff summaries if cheap enough

Deliverables:
- temporal module
- profile integration
- debug artifacts

Acceptance criteria:
- visible flicker reduction on benchmark clips
- stage can be turned off for diagnosis

---

### Phase 9. Video reconstruction and compositing pipeline
Objective: rebuild final video cleanly after swap stages.

At this phase, the engine should support:
- swap-only output
- swap+restore output
- swap+restore+temporal output
- optional downstream composition hooks

Video layer requirements:
- robust frame decoding
- deterministic frame ordering
- audio preservation strategy
- clean encode pipeline
- high-quality export settings configurable by profile

If crop-to-swap mode is used:
- recomposite swapped face region back into the original full frame cleanly
- preserve alignment and placement
- avoid visible patch edges

Deliverables:
- video decoder module
- video encoder module
- crop reintegration module if needed

Acceptance criteria:
- final reconstructed videos are stable and playable
- audio handling policy is explicit
- no frame order corruption

---

### Phase 10. Diagnostics and observability
Objective: make the new engine easy to debug on server and locally.

This phase should reuse good ideas from the recent diagnostics pass.

Required diagnostics:
- stage records
- warnings
- artifacts manifest
- profile used
- selected detector/tracker/swapper/restorer/temporal modules
- per-stage timing
- environment snapshot
- model version snapshot

Suggested debug structure:

```text
storage/debug/<job_id>/
    metadata/
    logs/
    artifacts/
        01_detection/
        02_tracking/
        03_alignment/
        04_swap/
        05_restore/
        06_temporal/
        07_reconstruction/
```

Deliverables:
- artifact store
- runtime report models
- debug manifest
- clean logging strategy

Acceptance criteria:
- every failed or degraded job explains itself
- every benchmark run is inspectable without guesswork

---

### Phase 11. RTX 5090 optimization pass
Objective: make the engine exploit available hardware instead of acting like a low-end laptop pipeline.

Optimization targets:
- fp16 where safe
- batched inference where model supports it
- frame chunking
- preloaded models
- async/staged buffering where helpful
- memory-aware profile presets
- optional compilation/optimization for stable production paths

Requirements:
- optimization must not destroy correctness
- benchmark runs must include timing comparisons
- no hidden magic toggles without config entries

Profiles to define:
- `balanced`
- `quality_max`
- `throughput_max`

Potential config areas:
- batch size
- crop resolution
- target processing FPS cap
- restore stage batch size
- temporal stage chunk size
- memory limit guardrails

Deliverables:
- GPU-aware runtime config
- performance notes
- benchmark comparison tables

Acceptance criteria:
- engine runs stably on 5090
- memory usage is acceptable
- throughput improves relative to naive execution

---

### Phase 12. MorphFlow integration adapter
Objective: plug the new engine back into the MorphFlow product cleanly.

Rules:
- current frontend/API should not need chaotic rewrites
- existing process flow should call the new engine through an adapter
- old engine path may remain as fallback behind a flag during transition

Integration requirements:
- process request -> new engine request mapping
- profile selection support
- swap-only support
- diagnostics exposure
- benchmark/dev flags hidden from normal production use

Deliverables:
- MorphFlow adapter layer
- config flags for engine backend selection
- migration notes

Acceptance criteria:
- current MorphFlow can invoke the new engine
- feature flag can switch between old and new engine during rollout

---

### Phase 13. Validation and rollout
Objective: verify that the new engine is worth keeping.

Validation flow:
1. run benchmark suite on baseline old stack
2. run benchmark suite on new stack
3. compare outputs visually and operationally
4. identify winning profile
5. run server acceptance tests
6. switch default engine only after evidence

Acceptance checklist:
- face swap visibly stronger on core benchmark cases
- tracking is more stable
- flicker is lower or at least not worse
- debugability is better than baseline
- server execution is stable
- integration path works through existing MorphFlow shell

---

## Repository plan

Suggested repository layout after transition:

```text
repo/
    src/
        morphflow_swap_engine/
    datasets/
        benchmark/
    docs/
        architecture/
        benchmarking/
        deployment/
    scripts/
        benchmark/
        debug/
        deployment/
    tests/
        unit/
        integration/
        benchmark/
```

---

## Execution order for agents

### Agent Wave 1. Foundation
Scope:
- Phase 0
- Phase 1
- initial docs

Output:
- package skeleton
- contracts
- config
- adapter entrypoint

### Agent Wave 2. Benchmark harness
Scope:
- Phase 2
- benchmark CLI
- storage layout
- run manifests

Output:
- benchmark runner
- sample cases
- run reports

### Agent Wave 3. Detection and tracking
Scope:
- Phase 3
- Phase 4
- Phase 5

Output:
- detector
- tracker
- alignment/crop
- visual debug artifacts

### Agent Wave 4. Swapper and restoration
Scope:
- Phase 6
- Phase 7

Output:
- primary swapper
- fallback/comparison swapper
- restoration layer
- ON/OFF comparison support

### Agent Wave 5. Temporal and reconstruction
Scope:
- Phase 8
- Phase 9

Output:
- temporal stabilization
- reconstruction/export

### Agent Wave 6. Diagnostics and optimization
Scope:
- Phase 10
- Phase 11

Output:
- debug system
- GPU optimization profiles

### Agent Wave 7. Integration and rollout
Scope:
- Phase 12
- Phase 13

Output:
- MorphFlow integration
- migration plan
- rollout checklist

---

## Rules for all agents

1. Do not rewrite the whole repo blindly.
2. Do not mix architecture, benchmark, and model changes in one giant commit.
3. Every stage must preserve a runnable state.
4. Every new model/tool must be behind a clean contract.
5. Every stage must add artifacts/logging for inspection.
6. Do not silently remove the old path until the new path wins.
7. Do not invent success without benchmark evidence.
8. Keep code modular and replaceable.
9. Use Windows-friendly local development instructions where needed.
10. Comments in code must be English only.

---

## Commit strategy

Suggested commit sequence:

1. `chore: fork baseline and tag legacy swap pipeline`
2. `feat: add morphflow swap engine package skeleton and core contracts`
3. `feat: add benchmark runner and benchmark case schema`
4. `feat: add detector abstraction and first detector implementation`
5. `feat: add target face tracking and selection strategy`
6. `feat: add alignment and crop pipeline`
7. `feat: add primary swapper integration`
8. `feat: add swapper profile registry and fallback comparison path`
9. `feat: add restoration layer with profile toggles`
10. `feat: add temporal stabilization stage`
11. `feat: add reconstruction and export pipeline`
12. `feat: add diagnostics manifest and debug artifact store`
13. `feat: add gpu runtime profiles for rtx 5090`
14. `feat: integrate new engine into morphflow through adapter`
15. `docs: add benchmark and rollout documentation`
16. `test: add integration and benchmark verification coverage`

---

## Detailed technical checklist

### Foundation checklist
- [ ] fork repo
- [ ] create baseline tag
- [ ] add new engine package
- [ ] add contracts
- [ ] add config schema
- [ ] add adapter entrypoint

### Benchmark checklist
- [ ] benchmark case schema
- [ ] benchmark CLI
- [ ] benchmark output folder layout
- [ ] benchmark manifest JSON
- [ ] sample benchmark cases
- [ ] benchmark docs

### Detection checklist
- [ ] detector interface
- [ ] detector implementation
- [ ] reference analyzer
- [ ] target analyzer
- [ ] false positive filtering
- [ ] detection artifacts

### Tracking checklist
- [ ] face track builder
- [ ] track scorer
- [ ] primary track selector
- [ ] tracking artifacts
- [ ] JSON track summaries

### Alignment/crop checklist
- [ ] reference alignment
- [ ] target crop extraction
- [ ] configurable crop margin
- [ ] crop-to-swap mode
- [ ] crop preview artifacts

### Swapper checklist
- [ ] primary swapper adapter
- [ ] fallback swapper adapter or abstraction-ready slot
- [ ] profile registry
- [ ] raw swap artifacts
- [ ] profile config

### Restoration checklist
- [ ] restorer interface
- [ ] ON/OFF mode
- [ ] restored output artifacts
- [ ] compare restore enabled vs disabled

### Temporal checklist
- [ ] temporal interface
- [ ] temporal implementation
- [ ] pre/post temporal artifacts
- [ ] profile toggles

### Reconstruction checklist
- [ ] video decode layer
- [ ] video encode layer
- [ ] audio preservation policy
- [ ] reintegration logic for crop mode
- [ ] export config

### Diagnostics checklist
- [ ] stage records
- [ ] warnings model
- [ ] artifacts manifest
- [ ] runtime report
- [ ] environment snapshot
- [ ] model version snapshot

### Optimization checklist
- [ ] fp16 support
- [ ] batch inference config
- [ ] profile-specific runtime config
- [ ] timing benchmark output
- [ ] memory guardrails

### Integration checklist
- [ ] MorphFlow adapter
- [ ] engine backend feature flag
- [ ] diagnostics surfaced to existing API
- [ ] migration note
- [ ] fallback to old engine retained during transition

---

## Testing strategy

### Unit tests
Must cover:
- config parsing
- profile registry
- detector result filtering
- track scoring
- crop calculations
- artifact manifest creation
- adapter request mapping

### Integration tests
Must cover:
- benchmark runner execution
- one full swap-only pass
- one swap+restore pass
- one swap+restore+temporal pass
- MorphFlow adapter invocation

### Manual evaluation
Must cover:
- visual comparison on benchmark cases
- identity stability
- reduced flicker
- stable target selection
- performance sanity on RTX 5090

---

## Documentation plan

Create these docs:
- `docs/architecture/morphflow_swap_engine_overview.md`
- `docs/architecture/module_map.md`
- `docs/benchmarking/benchmark_runner.md`
- `docs/benchmarking/case_format.md`
- `docs/deployment/runpod_deploy.md`
- `docs/deployment/runtime_profiles.md`
- `docs/migration/legacy_to_new_engine.md`

Each doc must be kept in sync with the implementation.

---

## Non-goals

Do not spend time right now on:
- frontend redesign
- marketing pages
- broad unrelated API redesign
- perfect cinema-grade background replacement
- random refactors unrelated to swap engine migration

---

## Final target state

At the end of this plan we must have:
- a new `morphflow-swap-engine`
- modular modern CV stack
- benchmark harness for fast comparison
- stable target face tracking
- stronger swap quality than current baseline
- controllable restoration and temporal stages
- RTX 5090 optimized runtime profiles
- clean integration back into MorphFlow

This is the bar.
No vague “it feels better”.
The new engine must be stronger, more stable, and easier to debug than the current one.
