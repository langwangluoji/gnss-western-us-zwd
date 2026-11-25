# Native Mean Fusion V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Root has authorized inline implementation; no Git actions or formal GPU training in this subtask.

**Goal:** Add independent C/H/S mean continuation with exact nonzero-source warm start and matched training/export/evaluation.

**Architecture:** Preserve original NativePhysicsLM names and forward semantics. H adds a zero signed current-innovation readout; S wraps the original StationEncoder input with one gated same-time K8/16 spatial message. Existing cache, loss and point metrics are imported unchanged.

**Tech Stack:** Python, PyTorch, NumPy, pytest.

**Spec:** docs/superpowers/specs/2026-11-26-native-mean-fusion-v2.md

## Global Constraints

- Only new files; no edits to historical models, caches or shared probability entry points.
- No Git actions, confirmation/C/D reads, or formal GPU training.
- Warm start maximum difference 2e-5mm; aux0.25; 3000 continuous cached blocks.
- Old/new LR1e-4/3e-4, cosine restart3000, eta_min each0.1×.

## Review Focus

- Nonzero source heads expose accidental output changes.
- Duplicate coordinates and changing context order require station-ID-stable ties.
- Missing neighbors must not inject masked numeric values through the new message.
- Existing AdamW step/moments survive; new parameters have no prior state.
- Export means are absolute millimeters, not raw corrections.

### Task 1: Models and CPU contracts

Create tools/terrain_diffusion/test_native_mean_fusion_v2.py and native_mean_fusion_v2.py.

- [x] RED: tests fail for missing v2 model; test nonzero-source equality, causal prefixes, reordering/ties, query chunks, missing neighbors and new gradients.
- [x] GREEN: implement NativeMeanFusionV2, load_source_state, C/H/S-only validation.
- [x] Run new tests plus existing native mean/physics tests on CPU.

### Task 2: Independent experiment runner

Create tools/terrain_diffusion/native_mean_fusion_v2_experiment.py; extend new test file.

- [x] RED: optimizer restoration/restarted scheduler and absolute-mm prediction tests fail for absent helpers.
- [x] GREEN: implement matched train, export, evaluate modes; archive source/checkpoint/cache/code lineage and actual updates.
- [x] Verify CLI help and targeted CPU tests. Root requested no broader old-suite expansion; root handles integrated tests.

### Task 3: Handoff

- [x] Record commands/results and owned file list; root runs real-checkpoint/cache four-step smoke and formal continuation.
- [x] No commits here; root integrates and synchronizes all agents' authorized changes.

## CPU execution evidence

- RED model: missing independent v2 implementation; then11 model contracts passed.
- RED runner: missing independent continuation runner; then15 model/optimizer/export contracts passed.
- Actual source checkpoints: seed42/17 weights loaded only onCPU; original and all3 v2 arms match within2e-5mm on synthetic72h histories.
- RED smoke guard: missing explicit budget guard; then matched-cache and independent ≤4-update smoke guards passed.
- Final command: `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tools/terrain_diffusion/test_native_mean_fusion_v2.py tools/terrain_diffusion/test_native_physics_lm.py tools/terrain_diffusion/test_native_weather_models.py -q` with CUDA_VISIBLE_DEVICES=-1:37 passed.
- Earlier related native suite:63 passed before the smoke guard was added. An old whole-project suite was started, then interrupted onroot instruction; no full-suite completion claim.
- CLI `--help` verified. Real cache4-update and formal3000-update runs remain forroot; no GPU runs or Git operations performed by this implementation task.
