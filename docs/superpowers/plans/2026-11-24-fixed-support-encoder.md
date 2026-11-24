# Fixed Support Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimally invasive fixed-support encoder prototype and verify it locally against the existing foundation-field pipeline.

**Architecture:** Reuse the current data contract and 113-dimensional point features. Add multi-scale observation summaries, a fixed support-grid branch, a small zero-initialized U-Net correction, and a point readout. Keep the prototype separate from v13 and from diffusion training.

**Tech Stack:** Python, PyTorch, NumPy, existing `terrain_diffusion` modules, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-fixed-support-encoder-design.md`

## Global Constraints

- Do not read confirmation/C/D labels.
- Do not use PWV/IWV/TCWV/Tm.
- Do not modify existing v13 source, weights, locks, or results.
- Use the existing 410/137 contract and validation metrics.

## Review Focus

- Masked or empty support must produce finite features and explicit zero support weight.
- Height transport must use condition and query heights in opposite directions.
- Query-set expansion must not change the support branch.
- U-Net correction must start at zero so the first checkpoint is the point baseline.
- Training and evaluation must preserve the existing target mask and station-balanced metrics.

### Task 1: Prototype module

**Files:**
- Create: `tools/terrain_diffusion/fixed_support.py`
- Test: `tools/terrain_diffusion/test_fixed_support.py`

- [ ] Write tests for finite masked summaries, zero-initialized correction, and query-set invariance.
- [ ] Run the tests and observe the expected import failure.
- [ ] Implement `FixedSupportModel` and `run_smoke()` using existing `Experiment` data helpers.
- [ ] Run unit tests and the smoke command.

### Task 2: Local paired screen

**Files:**
- Modify: none; output under `results/20261124_fixed_support_encoder_p14/`

- [ ] Run two seeds for A1 and A2 under a bounded local budget.
- [ ] Recompute metrics from saved predictions and verify hashes.
- [ ] Write a result report with distance/elevation/pair/increment diagnostics.

### Task 3: Verification and synchronization

- [ ] Run the focused tests and existing terrain-diffusion audit checks.
- [ ] Inspect `git diff` and confirm only the new spec, plan, module, tests, and lightweight evidence changed.
- [ ] Commit and push the lightweight changes only if verification passes.
