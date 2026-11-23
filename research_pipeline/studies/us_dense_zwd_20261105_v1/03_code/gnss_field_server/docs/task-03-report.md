# Production stage pipeline implementation

The dispatcher in `training/pipeline.py` implements all eighteen pipeline aliases.
Its artifact root is `campaign/seed_<seed>`. Each alias returns saved artifact paths
for the parent CLI receipt. `ScientificGateFailed` identifies a scientific stop,
and `PauseRequested` preserves a recoverable planned pause.

## Executable training paths

- M0-base uses B2016–2017. M0-all trains independently on B2016–2018. Both use the
  packed 2016–2017 input statistics, station-balanced masks, the fixed Z0 scale,
  full patch mosaics, supported-stencil curvature, EMA screening and full C2019
  reranking before freezing.
- Basis cross-validation holds out complete 50 km B groups within 2018. Each
  update evaluates the normalized neural basis on the publication grid, derives
  its B observation matrix from that grid, and differentiates through the FP64
  regularized coefficient solve. Diagnostics retain the observability rank and
  held-group point/increment metrics. The final basis refits every qualified
  coefficient and saves Phi, H, coefficient transforms and neural weights.
- The shared condition encoder trains with the Det head, receives concrete
  full-C2019 checkpoint selection, and produces monthly frozen-h caches.
  Functional heads use the same frozen encoder and spatial decoder. Estimated
  coefficients enter pretraining. Observation refinement uses every qualified
  2018 B hour and the published-grid station map.
- Det, Gaussian, EDM and GAN run distinct pretraining and observation phases.
  EDM observation gradients traverse all 35 denoiser evaluations of Heun-18.
  GAN performs five complete effective-batch critic updates before one generator
  update, with GP10 and Adam betas `(0, .9)`. The observation phase freezes the
  critic. M2-F and M2-J use identical initial weights and training cursors,
  separate mean/variance optimizers and clipping groups, and the registered
  surface scale. M2-F detaches the residual and feature inputs to its variance
  objective.

## Search and scientific selection

Optuna trials use quarter-budget caps with full-horizon schedules. Each trial
preserves its pretraining checkpoint separately from observation-phase weights.
M2-F registers its real-observation checkpoint as its warm-start source.
Promotion checks stage, data, seed, architecture, horizon, hyperparameters and
software-release equality before restoring the source model, EMA, optimizers,
RNG and cursor into a new full-budget directory. Observation refinement starts
as a new phase after the extended pretraining run. M2-J starts from the original
seed initialization for the selected M2-F configurations.

Full-C2019 reranking evaluates the union of each product's top three candidates.
The stability stage evaluates nested K64/K128 ensembles for every generative
candidate and compares point shifts, candidate order and gate outcomes. A
changed result uses the uniform full K128 rerank and retains an instability flag.
Bstar uses the registered deployment baselines and point controls. Fine and
probability direction filters run independently for each family, including GAN.
`STOPPED_NO_GAIN` is recorded when every family fails its filters.

The ablation dispatcher implements fixed-pretraining evaluation, zero geometry
bias, current-only inputs, rank64, rank256 and joint-encoder EDM refinement.
Seeds 17 and 101 rebuild their mean, basis and condition foundations and train
the locked complete configurations. Point/joint configuration unions are kept
through their separate full-C2019 selections.

## Retrospective access and exports

The immutable product lock binds the instantiated configuration, packed manifest,
software release, selected products, controls, foundations, selected seed repeats
and selected ablations. Current hashes are checked before constructing the D2020
store. Building the 2020 mean cache follows this validation. D2020 execution
evaluates the locked products and controls, repeats and ablations. The formal
summary recomputes paired calendar-block point uncertainty, crossed
calendar-block/group increment uncertainty, probability-score uncertainty,
coverage-difference intervals and Holm-adjusted selected-product claims.

Export writes monthly winning mean/std Z0 grids, per-hour support, land masks and
event members. The lock records an A-input-only event rule, selecting the twelve
largest daily mean absolute Z0 innovations for export. This event rule is an
implementation choice requiring protocol-owner review.

## Recovery and verification

`TrainEngine` gained optional `auxiliary_optimizers`, `before_update` and
`pause_paths` arguments. Auxiliary optimizer states are serialized and restored.
The callback runs once per complete effective batch. Known campaign pause files
are checked before each update, including nested trial directories. Optimizer
groups may set `lr_scale` for the joint-encoder 1e-5/1e-4 learning-rate ratio.

Fresh validation with the standard PyTorch environment completed with
`105 passed in 18.95s` using `python -m pytest -q`. The thirteen pipeline tests
cover an actual published-grid mean update, the differentiable basis solve and
held-current-B exclusion, all four functional training-phase paths, the
35-evaluation EDM observation gradient, frozen foundations, GAN critic state,
promotion-source validation, optimizer/RNG/cursor preservation, alias coverage,
campaign pause handling and the pre-lock 2020 access gate. The legacy-model
tests exercise faithful variance-gradient behavior, and evaluation tests cover
pretraining/startup exclusion and selection locks.

These checks establish engineering behavior on synthetic inputs. Full training,
actual-data GPU smoke validation and the scientific direction/confirmation
outcomes belong to the parent validation run and the eventual campaign.

## Review round 1

The three P1 findings in `task-03-review.md` are addressed.

- Every pipeline evaluation entry point supplies the explicit history policy.
  `evaluate_predictor` constructs its context builder with that policy and binds
  the policy and four-frame schema into its artifact hash and saved result.
  This covers mean screening, full-year selection, condition selection,
  deployment controls and retrospective inference. Functional and pixel
  predictors check that current-only inputs contain masked historical values
  and availability. Regression tests exercise actual mean screening/full-year
  inference and an uncached functional role-D inference path.
- Engine wall limits use the monotonic elapsed time of the current `run`
  invocation. Cumulative checkpoint elapsed time remains available for
  accounting. Allocation-local elapsed time is included in status and training
  records. Tests exhaust an allocation, resume it in a fresh engine, and extend
  a promoted checkpoint whose prior elapsed time exceeds the new allocation's
  limit. Each fresh allocation advances its optimizer.
- Auxiliary selections use `selected_products[family][product]`. A child may
  select an epoch within the parent-assigned complete configuration for that
  family/product. The parent configuration key and learning rates are retained
  in every selected record. Pretraining ablations evaluate their fixed parent
  source. The lock includes parent family configurations and source hashes,
  validates complete family/product coverage, and rejects changed
  configurations. Retrospective outputs retain separate family/product paths
  and formal seed summaries. Regression cases reverse EDM/GAN rankings across
  seeds and give another configuration a better off-product score. Every
  required family and its assigned configuration reaches evaluation.

`RuntimeConfig.cache_root` now holds both mean and condition caches. These use
content-addressed paths under `mean/` and `condition/`. Each seed records its
resolved hot-cache root and input policy in `cache_binding.json`. The condition
artifact records its external cache path in `condition/cache.json`. Cache keys
bind input hashes, normalization statistics, grid, foundations, encoder,
scenario and history policy. Monthly condition caches retain checksum checks.

Sampling stability passes explicit member IDs `0..63` and `0..127`. Each call
supplies matching nonempty provenance containing sampler settings and hash,
sampling seed, checkpoint hash, release hash, input hash, foundation hashes,
hours and station IDs. Forecast-prefix comparison uses a separate 0.001 mm
FP32 tolerance. The point-RMSE stability tolerance remains 0.005 mm.

The tests-first reproduction command was

```powershell
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_pipeline_review.py -q
```

Its initial output was `9 failed in 4.37s`. The failures reproduced the restored
history channels, missing history contract argument, exhausted resume clocks,
pooled family selection, dropped retrospective families, ignored hot-cache root
and missing stability provenance. After the fixes and additional configuration
and lock regressions, the targeted validation command was

```powershell
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_pipeline_review.py tests/test_pipeline.py tests/test_training.py tests/test_system.py -q
```

Its output was `29 passed in 9.91s`. The final repository validation command was

```powershell
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q
```

Its output was `148 passed in 21.82s`. All commands ran from the
`gnss_field_server` project root. Full scientific training remains pending.
