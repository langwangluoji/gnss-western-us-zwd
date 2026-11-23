> 2026-11-21核验修订：两折U-Net外层RMSE9.775/9.615mm，原底座7.577/7.631mm。因此本轮共同均值明确锁定M0+IDW；保留U-Net生成器结构，旧均值修正权重只作审计，不晋级。以下涉及冻结旧U-Net均值的早先条款由本条替代。

# Grid generation comparison implementation plan

> For agentic workers: use executing-plans inline, tests first, one fresh whole-change review.

Goal: implement and start one bounded A800 sequential stochastic ResU-Net / DF / GAN campaign.
Architecture: frozen fold-specific mean, shared read-only conditions and Gaussian prior; free-grid residual generators with station-observation losses.
Tech Stack: existing native PyTorch, TrainEngine, NPY and Slurm; no environment installation.
Spec: docs/superpowers/specs/2026-11-21-grid-generation-v01.md

Global constraints: preserve old releases/runs; 5472 train hours, 96 inner anchors, two B2018 folds, no C/D; one A800, cumulative14400sec cap. 684 updates per arm/fold, EMA .99, K4 train/K16 inner/K32 outer96. Diffusion18 Heun35 calls. Tau once in observation space.

Review focus: Gaussian identity at step0; gradient through all35 calls; masks never enter generator; keyed member prefix; GAN optimizer/critic recovery and no dropped scoring rows; no duplicate or over-budget submission.

## Task1: models and prior
- [ ] Write failing tests for exact Gaussian bypass, gradient, sample-prefix, station query and masked permutation-invariant critic.
- [ ] Add src/gnss_field/models/grid_generation.py, reuse residual architecture, implement keyed factors and analytical-initialized diffusion.
- [ ] Green tests and real-shape gradient/memory smoke.

## Task2: data/training/evaluation
- [ ] Add tools/grid_generation_data.py: old release/cache verifies, fold-specific frozen mean, no new input schema.
- [ ] Add tools/grid_generation_train.py and grid_generation_metrics.py: independent stages, critic auxiliary optimizer, joint/point candidates, finite probabilistic and shuffled-group metrics; explicit pilot cohorts.
- [ ] Test actual100+100 versus200 with models, EMA, critic, optimizer, RNG/cursor; restore paused evaluation.

## Task3: deployment and monitoring
- [ ] Add locked config, isolated release builder, bounded Slurm queue/controller and transfer script; no code overwrite.
- [ ] Fresh review, fix critical/important once, run all relevant tests; create immutable release.
- [ ] Deploy and verify bytes; one A800 short benchmark gates budget before training; open visible log/monitor and verify startup.

Non-Git ruling: current directory is not a checkout, so isolate with added files and immutable SHA256 release rather than changing another repository. User accepted the preceding design and explicitly requested execution; routine choices are recorded in spec without repeated approval.
