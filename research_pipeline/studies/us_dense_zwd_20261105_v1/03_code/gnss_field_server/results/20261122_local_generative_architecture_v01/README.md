# Local GAN / Diffusion / NDP pilot

Read RESULTS.md for metrics and conclusions; PLAN.md for original six grid configurations; GP_NDP_PLAN.md for the adaptive unit-marginal correlated-noise follow-up.24 result records contain20 actual fitted models plus4 Gaussian controls. No server job was submitted.

- fold_0 and fold_1 are separate spatial/weather development folds, not one deployed joint-field product.
- G0 / G-pair / G-noise: matched GAN, relation critic, multiscale noise.
- D0 / D-attn / D-MP: matched diffusion, bottleneck condition attention, functional magnitude-preserving blocks.
- N-point / N-set: observation-space EDM-DSM with independent noise.
- N-GP-point / N-GP-set: same DSM but fixed Gaussian-prior unit-marginal correlated noise.
- candidate_*.pt are raw candidates; latest.pt includes actual final updated model/optimizer/RNG/history. result.json locks the selected candidate. It may be step0 for the grid pilot.
- outer_predictions.npz are original K32 samples and locked point estimates. stability_K512.npz are fixed-checkpoint member-prefix audits. control_variate_point_predictions.npz contains estimator-only point means, not changed ensembles.
- sampling_stability_v02.lock.json and sampling_dependency_validation.json bind the final audit to model/data/factor/checkpoint hashes. v01 files are interrupted exploratory audit history; v02 is authoritative.
- cache/conditions69.npy is the reusable full912-hour FP32 input cache; each row has a hash. cache_builder_snapshot.py exactly matches the original cache-builder hash and can rebuild/resume that cache.
- trained_mechanism_checks.json verifies raw updated models, not just selected0-step products. dependence_shuffle.json checks member alignment while preserving marginals.
- base_field_audit.json is descriptive roughness on supported land, not a dense-truth metric.

## Existing Windows environment

Run from gnss_field_server with PYTHONPATH=src;tools;results/20261122_local_generative_architecture_v01 and PYTHONUTF8=1. Interpreter: D:/anaconda3-2025.12-2/envs/pytorch/python.exe. No environment install is required.

Commands in order (omit the cache builder when cache is already complete):

```powershell
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/cache_builder_snapshot.py
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/run_pilot.py --mode all --steps 48 --set-steps 480
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/gp_ndp.py
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/sampling_stability.py
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/diagnose_pilot.py
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/base_field_audit.py
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' results/20261122_local_generative_architecture_v01/report_pilot.py
```

Create a PAUSE file in this result directory to stop training at the next check boundary. Remove only this marker to resume with the same commands and unchanged code/locks. This prototype reports the pause exception as FAILED; the saved optimizer checkpoint is retained and unsaved updates may be repeated. It is separate from production Slurm pause/resume.

Scientific scope: B2018 historically exposed incremental development. Inner/outer12 calendar-selected hours for this local screen; no new C2019/D2020 labels.48 grid steps do not establish convergence or exhaust an architecture's potential. NDP prototypes generate observations on a fixed station set and do not claim continuous-field/query-subset consistency.
