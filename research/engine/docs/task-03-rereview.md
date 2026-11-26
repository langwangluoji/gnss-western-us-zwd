# Production pipeline rereview, round 1

No actionable findings remain in the scoped fixes and their changed-code integration.

- **Current-only inference:** all four pipeline calls to `evaluate_predictor` pass the science history policy. The evaluator constructs the matching context builder and includes that policy in its artifact hash. Functional and pixel inference reject restored historical value/mask channels. The regressions cover mean screening, mean full-year inference, uncached functional role-D inference, and cache-contract rejection when the history policy changes.
- **Allocation wall limit:** `TrainEngine.run` compares its per-invocation monotonic elapsed time with the allocation limit while retaining cumulative checkpoint elapsed time for accounting. Both fresh resumption after exhaustion and promotion from an older completed stage advance in the targeted tests.
- **Auxiliary family/configuration identity:** selections retain `selected_products[family][product]`; candidates remain within their parent-assigned complete configuration. Fixed pretraining ablations preserve their parent source. Lock validation checks family/product coverage, parent configuration assignments and source hashes. Retrospective dispatch keeps separate family/product paths and does not inherit another family's main-product arrays into a seed's formal comparison.
- **Integration:** runtime cache paths use the configured cache root and content-addressed mean/condition identities. History is part of those identities. Sampling stability receives explicit nested member IDs and matching nonempty provenance, with separate forecast-prefix and point-metric tolerances.

Fresh scoped verification from the project root:

```powershell
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_pipeline_review.py -q
```

Result: **13 passed in 6.37s**. This rereview inspected `pipeline.py`, `prediction.py`, `engine.py`, `tests/test_pipeline_review.py` and the appended task report. It did not rerun the full suite or execute scientific training.
