# Performance orchestration correction

The campaign now separates the short measured benchmark from full cache preparation and the registered scientific diagnostic pilots. A successful measured budget lock permits the sequential preparation and diagnostic stages; model training retains its diagnostic and representation gates.

## Runtime changes

- `runtime.staged_preparation=true` orders `prepare → diagnostics → control → vae_representation → posterior_cache → prior/shared/complete stages → final_evaluation`. Its default remains false for existing runs.
- `run_full_generative_preflight.py` supports the staged benchmark-only path: benchmark, estimate the complete campaign, create the immutable budget lock, and publish `COMPLETE`. Its CPU handoff starts the first pending stage.
- Full cache preparation receives one GPU and eight CPUs for four workers. It gets its whole remaining stage allocation, retains a 600-second atomic-month completion grace, and rejects an allocation that cannot provide 60 seconds of work plus that grace. Cooperative preparation pauses preserve committed shards and publish `paused`.
- Other stages retain at least 60 seconds of allocation and use `min(180,max(20,ceil(0.1×allocation_seconds)))` as signal and shutdown grace. A 60-second allocation therefore provides 40 seconds of useful work.
- Diagnostics receive their immutable cumulative stage cap and the current allocation deadline. Runtime evaluation batching is passed as eight hours and 16 decoded fields, with all registered samples, folds and five-epoch pilots preserved.

## Cost and accounting

- Sustained update costs are multiplied by the complete registered workload. `stage_startup_seconds` and `posterior_cache_startup_seconds` enter their own stage exactly once, before the 1.5 safety margin.
- Preparation and scientific pilots have their own stage caps within the complete 24-card-hour limit. The one-hour short-preflight cap covers measured benchmark and recheck allocations.
- Parallel preparation startup is added once through `preparation_startup_seconds`; `preparation_total_seconds` remains an audit value and is not added a second time.
- Frozen prior elapsed allocations and the current measured benchmark are added. The regression fixture preserves the historical 356 seconds from job 3697508 and adds the current four-second benchmark to obtain 360 seconds.
- Every later dispatch refreshes terminal Slurm elapsed accounting before reuse. Stage caps, total 24-card-hour cap, duplicate protection and controlled continuation behavior remain active.
- The registered training hours, epochs, batch populations, latent candidates, discrepancy candidates, member streams and final candidate count are unchanged.

## Validation

Test-first regressions were observed failing for the new staged prefix, auxiliary gates, full preparation/pilot budgets above one hour, one-time startup pricing and short-allocation progress. The benchmark-only runner was then exercised with a controlled CPU test, confirming that the budget lock is written without cache preparation or pilot execution.

Latest targeted command:

```text
PYTHONPATH=src;tools
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_full_generative_campaign.py -q
```

Result: **39 passed in 4.49 seconds**. This includes auxiliary preparation pause preservation, diagnostic cumulative-cap/current-deadline checks, and one-time parallel-preparation startup pricing. An earlier combined campaign/preparation run passed 47 tests; the subsequent combined run observed 48 passes and two new preparation-owner RED tests while that worker was implementing interval-union pricing and in-memory startup warmup. Those two tests are outside this worker's file ownership and require the preparation implementation before the integrated rerun.

No SSH connection, Slurm submission or paid GPU benchmark was performed by this worker. The new real-A800 sustained benchmark must pass the immutable 24-card-hour gate before the staged campaign is submitted.
