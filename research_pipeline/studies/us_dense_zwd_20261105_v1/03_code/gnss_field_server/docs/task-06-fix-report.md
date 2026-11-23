# Task 06 runtime review fixes

Implemented on 2026-11-17 against `docs/task-06-runtime-review.md`.

## P1 release binding

The independent CLI benchmark loads campaign `resolved.json`, validates explicit caller configuration and overrides, and compares recomputed science, pack, release and protocol hashes with `campaign.json` before entering benchmark. Worker and benchmark share `verified_release_hashes`. It verifies the manifest, shared files and every shard checksum and requires a registered observational pack. The exported runtime benchmark also checks the actual release before doctor and training, covering direct Python callers. Its existing positional API is preserved with an optional keyword `project_root`.

## P2 synchronization display

SSH state records the latest attempt and last successful synchronization across failures. The page displays synchronization success or failure, timestamps and error text independently of training state. Last observed training metrics and timestamps remain visible. A first failed synchronization is also visible before a remote snapshot exists. A campaign manifest supplies a read-only queue summary with state, job count and reserved card-hours.

## Timing metadata

M0 estimates use the measured effective-eight-hour microbatch sweep matching runtime settings. Basis uses effective16, conditional arms effective32, and M2 effective8 for explicit linear two-hour-kernel extrapolations. Each stage includes its configured update ceiling, effective hours, estimated seconds per update, card-hours and GPU-only CNY at 2 CNY per card-hour. Estimates cover a single stage instance. Validation, cache construction, I/O, repeated trials, folds, retries and queue time require separate accounting. Local smoke retains `verified_for_tongji_budget=false`. Real allocation, signal and resume checks continue to gate budget verification.

## Verification

Commands ran from the project root using the requested PyTorch interpreter.

```powershell
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_cli_monitor.py tests/test_runtime.py -q
```

The initial test-first run produced 7 failed and 10 passed in 5.90 seconds, covering missing release validation and synchronization display. The final run produced 23 passed in 4.80 seconds. Coverage includes each campaign hash mismatch, synthetic pack rejection, changed scientific configuration and source bytes, corrupt shard rejection, explicit config and data-path overrides, direct runtime release validation, retained last-success metadata, queue manifest reads, executed Node frontend success-to-disconnect rendering, and effective-hour/cost estimate units. The frontend execution uses lightweight DOM stubs and preserves rendered metrics across disconnect.

Scope consists of `cli.py`, `runtime.py`, `monitor.py`, `resources/monitor.html`, `tests/test_cli_monitor.py`, `tests/test_runtime.py` and this report. This fix round used local tests and mocks. Actual Linux/Slurm allocation and signal acceptance remain server-side verification. GPU profiling, SSH and paid jobs were outside this run.
