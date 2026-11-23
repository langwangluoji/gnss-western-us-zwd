# Task 5 scoped rereview, round 1

Reviewed the four fixes from `docs/task-05-review.md` on 2026-11-17. Scope covered the changed orchestration areas and their regression tests.

## Disposition

| Original finding | Result | Verification |
| --- | --- | --- |
| P1 initial benchmark ceiling bypass through a full budget lock | ADDRESSED | `scheduler.py` lines 215-218 sums all declared benchmark attempts before either authorization path. The two parameterized regressions reject five hours in one attempt and six hours across two attempts, with zero submitted jobs and zero reservations. |
| P1 coordinator reload loses environment and working directory | ADDRESSED | `scheduler.py` lines 68-85, 139-152 and 166-185 persists and restores execution settings. Reload uses the read-only initialization path. Tests verify preserved script bytes and timestamps, pure-status directory snapshots, and a retry script with the saved environment and directory. |
| P2 mixed CPU/GPU phase underreserves allocated GPU time | ADDRESSED | `campaign.py` lines 67-69 rejects mixed phases. `scheduler.py` lines 92-93 repeats validation at the renderer boundary. The regression exercises both boundaries. |
| P2 sampler/checksum interruption prevents restart | ADDRESSED | `study.py` lines 146-161 and 180-182 stores payload and checksum in one atomic container, verifies it before deserialization and retains valid legacy-pair loading. Fault injection immediately after the durable write resumes the same RUNNING trial parameters, number and directory. |

No additional actionable regression was found in the four changed areas.

## Verification evidence

The targeted command `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q tests/test_orchestration_campaign.py tests/test_orchestration_study.py` passed independently during this rereview with **34 passed in 6.30 seconds**.

Additional temporary-directory checks independently verified that a legacy campaign lacking `execution.json` recovers its custom environment and spaced working directory without modifying existing files, and that a valid legacy sampler/checksum pair resumes its RUNNING trial and migrates to `sampler.state`.

The owner reports six failing regression cases before implementation changes in `docs/task-05-report.md` line 117. This rereview inspected those regressions and observed their passing state. Their earlier failing executions were owner-reported. The owner's full-suite result of 111 passed is recorded at line 127. The independent rerun here covered the 34 orchestration tests.

The documentation-only deliverable is this file. Cluster execution, live Slurm signals and the parent's ongoing CLI integration remain outside this scoped rereview.
