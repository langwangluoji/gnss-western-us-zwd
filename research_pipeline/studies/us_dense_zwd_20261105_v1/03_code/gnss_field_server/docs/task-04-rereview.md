# Task 4 round 1 rereview

## Evaluation findings resolved

- Point bootstrap fixes the original scored station set through `active_stations`, rejects time draws missing any active station and reports attempts, accepted replicates and rejection counts. The original disjoint-availability probe now returns 5.5 mm for every accepted replicate. It accepted 20 replicates after rejecting 25 incomplete draws. Bounded exhaustion exposes rejection metadata through `BootstrapExhaustedError`.
- Crossed bootstrap fixes its endpoint-eligible group IDs before drawing groups and reports excluded IDs. A group with no observations in the registered endpoint population stays outside the draw population. Every selected group occurrence retains an observation-bearing mask in an accepted replicate, preserving duplicate-group weights. The original inactive-group probe now scores group `a`, records group `b` as excluded and returns a finite constant interval.
- Bstar selection requires learned controls to have step greater than 200. Explicit registered fixed controls retain their step-zero eligibility. The original learned step-100 probe now selects the eligible step-1000 checkpoint.
- Product locks store the normalized checkpoint SHA256 before calculating the lock fingerprint. The original uppercase-digest probe now passes retrospective validation.

The corresponding regression tests exercise missing stations, wholly empty draws, endpoint-empty groups, selected-group rejection, bounded exhaustion, learned steps 0/1/100/199/200 and uppercase lock round trips. Their assertions follow the repaired contracts.

## Stability provenance and numerical tolerance

`sampling_stability` now requires unique matching member-ID prefixes and matching nonempty fixed-noise provenance to return `stable=True`. It calculates output-prefix differences in float64 and caps the configurable prefix tolerance at 0.001 mm. The 0.005 mm point-RMSE tolerance remains a separate check. Missing or mismatched provenance returns an unstable result.

A direct FP32 probe with matching provenance and a 0.0005 mm prefix perturbation returns stable, with a measured maximum difference of 0.000499725341796875 mm. The same arrays through the previous caller signature return unstable because provenance is absent. The updated report explains both tolerance scales and the provenance contract accurately. No additional evaluation-local actionable defect was found in the scoped corrections.

## Integration follow-up

**[P2] Adapt the production stability caller to the new provenance requirement.** At this rereview snapshot, `src/gnss_field/training/pipeline.py`, lines 860–862, calls `sampling_stability` without member IDs or fixed-noise provenance. Every stochastic checkpoint therefore receives `stable=False`, including matching prefixes, forcing K128 selection and recording `point_instability=True`. Persist the forecast's global member IDs and fixed-noise stream identity, then pass the K64/K128 evidence to this call. This caller belongs to the parent's pipeline integration scope and has been reported directly.

## Gate interpretation and verification scope

The formal fine-scale IUT still requires all four paired CI upper bounds to be strictly below zero. C2019 advancement uses the four negative differences through `formal=False`. The repaired bootstrap populations provide valid fixed-population inputs to these separate decisions.

This rereview inspected implementation and regression assertions and reran the original failing probes plus the provenance/FP32 probe. The parent's reported 34 evaluation tests and 127 full-package tests were supplied context. This rereview performed focused probes.
