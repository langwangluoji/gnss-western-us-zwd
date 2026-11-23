# Local pilot ledger

Plan: ../20261122_model_architecture_pro_v01/MODEL_PLAN.md

- Scope: local experiments only; no server submissions, environment changes or changes to immutable releases.
- Ruling: use all 384 previously cached core hours (32 per month) as the local direction-test pool, then remove four new inner weather weeks and buffers. Full 5472-hour condition reconstruction takes approximately one local GPU hour and belongs to the later full experiment. The small-pool result cannot claim full-pool convergence.
- Ruling: the heads are pointwise. Training at every station interpolation corner is numerically identical to full-grid forward and query; verify this with a nonzero-head gradient test and full-grid publication. Do not interpolate conditions before the nonlinear head.
- Ruling: include an exact bounded least-squares constant-innovation fit as an optimization diagnostic; the registered tiny-LR constant head also runs for the matched two epochs. This distinguishes an under-updated scalar from a failed innovation hypothesis.
- Pre-flight: feature cache raw67 uses untransformed densities; apply the existing feature69 log-density transformation without changing other channels. New innovation J is relative to M0+IDW, while existing feature innovation remains relative to M0.
- Tasks: (1) numerical contracts; (2) data and nested locks; (3) two-epoch point comparisons and recovery; (4) Gaussian scale audit; (5) report and independent review.
