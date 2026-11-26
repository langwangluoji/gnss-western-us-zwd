# Pair-centering and spectral evidence review

- Maintenance suite: **26 passed**. New finite-support enumeration tests prove mean, variance and cross-coordinate covariance; query subsetting and invalid inputs are checked.
- Independent read-only review found **Critical 0 / Important 0**. Both minor wording findings were resolved: inference is restricted to the tested pair-difference scheme, and fourth-cumulant statements explicitly require the corresponding high-order moments.
- The reviewer independently recomputed paired Local/Affine F6 scores for both seeds from source NPZ files, and all 720 near-pair F6 scores for seed42. Results matched.
- Five selected shared-month intervals matched independently computed values to `1e-13`.
- Twenty spectral evaluation rows were checked against actual 64-dimensional input weights and exact tensor equality with the final3000 checkpoints. Weight, protocol and checkpoint hashes matched.
- Spectral sampler/source distinctions are retained: DDIM40 and Heun20 both use40NFE; Gaussian uses1NFE; seed42 original/legacy versus seed17 commuted frequency inference is reported accurately. Original evaluator hashes are retained, with current independent member-score recomputation.
- Both figures were visually inspected. New report evidence links resolve. The mathematical wording regeneration left every numerical record unchanged.
- Raw32-to-pair16 changes the center and higher-order shape together; it does not isolate a pure mean intervention. Members remain IID because pairs do not overlap. Population covariance preservation does not mean identical finite sample covariance.
- C64 spectral comparisons are matched to C64 joint models, not to the richer S175/G265 information conditions. Development labels remain exposed, strict OOF is false, and confirmation/C/D labels were not read.
- The lead-dependent center specification is a next experiment, not an executed result. No original weight, protocol lock or raw prediction was overwritten.
