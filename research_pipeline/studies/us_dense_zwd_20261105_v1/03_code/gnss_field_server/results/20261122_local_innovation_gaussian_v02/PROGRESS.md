# Local experiment v02

- Plan and local-pool rulings: ../20261122_local_innovation_gaussian_v01/PROGRESS.md.
- v01 stopped during engineering, before scientific training: loading all tensors to CUDA moved the CPU RNG state to CUDA. The failing actual-runner recovery regression reproduced the error. v02 restores checkpoint on CPU before loading optimizer/model to their live devices. v01 remains as failed-engineering provenance; no scientific results are taken from it.
- Source dataset, M0, original server bundle and allowed hours are read-only and hash checked.
- v02 immediate queue: data lock → real 100+100 vs 200 recovery for all three heads → matched two-epoch point heads and bounded exact constant diagnostic → frozen-mean Gaussian allocation audit → report/review.

- Complete: all6 scientific arms84/84 updates; actual recovery3arms bitwise; corrected Gaussian bootstrap100/100 per fold; paired weather/group intervals2000; K64→128 Gaussian rank stable.
- Final verification:323 tests passed. See delivery_validation.json, reviewed_results.json and RESULTS.md. No remote training submitted.
