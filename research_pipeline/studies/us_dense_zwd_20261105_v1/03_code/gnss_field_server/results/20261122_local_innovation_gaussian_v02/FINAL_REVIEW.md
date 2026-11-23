# Independent review disposition

Reviewer: local_innovation_review. Read-only review of source, tests and completed runs; no edits/training by reviewer.

Important findings resolved:

1. Preliminary Gaussian bootstrap changed the 200 km fitted composite estimator to 50 km groups. Replaced by same-estimator bootstrap; objective and gradient equivalence tested at unit multiplicities. Final reviewed bootstrap uses 100 synchronized weather-block x original200km-group replicates and drops none.
2. Exact bounded constant was incorrectly included in matched-budget selection. Removed from the reviewed selector; regenerated selected means and all Gaussian calibrations. It remains an explicitly exploratory optimization diagnostic.
3. PAUSE could flow into downstream completion. Added unconditional boundary checkpoint and explicit budget guards, observed failing pause regression then pass. Completed84/84 scientific runs were unaffected.
4. EMA.999 over84 updates retains91.94% of initialization. Kept the predeclared EMA selector, reported raw last weights separately, and limited conclusions to an optimizer/EMA direction pilot. No claim that architecture capacity has been exhausted.

Second independent review found no remaining Critical or Important correctness/estimand issue. Minor issues: acceptance marker preceded plots (moved after successful plot/report generation); full-grid export currently implements the actual gated winner path (other future winner paths require extension before use); target perturbation is a narrow inference check and not a claim to revalidate all historic data/statistics isolation.

Additional numerical finding: bootstrap replicate11 fold1 reached gradient[-2.03e-8,-4.36e-8] but strict L-BFGS line search returned ABNORMAL near floating precision. Kept main-fit tolerance1e-8, explicitly set bootstrap tolerance1e-7, preserved a spectral fixture and regression test. No failed bootstrap sample was silently discarded in final results.

Final sources: reviewed_results.json and fold_*/reviewed/. Preliminary Gaussian/selection/bootstrap files are superseded, retained for provenance. Actual training source hashes match code_snapshot/. No C2019/D2020 evaluation, no server submissions, no package installation or changes to old training artifacts.
