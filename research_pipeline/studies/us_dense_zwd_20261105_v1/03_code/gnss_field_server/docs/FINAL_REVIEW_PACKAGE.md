# Whole-project review package

This is a newly added standalone directory in a non-Git research workspace. No branch diff is available; all src/configs/slurm/tests files in this directory are the added implementation. Legacy code/data are outside scope and must remain unchanged.

Binding spec: APPROVED_PLAN.md. Execution and rulings: IMPLEMENTATION_LEDGER.md. Read README.md for actual CLI/deployment. Scientific details and task boundaries are in task-01-models.md, task-03-pipeline.md, task-04-evaluation.md, task-05-orchestration.md.

Task reviews: task-03-review.md / task-03-rereview.md, task-04-review.md / task-04-rereview.md, task-05-review.md / task-05-rereview.md, task-06-runtime-review.md and its fix report/rereview. Completed fixes should not be presumed correct without checking the integration.

Scope is the complete new package, especially production path contracts and operational behavior. Key files: training/pipeline.py, training/prediction.py, training/engine.py, runtime.py, cli.py, evaluation/*, orchestration/*. No paid jobs, remote commands or full scientific training permitted in review. No subagents.

Evidence: local_validation/data_pack_report.json and final_audit/data_audit.json (all240shards verified); final_real_cuda/smoke.json (real860B targets,77patches,35NFEES,200vs100+100bitwiseequal); profile_kernels/smoke.json (allhead paths andM2/cache). Full148tests passed before task06fix; final suite result will be supplied after it. Real Tongji signal/driver/throughput not yet observed; benchmark intentionally remains gated.

Controller flagged for examination: pipeline probability bootstrap has a separate implementation from the reviewed point/increment bootstrap; check denominator/population and formal coverage semantics. Check that production reports include approved14-day sensitivity. Observation refinement currently selects2018 with defaultminimum1; compare the approvedESminimum4 rule and determine effect. Review genuine scientific/operational correctness and missing load-bearing implementations, without requiring out-of-scope new architectures or paid cluster experiments.

Runtime integration implementer also noticed `_train_basis` and `_pixel_run` may pass effective32 hours where approved basis16 andM2 8 are required. Runtime estimate metadata uses approved16/8. Check production scientific batch declarations.

Return concrete defects with trigger, impact, exact source locations and targeted evidence. Record remote-only checks separately from implementation defects. Do not repeat existing full-suite or expensiveCUDA tests solely for review.
