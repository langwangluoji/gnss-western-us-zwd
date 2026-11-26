# Task 5: Slurm campaign, Optuna and budget controls

Own src/gnss_field/orchestration/, slurm/, tests/test_orchestration*.py ONLY. No agents, no legacy edits,
no paid/remote jobs, no SSH. Parent owns CLI/training and will wire your APIs. Write tests first and report
docs/task-05-report.md. Python D:/anaconda3-2025.12-2/envs/pytorch/python.exe. No Git repo.

## Binding design
PortablePyTorch, Slurm arrays phasebarriers concurrency2. OneL40/job (48GB),8CPU64GB,24hsegments.
Sources publichttp://nic.pages.tongji.edu.cn/hpc-doc/ . Firstbenchmark<=4L40cardhours; fullsubmitmusthave
budgetlock withtotal/per-stage andplan/data/code/protocol hashes. Localimplementation mustNEVERsubmitjobs.
Use dataclasses Task/Campaign, serializeimmutableJSON. Tasksargv not interpolatedshellcode.
Stages pack/audit->doctor/benchmark->budgetgate->mean_base+mean_all->mean_cache->basis_cvfold0/1/2
->representation_gate->basis_final->condition->searchstudies->promote->full2019rerank/stability
->directiongate->ablations+seeds17/101->productlock->2020evaluate->report/export.
Scientific dependencies are artifact success+hash andSlurmstatus, notfileexistence. Pause/FAIL/gatestop blockdownstream.
Per-taskJSONL only, coordinatorcampaignmanifest only. No sharedappend/DBmultiwriter.

Define and document ergonomic Python APIs in __init__.py; parent will adapt CLI to them.
Needed capabilities:
- build default campaign DAG and validate no duplicateID/missingdeps/cycles. Explicit phase grouping ensures
  arrays%2 acrossoneactivephase, afterokbarriers; afteranysummaryCPUjob. Eventual tasks carry scientificgates.
- render shell script --signal=USR1@180 (NO B:) and srunpython-u-mgnss_field worker --task-file ...
  --index ${SLURM_ARRAY_TASK_ID}; CPUs/memory/gres/partition,quote paths,precreate logs before sbatch;
  sourceconda /share/apps/miniconda3/etc/profile.d/conda.sh, configurableenv gnss-field; no sudo/noexportGPU0.
- dry-run writes scripts/manifest but no subprocess submission. Actualsubmit requires explicitflag+verified
  Tongji preflight+budgetlock (orbenchmarkcap<=4), registersjobIDs atomically, refuses duplicateactive jobs.
  Resume regenerateincomplete tasks preservingrunIDs, check dependencystates; pause onlycampaignowned IDs.
- reserve declaredworstcase cardhours sum<=lockedtotal; per-stagebounds inclfailed/resume attempts.
  No infinite autorequeue. GPUcostL402/A8004 forestimateonly; no inventedCPUfees; localRTXreportcannotunlock.
- per-jobpause/statushelper atomicartifact success commits; scientific STOPPED_NO_GAIN differsFAIL.

## Optuna APIs
Provide run_study(directory,arm,objective,*,trials=8,seed=20261117,startup=4,...).
objective(params:dict, trial_directory:Path, trial_number:int)->dict:
{'point': {'value': finite_RMSE,'checkpoint':str},
 'joint' OR 'marginal': {'value': finite_ESorCRPS,'checkpoint':str},
 'pretrain_checkpoint':str,'contract':str}
Det singleobjectivepoint; m2_f multiobjectivepoint+marginal; gaussian/edm/gan point+joint.
params keys pretrain_lr log3e-5..3e-4,refinement_lr log1e-5..1e-4; fixedbaseline1e-4both.
TPESampler n_startup_trials4, JournalStorage file singleowner perstudy, sequentialtrialloop.
NO multiobjective report/pruner. Budgetcounts FAIL aswellasCOMPLETE; RUNNINGresume same params/number/path first,
KeyboardInterrupt leavesRUNNING forplannedpause. parent PauseRequested willsubclassKeyboardInterrupt.
Three consecutive sameerrors stopsstudy; nonfiniteobjective FAIL, notbadscore. Completedcallsnotrerun.
Select best complete trial perproduct,union<=2configs, correct objective/checkpoint association.
Promotions must point savedPRETRAINcheckpoint; sourceES weightsnever sentbackpretrain; no overwritetrialartifacts.
Expected study callback returns referencepaths evenwhen testsusecheapsyntheticcallback; productionvalidatescontents.

## Acceptance tests
DAGbarriers,cycles,duplicatejobs via mockedSlurm; no sbatch duringdryrun orimport;
missingbudget/localprofilefail; bounds4benchmark,totalperstage,attempts; shellsignalroutesjobstep;
pause/resume/failgatedependants; Optuna restartnorepeattrials, resumeRUNNING, FAILcount,error3halt,
dual selectedscoresreferencecorrectcheckpoint, no unsupportedmultiobjectivepruner/report.
Read contracts.py/artifacts.py for parentsharedtypes. Do not implementthetrainingCLI or remoteSSH.
