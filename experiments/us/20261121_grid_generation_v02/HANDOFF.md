# Generation v02 runtime handoff

User authorized independent stochastic ResUNet + DF + GAN training, single A800. Original full outer results audited: old frozen M0+IDW base RMSE7.577401/7.631033mm; MLP7.556044/7.642464; U-Net9.774505/9.615438. Old U-Net mean is NOT promoted. Both gen folds share frozen M0+IDW mean, old Gaussian prior/amplitude/tau. New generator architecture remains small ResUNet,71 inputs. No C2019/D2020.

v01 GPU job3644321 FAILED only at budget gate after131s. Measured diffusion4.574s/update with recomputation, total conservative forecast >4h, so no scientific training began.

v02 changes only gradient recomputation to false. Same full5472hours, one epoch684updates/arm/fold, effective8hours and K4,18Heun35calls. Clone every v01 allocation into v02 history; total allocation cap remains14400s inclusive of131s. New submission requires --resume because history is inherited. No added budget.

304 full tests passed in48.24s; numerical output/gradient equality of full graph vs checkpointed diffusion passed. RTX4060 probe for2members144x96 full backward peak3876.69MiB, gradient finite/nonzero; actual A800 benchmark still required.

Release: fc6427b071b64328064147d738d53ed28a2f77d8af2c331c13a3ce4c4c10dc0c
Archive: 282cd1736fa19df93a42e09286b868ea6dbaaed574c5d9351df4dc039fa41b74

Client submission: dist/20261121_grid_generation_v02_submission; copied to C:/Users/user/AppData/Local/Temp/gnss_gen3_submit_v02. Visible legacy console title GNSS DF GAN A800 v02. Uses per-command Python TCP transport to avoid direct OpenSSH route timeouts, keeps strict host-key verification; stores no password, no global network changes. Authentication completed; job 3645090 submitted on gpu8001, one A800. Live SSH snapshot stream is active.

Server root: $HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117
New release: code/20261121_grid_generation_v02
New run: runs/20261121_us_dense_zwd_gen3_unet_df_gan_v02
Old input/cache source: runs/20261121_us_dense_zwd_grid69_processor_pair_v01, read-only.

After login: deployment verifies all bytes and old audit sources, copies accounting history, submits bounded one-A800 job. Then benchmark -> budget gate -> all-three true200 vs100+100 engineering checks -> fold0 stochastic/DF/GAN -> fold1 stochastic/DF/GAN -> pilot outer96 K32 evaluations and figures -> COMPLETE.

Watch temp submit.log GRID_SNAPSHOT records. New local monitor should use port8783, read new temp log and root. Before ending supervision: scientific weights truly update, first200-step finite inner check, log grows, visible page updates. Do not wait for full queue. If budget gate still stops, keep artifacts and do not expand budget or reduce scientific scope silently.

Pause: touch GRID_RUN/PAUSE. Resume using same v02 root and release with tools/grid_generation_submit.py --resume; controller deducts all previous sacctElapsedRaw. SSH disconnect does not stop Slurm.

Artifacts: point/joint candidates, outer forecast NPZ, ES/VS/CRPS/coverage/width/meanMCSE, same-cohort Gaussian difference, shuffled-member control, representative field mean/std/member NPZ/SVG and training/calibration SVG.


## Live acceptance in progress

Verified at 2026-11-22 00:22 Asia/Shanghai: job3645090 RUNNING on gpu8001; queue ENGINEERING_RECOVERY_CHECK. Benchmark and original-cap budget gate passed because execution reached this stage. All-three actual GPU continuous200 vs100+100 checks are in progress; scientific fold logs remain empty. Do not report engineering updates as scientific training or completed startup acceptance.

Monitor: http://127.0.0.1:8783/ (built-in browser tab3); API and visible queue heartbeat verified. Remote stream: C:/Users/user/AppData/Local/Temp/gnss_gen3_submit_v02/submit.log. Local latest snapshot: results/20261121_grid_generation_v02/latest_server_snapshot.json. Prior allocation131s remains charged against same14400s cap.

Runtime GAN objective is ES + 0.1*adversarial, as locked runner line59 and implemented protocol. The JSON informational adversarial_weight=0.01 is unused and stale; source is authoritative for this release. Preserve current immutable release; report the actual0.1 and align this dead config field in a later release. No effect on stochastic/diffusion.


## Startup acceptance passed

Verified 2026-11-22 00:42 Asia/Shanghai: job3645090 RUNNING on gpu8001, one A800. All-three actual GPU exact-resume checks and bounded benchmark gate passed; fold_0/stochastic has scientific logs at100,200,300 and a finite step200 startup evaluation. Fixed inner96 cohort244stations/22421station-hours: RMSE7.4028736265 ->7.3633789320mm; groupES4.8980572922 ->4.8648346904mm; CRPS4.4780769646 ->4.4508807249mm; mean correction MCSE0.0072760009mm. Fixed-draw evaluated changes confirm learned correction. Startup is excluded from selection and is not outer-fold evidence.

Actual UI verified step300, startup banner, live heartbeat, loss/metric chart regions and JSON training log. Monitor tab retained. Acceptance evidence: startup_acceptance.json, startup_snapshot.json. Scientific queue continues fold0 stochastic ->diffusion ->gan, then fold1 same order, each684updates, then pilot evaluations and figures. No C/D data. End interactive supervision now; do not wait for whole queue.

No separate native metrics console was created; live page provides log/curves. SSH disconnect does not stop Slurm. Preserve the live SSH window for status sync if possible; no password stored.

Server logs: runs/20261121_us_dense_zwd_gen3_unet_df_gan_v02/logs/slurm_*.out; training and candidates under fold_0/stochastic, fold_0/diffusion, fold_0/gan (similarly fold_1). Source initial.pt/last.pt and selected candidate paths are recorded by engine products and selection.lock.json.

Pause on server: touch "$GRID_RUN/PAUSE". For resume, remove only the exact project's PAUSE marker after verifying root, then use existing immutable release's grid_generation_submit.py --root "$GRID_RUN" --release "$GRID_RELEASE" --python "$GRID_PYTHON" --resume. The submitter deducts prior allocation history; do not use a new campaign to bypass cap.


## Queue completion recorded

2026-11-22 03:15:43 Asia/Shanghai snapshot: queue COMPLETE, all six runs completed684updates each;4104scientific updates. GENERATION_QUEUE_COMPLETE present in Slurm log. Captured final sacct still RUNNING at03:09:16 because monitor sampled immediately before process exit; current scheduler completion needs fresh SSH. Snapshot archived as completion_snapshot.json, compact inner metrics completion_status.json. Remote outer result files, report and grid artifacts require download before scientific ranking. No new training submitted.
