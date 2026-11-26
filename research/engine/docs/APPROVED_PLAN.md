# Tongji GNSS field server implementation specification

The user approved native PyTorch + Slurm, full-domain training, shared functional residual EDM,
observation-space energy-score refinement, Gaussian/GAN controls, role isolation, dual selection,
8-trial studies, full-year reranking, gated ablation and seed replication. Preserve legacy outputs.

## Immutable scientific contract
- A174 input, B511 train, C110 development, D110 retrospective. Nominal hourly time; EGM96.
- 2016-17 M0/stats; 2018 residual/basis/heads; C2019 selection; D2020 only after lock.
- LCC grid 144x96 at25km; 77 patches24x24, stride12. Sine-square normalized mosaic.
- Surface query = exp(-H/3000) * bilinear published Z0. Never query a neural decoder at station coordinates directly.
- Current and t-6/12/18h coarse fields are A-only IDW; five station variables jointly valid.
- Training B requires real finite surface ZWD, fixed valid query geometry, and current A support.
- Support: nearest A<=100km, >=3 A within150km, A hull, <=100km from fixed B geometry, complete stencil.
- Mean uses >=1 target/hour; latent fitting >=R+8; observation ES >=4. Evaluation F/S0/T unaffected by these training thresholds.
- Basis27->256->256->R, mainR128. Coordinate inputs xy,H plus sin/cos xy wavelengths50/100/200/400/800/1600km.
- Ridge in FP64, .01/R coefficient penalty and .001 supported curvature; s0 from2016-17 B Z0 std.
- Coefficients estimated from2018 B only are pretraining material, not dense truth.
- Freeze chosen M0, basis and selected shared condition encoder before distribution-head comparisons.
- Standard EDM sigma_data=.5, Pmean=-1.2 Pstd1.2; Heun18 rho7 .002..80 churn0 lastEuler35NFE.
- Gaussian/EDM/GAN observation refinement: K4 unbiased energy score, SAME 18step sampler, real B observations.
- Each hour/member has one global coefficient vector; no patchwise independent samples or time-trajectory claims.

## Workload and selection
M0/M0-all30k effective8h; basis3fold3k then10k effective16h; shared Det5k latent+15k observed effective32h;
M2-F/J20k effective8h; Gaussian10k+10k, EDM/GAN20k+10k effective32h.
AdamW1e-4 wd1e-4 EMA.999 clip1 warmup500 cosine1e-6; GAN adversarial Adam(0,.9), GP10 critic5.
Train log100; formal dev1000; startup200 excluded. Observation early stop after5k and6 stagnant evaluations in ALL products.
Point RMSE, joint group ES, marginal M2 CRPS products kept separately. ScreeningK16, top3/product full2019K64;
K128 prefix stability threshold .005mm or ranking/gate change. Never choose on2020.
Optuna8trials each Det/M2-F/Gaussian/EDM/GAN; TPEstartup4 seed20261117; short25% full schedule;
pretrain lr3e-5..3e-4 and observation lr1e-5..1e-4. No multiobjective pruning/trial.report.
One JournalStorage owner/study, sequential trials; fail3 same errors stops. Resume RUNNING first.
Promotion forks saved pretraining checkpoint and then fresh full observation refinement, not ES weights back into pretraining.

## Deployment boundary
Default L40 one GPU,8CPU,64GB,24h job, concurrency2 via phase arrays/barriers. No DDP.
Initial <=4L40cardhour benchmark ONLY. Formal submission requires locked total/per-stage budgets.
No paid jobs, SSH connection, email or remote installation during local implementation.
Linux Python3.11 PyTorch2.8cu126 baseline; verify actual driver/partitions/account before deployment.
Use persistent home for releases/artifacts; ssdfs for rebuildable hot cache. Journal/checkpoints single writer and atomic.
Checkpoint all RNG/cursor/phase/pending eval/product candidates/elapsed attempts; --signal=USR1@180 reaches job steps.
Pauses and failed gates block downstream; resume regenerates remaining DAG, no unlimited requeue.

## Acceptance
Legacy M0 parity<=.001mm; all contributor gradients; no C/D leakage after selection lock;
FP64 ridge/rank/mask tests; exact latent query<=.001mm; differentiable35NFE; keyed sample prefix/batch tests;
actual stage100+resume100=200; dual checkpoint association; Optuna/Slurm recovery; finite failures never silently dropped.
Evaluation: station macro/micro errors; 0-50km group increments; globally aligned4x4 coarsening;
CRPS,80/90/95 coverage/width, group ES/VS; SSIM7/spectra diagnostics; paired7day block bootstrap2000 with14day sensitivity.
No full-year member grids for every trial. Save station ensembles, winning mean/std grids, predeclared events.
