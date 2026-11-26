# Full conditional VAE/GAN/diffusion implementation plan

> For agentic workers: use subagent-driven-development with independent file ownership and TDD, then one integrated scientific and engineering review. User explicitly requests implementation and training after architecture introduction; no additional design approval round is required.

**Goal:** Deliver complete observation-supervised conditional VAE, latent GAN and latent EDM systems, full eligible 2016–2018 training, independent C2019 development, and a bounded sequential single-A800 campaign.

**Spec:** Final ARCHITECTURE.md plus INTERFACES.md; Pro reviews PRO_REVIEW_FULL.txt and PRO_FINAL_REVIEW.txt. The final design supersedes DESIGN_DRAFT.md; that file is retained as design history. Server submission follows local acceptance and immutable packaging.

**Tech stack:** Existing native PyTorch, NumPy/SciPy, monthly read-only NPY inputs, Slurm. This project is not a Git checkout; isolate new work under src/gnss_field/generative and versioned results/configs. Preserve old sources and experiments; identify the final release by file SHA256.

## Binding decisions

- Shared-representation comparison freezes one condition encoder/decoder and observation discrepancy kernel; all three priors receive matched real-station ES refinement.
- Complete-model refinement is a separate product/stage: three copies may update their own condition encoder, decoder and prior under real observations. Retain original-objective and shared-decoder checkpoints. No stale-posterior DSM after moving the decoder; describe final diffusion as observation-score-refined sampler.
- A Gaussian latent prior with a nonlinear decoder is not a Gaussian field distribution.
- Final observation discrepancy is fixed by training-only physical-group diagnostics; do not silently infer sensor noise from the old 6.1827mm nugget. Three registered candidate effective scales:3.18/6.36/12.72mm; record the likelihood as a working discrepancy model. No C/D labels determine these scales.
- Latent64 main design and latent128 diagnostic use identical registered training-only panels. Representation failure stops dependent priors; never train against manufactured dense truth.
- Four frame offsets are0/6/12/18 hours. All final training hours pass current input support and fixed query geometry with at least4 eligible B stations. Keep excluded-hour reasons.
- Probabilistic ES-U members are IID. Separate point-mean estimation may use antithetic noise; no same-batch ensemble recentering in probability scores.
- Full epochs, explicit minimum training, late learning-rate decay, checkpoint/optimizer/RNG/cursor recovery. Runtime benchmark covers cold/full-working-set input reads, writes and actual sampling backward.
- Single A800 only. Scientific phase caps are independent of allocation chunks; each allocation is bounded and stops safely. Total/stage budgets must be locked from measured work before full submission. No blind automatic requeue.

## Task1: neural architecture and observation math

Owner files: architecture.py, observations.py, tests/test_full_generative_architecture.py. Implement exact layer contracts in INTERFACES.md. Tests first: shapes, permutation/all-missing handling, height query equivalence, all gradient paths, nonzero random dependence, Gaussian KL analytic check, likelihood/KL reductions, independent keyed member prefixes. Run targeted tests and report actual evidence.

## Task2: compact full-data contract and cache

Owner files: data.py, tests/test_full_generative_data.py. No C/D labels in inference/context builder. Build37-grid and31-station features from raw A and static metadata. Monthly compact caches; training-only norms; physical support; query weights; full eligible lists and reason codes. Reuse only verified frozen background as an external component. No old69-channel cache. Test label perturbation, zero/single/missing input, exact time offsets, cache identity and query validity.

## Task3: staged training and recovery

Owner files: training.py, tests/test_full_generative_training.py. Consume architecture/data interfaces. Implement deterministic control, complete VAE ELBO, frozen-posterior cache, latent GAN WGAN-GP, latent EDM DSM, matched ES prior refinement and separately named complete-model refinement. Capture EMA, optimizer, RNG, shuffled epoch cursor, unfinished evaluation, stage and dependency hashes. Tests first, meaningful synthetic fitting and resumed-vs-continuous trajectories. Never launch server jobs.

## Task4: evaluation, full campaign and integration

Root owner: evaluation.py, campaign.py, __init__.py, tools/train_full_generative.py, config, Slurm templates, packaging and reports. Build fixed C2019 lists, dual products with step0 clearly identified, metrics, result/curve exports, budget and artifact gates. Test with real local small training-hour panel, verify background/query/caches and all three optimization paths. Save final architecture document with every layer and measured parameter counts. Present architecture to user before server submission.

## Review focus

1. Sparse station supervision never becomes fake dense truth; q labels never reach inference.
2. Same masks/conditions and discrepancy added exactly once for real-vs-generated comparison.
3. Decoder and condition updates invalidate latent targets/cache where required.
4. Mean-estimation variance is separated from IID probabilistic member scores.
5. Every budget retry consumes prior accounted allocations; terminal/paused/error states differ and stage ordering is fail-closed.

## Ledger

- Data audit complete:26304 raw training hours,26280 pass raw B>=4 before frame/geometry support,13132476 valid B station-hours.
- Pro full review and final targeted correction review retrieved; final markerGVD-FINAL-END. Two-layer shared-representation/full-model comparison adopted.
- Read-only technical review complete: shared frozen representation alone limits claims to prior comparison; separate complete-model refinement adopted.
- Old checkpoint sources and all old experiments unchanged. No new server jobs submitted.
- New full-field CUDA engineering acceptance completed on real B inputs. VAE representation, latent GAN and latent diffusion each passed continuous200 versus100+resume100 with exactly equal raw/EMA weights and CPU RNG.
- Original C2019 population reproduced:76 stations,642258 station-hours; new geometric masks cover every original locked score. D values remain sealed.
- Full raw/EMA development histories and formal point/joint top-three candidate pools added. Evaluation resumes content-addressed weekly blocks and partially completed prefixes.
- Formal training uses full epochs. Before any pilot, the actual A800 benchmark must pass the entire24card-hour plan and shared1card-hour preflight envelope.
