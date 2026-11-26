# Batched evaluation performance correction

The evaluator now processes up to sixteen hours in a shared model invocation. This increases the decoder working set while retaining the registered evaluation population, member counts, random streams, model architecture, and training epochs.

## Measured baseline and cause

The downloaded A800 measurement in `results/20261122_full_generative_design_v01/server_downloads/e5cddac74c7e/benchmark.json` reports K128 final evaluation costs of 0.4364438603 seconds/hour for VAE, 0.4280426457 for GAN, and 0.5875857792 for diffusion. Its estimated final evaluation total is 206002.7474 seconds (57.22 hours). The original evaluator called the encoder, latent sampler and decoder separately for every hour, with decoder member chunks of four. This produces many small launches and repeated host/device transfers.

## Runtime controls

`FullTrainingConfig.evaluation_hours_batch` defaults to 1, preserving the existing hourly behavior. Set 8 or 16 to combine hours. The allowed range is 1 through 16. Existing `evaluation_chunk` controls decoder members per hour; its effective value is bounded by `floor(256 / actual_hours_in_batch)` so each decoder invocation processes at most 256 hour/member pairs. Useful benchmark configurations are B8/chunk16 (128 pairs), B8/chunk32 (256 pairs), B16/chunk8 (128 pairs), and B16/chunk16 (256 pairs). Member count remains independent of the decoding chunk; every requested member is decoded.

The B-only prior and withheld-posterior diagnostics accept the same hour-batch and member-chunk controls. `_execute_pilot` forwards `runtime.evaluation_hours_batch` and `runtime.evaluation_chunk` to these scorers. Their established defaults remain 1 and 4. The prior score retains its original IID K16 probability pool and independent K32 point pool. The withheld-posterior score retains all sixteen posterior members, whole held-group masks and per-hour checks for a nonempty fitting population.

## Scientific and recovery contracts

- Hour order, station order, masks, empty hours, sanitized labels and locked-population checks remain intact. Batched geometry uses the data API's existing static station geometry.
- Keyed noise uses the same hour IDs, seeds, member indices and latent dimensions. The independent antithetic point stream and IID probability stream remain separate. Gaussian observation discrepancy is added once to probability samples; the point mean retains analytical zero-mean discrepancy.
- Point estimates reduce only the member axis of each hour. Antithetic MC standard errors still use adjacent pair means. Mixture NLL still uses clean-field components and the original fixed effective Gaussian discrepancy. Coarse estimates still coarsen each hour's mean field once.
- The existing 168-hour blocks, content-addressed immutable NPZ files, atomic JSON pointers, checksums, lock identities and partial-prefix recovery are retained. Interrupts are checked at hour-batch boundaries, at most sixteen hours apart. A completed prefix can resume with another hour-batch or member-chunk setting under the same evaluation source. Source changes continue to invalidate old source-bound evaluation locks.
- This change adds one training-config field. It does not edit model architecture, epochs, budgets, sampling steps, training populations or cohort definitions.

## Verification

Targeted command (PowerShell):

```powershell
$env:PYTHONPATH='src;tools'
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_full_generative_evaluation.py tests/test_full_generative_diagnostics.py -q
```

Result: **36 passed in 9.46 seconds** on the local CPU environment. New batch tests were observed failing on the hourly implementation before implementation. The suite includes B1 versus B4/B8/B16 comparisons; K128 probability and K64 point pools; all saved arrays and resulting metrics with absolute tolerance 0.001 and zero relative tolerance; unsorted hour order; partial and fully empty masks; locked masks and missing-label rejection; exact equality of hour/member noise draws across batch shapes; 128/256 decoder-pair bounds; prefix interruption and resume; and B-only prior/posterior equivalence with whole-group withholding. Native VAE, GAN and diffusion models are compared using the registered eighteen diffusion steps, with station arrays within 0.001 mm. Existing pause-after-three-hours behavior passes with the default hour batch of one.

An exploratory untrained diffusion check with two sampling steps amplified floating-point differences to 0.1855 mm; the production eighteen-step test passes the unchanged 0.001-mm tolerance. The two-step configuration is not the registered scientific sampler.

The A800 baseline above is an existing measured result. No new server, SSH or CUDA job was launched for this patch. New A800 throughput, peak memory and numerical differences must be measured for each runtime tile before claiming a speedup or replacing the original cost estimate. Local correctness tests establish equivalent outputs for the tested configurations; they do not establish A800 throughput.
