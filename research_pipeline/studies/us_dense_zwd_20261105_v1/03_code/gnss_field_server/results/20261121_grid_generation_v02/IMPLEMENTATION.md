# Runtime optimization v02

Parent3644321 stopped at the budget gate after131s, before scientific training. Benchmark: stochastic0.1653s/update, diffusion4.5742s/update, GAN0.5365s/update; peak GPU4.25GiB with rematerialization. Forecast including1.5margin exceeded4h.

Ruling: disable gradient recomputation only, using A80080GB memory to reduce repeated denoiser evaluations. Keep full data,684updates/arm/fold,effective8,K4,18Heun and FP32 unchanged. New immutable code/run v02; all old allocations copied and deducted from the same14400s ceiling. Fresh benchmark required. No unlimited retry or expanded budget.

Mean remains audited frozen M0+IDW; old U-Net mean not promoted due to both outer-fold regressions.

Minor: Unused GAN config field adversarial_weight=0.01 differs from actual immutable runner ES+0.1*adversarial. Actual0.1 matches implemented protocol; report actual value. Align unused field in future release, preserve running bytes and all hashes.
