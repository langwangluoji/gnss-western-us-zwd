# Task 1: self-contained neural models

Own only src/gnss_field/models/ and tests/test_models*.py. No subagents. Do not edit legacy code.
Write and run meaningful failing tests before implementation. Report to docs/task-01-report.md.
Project isn't Git; no commit/init. Python D:/anaconda3-2025.12-2/envs/pytorch/python.exe; pytest available.

## Inputs and exact interfaces
Parent will create gnss_field.contracts.ContextBatch dataclass with tensors:
hours[B]int64, stations[B,N,17], padding[B,N]bool, coarse[B,4,15,Hc,Wc], maps[B,6,H,W], baseline[B,1,H,W],
grid_id:str,scenario_hash:str. stations 5 standardizedvalues+5validflags+(x/400km,y/400km,H/3000)+4cycles.
maps H/3000,x/400km,y/400km,support,nearest/400km clipped2.5,count/16. Coarse ch0:5 normalizedvalues,
5support,6height,7x,8y,9:13cycles,13near,14count. No targets. Parent builds legacy patch dictionaries
stations,padding,coarse,maps,baseline,baseline_condition,mu_condition,domain and does full mosaic.

- models.mean.MeanField(): exact stand-alone extraction of current field_generation_v1.model.MeanField,
  forward(patch_dict)->(field_z0_mm[B,1,24,24],None). Preserve state_dict keys for parity checkpoint.
  Resolve neural class deps ONLY from architecture_search,regional_temporal_mae,mae_gan_cvae,cvae_pilot,
  moment_superresolution etc; extract only pure network classes/methods into models/legacy_core.py;
  NO runtime import of historical scripts or hidden data loaders. No pretrained weights used scientifically.
- models.pixel.ResidualField(): current M2 model standalone, forward(patch_dict)->residual,features32;
  standard_deviation(features,geometry,scale,faithful); faithful losses detach mean/features as existing.
- models.condition.ConditionEncoder(condition_dim=128): forward(context:ContextBatch)->[B,128].
  Station17->64->64+amplitude5->64; null token allmissing. coarseflatten60->32->32.
  Geometry attention at coarse grid:64dim4head relative dx,dy,dH,distance bias4->16->4.
  concatattn64+coarse32+downsamplebaseline(or mean if context has mean attribute optional)1+maps6=103,
  conv103->64->64,adaptive4x4 flatten1024->128. Independent fixed support not truth masks.
- models.basis.SpatialBasis(rank=128): forward(xyh[G,3]physicalmetres,land_mask[G]bool)->Phi[G,R].
  27features=xy/400km,H/3000,+24sin/cos at50,100,200,400,800,1600km wavelengths; hidden256,256,linearR;
  normalize each output columnRMS on supplied fixed land. Samewholegrid always, no direct stationdecoder.
- models.heads.DeterministicHead(rank,condition_dim=128,width=256): forward(h)->normalizedc[B,R].
- models.heads.GaussianHead(...): forward(h)->(mean,log_std), logstdclamp[-5,3].
- models.heads.GANGenerator(rank,...): forward(noise[B,R],h)->c; ConditionalCritic(rank,...): forward(c,h)->[B].
- models.diffusion.EDMHead(rank,...): denoise(x[B,R],sigma scalaror[B],h[B,128])->clean;
  forward aliasdenoise; sample(h,hour_ids,members,seed)->[B,K,R] DIFFERENTIABLE for ES training.
  StandardEDM.5 and 4resMLPblocks256 FiLMcondition128+sigma64. No spatialmeaning coefficientindices.
- models.diffusion.edm_coefficients(sigma,sigma_data=.5): existing5coefficients.
- models.diffusion.keyed_noise(hour_ids,members,seed,shape=(R,),device=...,dtype=...): deterministic CPU keyed streams,
  fixed prefixes,batch-independent; adapt existing pure function. heun_sample supports tensorstates no no_grad.

Keep implementations focused. Tests: portable no legacy imports at runtime; M0 state_dict/output parity against
actual old MeanField on CPU (legacy path in tests only), 0.001mm; M2 param/forward/faithfulgrad parity;
empty/padded/permuted stationencoder; basisactual rankcapacity; Gaussianvariance; EDMformula,35calls,
sample prefix/batch, gradient all the way through35calls; all headshapes.
Do not run full training or paid/remote jobs. Include exact tests/results and unresolved concerns in report.
