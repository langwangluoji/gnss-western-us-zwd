# Full generative implementation interfaces

All tensor dimensions are batch-first, all coordinates are projected metres unless explicitly normalized, and all labels are surface ZWD millimetres. New package src/gnss_field/generative. Public neural modules are freshly defined, not historical subclass wrappers. Existing pure numerical artifact, diffusion sampler and metric utilities may be reused with contracts verified.

## Observation arrays and compact sample

A compact dataset item/batch dictionary has:

- `hour_ids`:int64[B].
- `grid_condition`:float32[B,37,H,W], defaultH144,W96.
- `station_tokens`:float32[B,A,31], defaultA174.
- `station_mask`:bool[B,A], any four-frame jointly valid input for attention; per-frame flags remain in tokens.
- `grid_geometry`:float32[H,W,3], projected x/y and EGM96 height, static.
- `input_geometry`:float32[A,3], projected x/y/height, static.
- `background`:float32[B,1,H,W], common-height mm.
- `query_indices`:int64[N,4], flattened pixel indices in the same published grid.
- `query_weights`:float32[N,4], PURE bilinear weights summing1; altitude not absorbed.
- `target_geometry`:float32[N,3], projected x/y/height, static.
- `target`:float32[B,N], real surface ZWD labels; zeros at invalid positions, validity separate.
- `target_mask`:bool[B,N], valid labels AND fixed query geometry AND current A-derived support. Conditions never read it.
- `support`:bool[B,H,W], current-input/fixed-geometry support, no current target labels.

Compact monthly files may retain coarse20@H/4,W/4, raw normalized four-frame A values/flags, background, station-distance/density and B labels separately instead of expanded37@H,W. `collate` reconstructs the public batch on the requested device. Static geometry is not repeated to disk per hour.

##37 grid channels, in exact order

0:20=frames0/6/12/18h × five jointly valid normalized input variables; coarse IDW at100km centres, bilinear to25km.20:24=per-frame valid A fractions, spatial broadcast.24=height/3000.25:27=east/north slope divided0.1.27:29=domain-normalized x/y.29=land.30=normalized frozen fusion background.31=min(current nearest input distance,1000km)/150km.32=log1p(current A count within150km)/log(175).33:37=sin/cos day and sin/cos year.

##31 station channels, in exact order

0:20=four frames×five normalized variables, invalid values zero after normalization.20:24=four jointly valid flags.24:26=normalized x/y.26=height/3000.27:31=day/year sin/cos. Frame offsets0,6,12,18h.

## Neural API, architecture.py

`EncodedCondition(pyramid:tuple[Tensor,...], global_features:Tensor)`; pyramid widths32/64/96/128/192, sizes(H,W),(H/2,W/2),(H/4,W/4),(H/8,W/8),(H/16,W/16); global_features[B,192].

`ConditionEncoder().forward(grid_condition,station_tokens,station_mask,grid_geometry,input_geometry)->EncodedCondition`. Five-scale blocks per design, GroupNorm8+SiLU, two Conv3x3 per residual block. Station MLP31→64→64. AtH/4,W/4:Q96→64,K/V64→64,4 heads16,output64→96. Geometry bias from projected distance/150km and height difference/3000; explicit null/empty-set handling. Input dimensions are validated.

`FieldDecoder(latent_dim=64,field_scale_mm=10.).forward(encoded,z,background)->Tensor[B,K,1,H,W]`, z[B,K,R]. LatentR→256→32*(H/16)*(W/16); default32×9×6. Concatenations224→192,320→128,224→96,160→64,96→32,Conv1x132→1. Each stage has latentFiLM, small nonzero output initialization. No Gaussian-field identity bypass, no enforced zero ensemble mean. Support variable H/W multiples16 for tests; constructor `grid_shape=(144,96)`.

`ConditionalFieldVAE(latent_dim=64,grid_shape=(144,96))` has `.encoder`,`.decoder`,`.prior` and`.posterior` modules.

- `.encode(batch)->EncodedCondition` reads only condition keys.
- `.prior_parameters(encoded)->(mu,log_std)` each[B,R];192→256→2R.
- `.posterior_parameters(encoded,residual_surface,query_indices,query_weights,target_geometry,target_mask)->(mu,log_std)`. Query first-pyramid32 channels,Linear32→64; concatenate residual/10,normalizedXYZ,mask to69; MLP69→128→128;masked mean+max256 plusglobal192→448→256→2R. Empty posterior must reject training, not produce NaN.
- `.sample(encoded,background,noise)->fields`, noise[B,K,R], conditional prior reparameterization.
- `.posterior_sample(encoded,background,residual_surface,query_indices,query_weights,target_geometry,target_mask,noise)->(fields,mu_q,logstd_q,mu_p,logstd_p)`.

`LatentGenerator(latent_dim=64,condition_dim=192,width=256)`:noise[B,K,R],condition[B,192]→standardized latent[B,K,R]. InputR+192→256,4residualMLP256 withLN/SiLU andcondition/noiseFiLM,output256→R. No identity-to-fixed-Gaussian bypass.

`LatentCritic(latent_dim=64,condition_dim=192)`:latent[B,R],condition[B,192]→score[B]. MLP(R+192)→256→256→128→1,LeakyReLU0.2,noBatchNorm; Wasserstein GP is applied by trainer, no simultaneous spectral-norm requirement.

`LatentDenoiser(latent_dim=64,condition_dim=192,width=256)` exposes `.denoise(noisy[B,R],sigma[B]or scalar,condition[B,192])->[B,R]` withEDM sigma_data1 and `.sample(condition,hour_ids,members,seed,steps=18)->[B,K,R]` using independent keyed initial noise.64sin/cos timefeatures,4FiLM residualMLP256.18 positive levels80→0.002,rho7,terminalzero,17Heun+finalEuler=35NFE. Existing pure heun_sample/keyed_noise may be reused.

`ObservationCritic()`:values[B,N]surface residual/10,local_condition[B,N,32],geometry[N,3],mask[B,N]→score[B]. Masked pointMLP36→64→64,mean+variance128; deterministic pairs in four distance bins0–25/25–50/50–100/100–200km with at most256pairs/bin; pair input36(local32difference,residualdifference,absolute difference,distance,heightdifference),MLP36→64→64,pool4×64=256;combined384→128→1. Empty pair bins zero; real/fake masks and pair selection identical. WGAN-GP acts only on observed residual values.

## observations.py

`query_surface(fields,query_indices,query_weights,heights)->Tensor[B,K,N]`, accepts fields[B,K,1,H,W]or[B,1,H,W] (K1). Pure interpolation thenexp(-H/3000); fail onnonfinite predictions. `query_features(features[B,C,H,W],indices,weights)->[B,N,C]` no height factor.

`diagonal_gaussian_kl(mu_q,logstd_q,mu_p,logstd_p)->[B]` SUM latent dimensions.

`variational_loss(predictions[B,K,N],target[B,N],mask[B,N],mu_q,logstd_q,mu_p,logstd_p,sigma_eff_mm,beta=1.,reference_count=None)->dict(loss,nll,kl,kl_per_dim)` uses Gaussian NLL including log sigma and constant. One COMMON scalar reference count (training-only median eligible count) divides both NLLsum andKLsum; equal per-hour aggregation. No variable station weights in first release. Invalid label slots sanitized before arithmetic. Effective discrepancy is a working model scale, fixed per run; not asserted sensor noise.

`sample_surface(fields,batch,noise[B,K,N],sigma_eff_mm)` calls query_surface then adds sigma_eff_mm*noise ONCE. Field exports omit independent station noise.

## data.py public API

`FieldDataConfig` dataclass or plain dict with packed_root,cache_root,background_assets,train_years(2016,2017,2018),development_year2019,frame_offsets(0,6,12,18).

`FullFieldData(config,purpose='train')` supplies `.train_hours`,`.development_hours`,`.metadata` and `.batch(hour_ids,role='B',device='cpu')->batchdict`. `.condition_batch(hour_ids,device)` NEVER reads B/C/D target values. Explicit cache prepare/audit via `.prepare(hours,...)` or module`prepare_cache(config,...)`; role gates inherit PackedStore. Geometry labels metadata may be read, values isolated. Source schema field names, units and statistics are frozen in a manifest. Synthetic compact fixture must work without production paths.

Background is an external frozen source, not trained by new classes. Reuse old verified fusion/geometry pure operator only through explicit asset paths, weights/hash validation and no old training-script imports. Full-data cache prep is resumable/atomic and does not overwrite previous experiments.

## training.py public API

`FullTrainingConfig` includesstage,model_family,seed42,epochs,min_epochs,batch_size,lr,minimum_lr,warmup_updates,patience,output_dir,sigma_eff_mm,latent_dim,data_config,dependency_paths,budget_seconds. Separate scientific config from paths/chunk sizes. Stages:deterministic_control,vae_representation,posterior_cache,gan_pretrain,diffusion_pretrain,prior_es,complete_model_es. Model families:vae,gan,diffusion.

`run_stage(config,data=None)->dict` may receive a synthetic data fixture withsamebatchAPI; never submits jobs. Periodic development calls are injected/callable to keep tests bounded. Stage output includes dependency hashes,checkpoint,step,epochs,cursor,raw/EMA,optimizer(s),RNG,best point/joint metadata and status. Use existing atomic artifact/RNG utilities, not historical trainer inheritance. Resume preserves allscientific state andpendingevaluation.

`make_model_and_objective` or equivalent accepts the publicdict and returns differentiable losses. Latent cache contains training-only global_features,posterior_mu,posterior_logstd,hour_ids andstandardization; decoder- or encoder-changing stages disable/invalidate this cache.

Training objective details:VAEvalidNLL+KL;GANWGAN-GP lambda10,ncritic3 thenproper real-stationES;EDMweightedDSM withlog_sigmaN(-1.2,1.2²),sigma_data1;ESK4 IID normalized bysqrt(validN)and10mm. Complete-modelES updates independent encoder/decoder copies; no stale DSM. Save original objective andsharedrepresentation products separately.

Root supplies evaluation.py `evaluate_model(model,data,hours,config)->MetricBundle` andcampaign.pySlurm/hash/budgetgates. Training may expose lower-level injectionpoints for synthetic tests rather than duplicate metric code.
