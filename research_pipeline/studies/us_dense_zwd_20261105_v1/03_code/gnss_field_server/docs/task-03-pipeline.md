# Task 3: production stage pipeline and promotion

Own src/gnss_field/training/pipeline.py, tests/test_pipeline*.py; may extend engine.py for GANauxoptimizer+
before_update withfullcheckpoint/recovery tests. Parent owns CLI/benchmark/monitor and baselines.py.
No agents, legacy writes, remotejobs, full30ktraining. Testsfirst; report docs/task-03-report.md.
Projectroot is this brief's parent.parent; standardPython D:/anaconda3-2025.12-2/envs/pytorch/python.exe.

## Context available (READ these implemented APIs)
contracts ScienceConfig/RuntimeConfig(defaultCPU7), ContextBatch physicalmean/coarse_z0;
data.PackedStore,ContextBuilder,InputScenario; data.manifests TrainingManifest/build_training_manifest,
stratified_hours/spatial_folds. Real packeddata study/02_data/processed/gnss_field_server_v1 exists.
Parent local_validation/training_manifest.npz qualified17526base,8736residual,8712R128targethours.
training.fields predict_patches(model,context,store,chunk=8,residual=False,scale=1) fullglobal77patch,
MeanCache builds explicitmonthly caches; defaultscenario hash currently 'normal' mustpassInputScenario.identity.
training.system FunctionalModel(arm,rank,frozen_encoder) andFrozenDecoder(phi,shape,transform,s0);
FunctionalModel.condition scalesM0meanandbaselinebys0 beforeencoder; .coefficients h,ids,K,seed retains35NFEgrad.
objectives fit_latent FP64differentiable, CoefficientTransform, energy_scoreK4Ustat, edm_loss,gradient_penalty.
productionengine TrainEngine/StageSpec includes model/EMA/optim/RNG/cursor/pending_eval/products;
actualEDM100+resume100=200 testpassed. Needextendauxoptimizer/before_updateforGANncritic5 once/effectivebatch,
not5critstepspermicrobatch. All modeltypes standalone models/ ; pureMeanField legacydictforward,ResidualFieldM2.
training.prediction evaluate_predictor(storeDEV/C, hours, predictor, output,...), mean_predictor,
functional_predictor,pixel_predictor. Returns {'metrics':bundle,'values':selection_values,...}.
Evaltypes exact keys: primary.station_macro_rmse_mm,primary.station_macro_crps_mm;
joint.group_macro_es_mm/.group_macro_vs_mm; fine.model/coarsened pointmetrics;
fine.model_increments/coarsened_increments['0-50'].group_macro_rmse_mm.
metrics allsurfaceMM. Missing bins None not fabricated. evaluation.select_products handlespoint/joint/marginal,
full2019records; sampling_stability; build_product_lock; direction_gate; probability_gate.

## Class contract for CLI
TrainingPipeline(science, runtime, directory, seed=42, *, legacy_source=None, parent_directory=None)
directory=campaignroot (resolved.json there); seed artifacts underdirectory/seed_<seed>/...
run_task(alias, *,arm=None,fold=None,variant=None,seed=None)->dict[str,Path] outputpaths forreceipts.
Aliasesownedpipeline: mean_base,mean_all,mean_cache,basis_cv,representation_gate,basis_final,condition,search,promote,
full2019_rerank,stability,direction_gate,ablation,replicate,product_lock,evaluate2020,export.
ParentCLI owns pack/audit/doctor/benchmark/budget_gate/report andcalls yourrun_task otherwise.
Raise ScientificGateFailed (definehere) onsciencegate, PauseRequested onplannedpause; neverstubsuccess.
Evaluation2020mustvalidatecurrentlockandhashes beforeDview construction. No paidjobsubmission.

## Exact scientific flow
M0base2016-17fromscratch, M0all2016-18separatestrongbaseline, inputstats2016-17 both; budget30keff8h,
stationbalancedMSE basedmanifestcounts, trueBmask+fixeds0surface scale, .001curvature supportedstencils.
Trainhourfull77patchthenqueryallB. SelectEMAcheckpoint top3C2019screen576, full2019rerank beforefreezingM0.
NeveruseoldM0step1021asnewbase. Cachechosenbaseall2018+2019;2020cacheafterlockonly.

Basis3foldB50kmgroups within2018: fitbasisandcoefsusingfitfoldcurrentBonly; M0pastBhistoryalreadyseen,
so this isrepresentationinternal notindependentstations. AtleastR+8fitBperhour. Fitloss fromfit_latent+curv+ridge,
s0baseZ0std. Crossfoldeval2018calendarstratified576 onheldB, compareM0andself100km point+0-50increments,
effectiveobservrank tol1e-6>=R/2 on>=95%selecteddiagnostichours. Failgate stopsnotautoexpand.
Finalbasis10k,freezeEMAphi,refitcoeffsallqualified2018hours, savec/hours/transform(Phi,Cmean,2RMSactive),
neuralweights/gridhash/source. SavebasisPhiandHderivedfromPUBLISHEDgrid query notdirectneuralstationcoords.

Sharedencoder+Det: pretraincoefficientMSE5k thenrealstationMSE15k, encodertrainable;
thenfull2019pointselectconcretecheckpoint, frozenencoderhashsharedallheads. Condition+hcache explicitmonthly;
ifnotneeded cachingfirstimplementation inferfrozenhonceeachbatch, but addcachefor repeatedeval iffeasible.
HeadsDet/Gauss/EDM/GAN usefrozenM0,basis,hsource; Det5k+15k;Gauss10k+10k;EDM/GAN20k+10k.
Estimatedcoefs onlypretrain, allfinalgenheadsESreal2018BusingH_B exactly andsameHeun18K4.
Trainnoiseseed variesstep (notalwayssamehournosies); evaluationfixedseed/K16then64/128 prefixes.
GANcritic5updates per generator effective32h,GP10Adam0,.9,criticfrozenESstage; criticoptimizerRNGsaved.
M2-F/J20krealB each, s0Z0residual, surfaceSTDscale=store.stats.surface_std, FmeanMSE+detachedNLLfeature,
optimizers/clipsseparate, sameinit/data acrossF/J. Point/marginalCRPSselection, nofakejointES.

## Search/promotion
orchestration.run_study(directory,arm,objective,trials=8,...) callback(params,trialdir,num)->
{'point':{'value':RMSE,'checkpoint':path}, 'joint' OR'marginal':..., 'pretrain_checkpoint':path,'contract':path}.
Use25%budget caps butfullhorizon schedules. Trial0baseline registered. Savepretrainlast separately evenafterES.
M2-F onlyobservationaltraining: noestimatedpretrain. Treat itslast warm-up checkpointas validobservationwarmstart;
parentwilladaptvalidator toallowm2_fobservational source, donotfalselymarkES headaspretrain.
Completeconfigbestperproduct union<=2, fullpromotionforkPRETRAINmodel/opt/RNG/cursor andextendtofullhorizon,
thenNEWfullobservationphase. DoNOTsendESweightsbackpretrain. Sourcequarterbudgetdiff requiresverifiedforkpath
explicit stage/data/seed/arch/horizon/hyperparams equality, immutableoriginaltrials; nohashbypass.
M2Jfullruns selectedFconfigs fromsameinitialweights notfromtrainedFweights.

Full2019eachproducttop3rerankK64, addK128 when .005mm delta/order/gatechange;ifstillunstablepointflag,
lockchosenK andallparams before2020. Bstar candidates coarsebilinear,A_IDW,A_kriging,M0base,M0all,
M0innovationIDW,M2F/Jpointbest,Detpointbest,Gaussianpointbest. Parentbaselines API forthcoming
training.baselines.get_baseline_predictor(name,store,runtime,*,mean_cache=None,parameters=None)->callback;
fit_geostat(store,destination)->dictparameters path. No contemporaryBforbaseline.
Pointfinegate4strict2019signimprovements vsBstar+self100. JointgateESandVSlowerthanGaussianjoint_best,
CRPS<= andcoverageengineeringnominal+/-3pp; reportwidth. TheseCdirectionfiltersnotformalconfirmation.
Onlypassedfamily advances correspondingablations/seeds;GANindependentgate. Ifnoboth STOPPED_NO_GAIN notFAIL.

## Ablation/seed/export
variants pretrain_only (evaluatefixedpretraincheckpoint), no_geometry_bias(newsharedencoderrelativebiaszero),
current_only(retrainM0/entireaffectedchain withpastframeval/maskzero), rank64/rank256(rebuildbasis/condition/heads,
sameM0 whenhistoryunchanged;R256needrankgate), joint_encoder(startsamepretrainedEDM,encoderlr1e-5/head1e-4).
Onefactorseed42, nofactorial. Repeats17/101retrainM0,basis,conditionandlockedchosenheadconfigs, notOptunaagain;
sharefoundationswithinseed. Formal2020bothproducts+baselineandbootstrap viaevaluation APIs, no2020modelselection.
Exportwinningmean/stdgrids andeventmembers only, stationensembles alreadysavedprediction service.
Artifacts allsourcehashes configEMAphaseRNG ref; no untested claimofscientificsuccess.

## Tests minimum
Actualtrainpaths on tiny synthetic context for unitCI and realdataengineering1step separateparent smoke;
phase dispatchcomplete/no unimplementedaliases, role/yearforbidden, frozenfoundationweightshashunchanged,
ESupdatesheadthrough35calls, M2Fmean independentvargrad, promotionNEVERESintopretrain,
restarttop3association, pretraincandidatesexcluded, 2020requireslock, stagepauseentry preventsnextupdates.
No requirementtoactuallyrun30ktraining locally. No standalone dataloaderextraimportsor unknownaccountsettings.
