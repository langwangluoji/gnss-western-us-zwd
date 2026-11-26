# Task 4: complete common evaluation and dual-product selection

Own ONLY src/gnss_field/evaluation/ and tests/test_evaluation*.py. Parent supplies predictions/inference,
data and training. No agents, legacy edits, paidjobs. Read contracts.py +artifacts.py andgeometry/ forunits.
Testsfirst; report docs/task-04-report.md. Python D:/anaconda3-2025.12-2/envs/pytorch/python.exe.

## Required API choose simple array-based functions, document early
evaluate_arrays(hours[T],station_ids[N],truth[T,N],prediction[T,N],population_F[T,N]bool,
normal_support_S0[T,N]bool,*,samples[T,K,N]=None,std[T,N]=None,metadataDataFrame,
coarsened_prediction[T,N]=None, safe_T[N]or[T,N]=None,coarsened_samples=None)->dict.
Metadata subset matchesN andcontains x,y,height,dense_group,cohort; station_ids stableglobalindices.
All predictionsfinite forFULLF, samplesfiniteallF, failures raise NOTdrop; targetnanoutsideF allowed.
S0cannotshrinkF. Return metrics keys predictablepointprimary station_macro_rmse_mm (compute root ofMSEeachstationthenmean),
micro_rmse_mm,macro/microMAE,bias,P95,Nstations,Nstationhours; scopesfull,primary,unsupported.
PairincrementsZ0 (multiplysurfacepred/truth expH), groupedmacroRMSE dense_group50km, samepublishedfieldhour,
bins0-25/25-50/50-100/100-200km; sameheight<=100m surface check; mainfine0-50 incl across0/25bins,
need pairswithinS0and safeT, notrandomcross-hour/fold. Coarsenedpoint/pairs samecommonT population;
do notclaimfullareamean. Groups predeclared; nofinitepredictiondeletion.

Probability: empiricalCRPS finiteK distribution (V-stat denominatorK² standardforecast CRPS) stationmacro;
GaussiananalyticCRPS ifstd;80/90/95intervalcoverage+width fromnp.quantilemethodlinearHF7, stationmacropluscounts;
GaussianNLLand90coverageetc. JointES L2normpower1/sqrt(n), finiteensembleVstat forforecast evaluation
(traininglossusesUstat; explicitlydifferent), VS p.5 weight1/max(distancekm,25), normalizeweights;
jointdensegroupsn>=4sameglobalmemberID, groupmacroESselection. M2stdonly NO fakejointES/indepmarginalconstruction.
Nooracle member. AllF/S0 extendedtables currenttruevalidA currentteacherneverreadbytheseprimitives.

Image functionsdiagnostics: SSIM7uniformwindow fixedtrainingrange (noimageautoscale), commonmaskfullwindows,
PSD2d detrendHannenergycorrectedphysical25km withbins50-100/100-200/200-400/>400, totals/member/mean
separate callerlabels; phase-sensitiveerror plusradialpower, no claimhighfreqpowertruth.

Paired bootstrap2000seed20261117 same7daytimeblocks ALLstations, recompute stationmacroRMSEperrep;
14daysensitivityparam. Groupincrement resamplingcrossedtime+fixedgroupids, eventwholeblockbootstrap helper.
Pointdiff=model-baseline,95CI. Formalfinegate4allCIupper<0; pilotgate4differencesstrictly<0;
jointgateES&VSimproveandCRPS<=baseline,coverageengineering nominal+/-3pp alllevels,
formalcoverageconfirmationCIinsidebandelseindeterminate. Reportfamilyflagsseparate,nocompositescore.
Holm helperforpreregisteredselectedmodel/productclaimsp-values, notpselectfromrawtables.

Product selection: keep_top_candidates(records, products,top3); rerankrecords fixedpoint(RMSE),joint(ES),
marginal(CRPS), unrounded sort +step tie. finalpretrainingexcluded/startup200excluded. Lockrequiresfull2019rerank,
K64/128stable basedpointdelta>.005mmororder/gatechange; samplingstability helperrecord. BestBstarfromregisteredpointcontrols.
Artifact lockmetadataJSON includeconfig/checkpoint/manifest hashes &products; validate2020access requireslock
year2019selection androleD2020 no paramschanged. No actualdataI/OorlegacyFieldData constructor needed.
Protocol_v2.json generatedcopyexistingv1 runtime byparent; evaluatorsusefixedprotocol noout-of-scopeimageVAE.

Tests mathematicalhandcases macrovsdensemicro; samePSDphasecounterexample; unchangedCRPSbutshuffledmember
ES/VSworse; nojointM2; PICPfinitequantiles; pairedblocksameindices; coarsecommonTcounts; nofailuredeletion;
dualcheckpointselectioncorrect/no200pretrain; stabilityprefix; bootstraprecomputeRMSEratherthanavgerrors.
KeepcodeusableasAPI, nounimplementedstubs. Parentneeds predictablynamed evaluate_arrays, point_metrics,
energy_score_ensemble, variogram_score, crps_ensemble, paired_block_bootstrap, select_products, direction_gate.
Message exact signatures assoonfixed; functions mayincludeextrasfor diagnostics. No training/inferenceengineownedhere.
