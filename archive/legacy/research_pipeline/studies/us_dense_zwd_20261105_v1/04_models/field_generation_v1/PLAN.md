# GNSS field generation: approved stage-one execution

Final research objective: conditional residual EDM, with conditional GAN comparison,
for observation-consistent high-resolution ZWD fields. Stage one only: M0 -> M2-F
-> M2-J. Never implicitly launch diffusion, GAN, Optuna or additional seeds.

Data: all land 25--50 N / 125--110 W; 2016--2018 train, 2019 development,
2020 retrospective. New weights/statistics. Five input variables, coarse frames
t,t-6,t-12,t-18. Fixed input per 100 km LCC cell; physical groups isolated.
Candidates >=90% valid in each training year, inputs prefer >=95%. Non-input
roles approximately 70/15/15; half held-out cohorts dense 50 km groups >=4 sites.
Split seed 20261117; model seed42. EGM96 terrain and station height.

LCC WGS84 lon0=-117.5 lat0=37.5 lat1=30 lat2=45. 25 km grid, 600 km/24-cell
patches, 300 km stride, normalized separable sine-squared mosaic. Shared training,
publication and evaluation operator: interpolate Z0, then exp(-H/3000).
Support: nearest input <=100 km, >=3 within150 km, input hull, train-label distance
<=100 km and complete interpolation stencil. Target availability never an input.

M0 retains RAW_CURRENT query_film backbone, variable station count and patch terrain.
M2 station17->64->64, amplitude bypass; coarse60->32->32; U-Net32/64/128,
geometry attention12/6. Zero-initialized residual output, blended32 features.
Query uncertainty head35->32->1 (features,height,distance,count), positive surface
standard deviation. F: MSE + detached-mean/feature NLL; J: ordinary NLL.
Separate mean/variance optimizers and clipping1. Weak final-Z0 Laplacian penalty
1e-3, normalized by training Z0 std, supported complete five-point stencils.
No IDW teacher, no innovation inputs. Train-only residual normalization.

AdamW lr1e-4 wd1e-4,500 warmup/cosine1e-6,FP32,EMA.999; accumulated8 uniformly
sampled stations then valid times; complete contributor gradients, recompute if needed.
M0<=30000 updates, each M2<=20000. Startup evaluation200 excluded from selection;
screen every1000, stop after6 non-improvements. Shared month/hour576 maximum screen.
Top3 screen checkpoints reranked on complete2019, exact RMSE then CRPS ties.

Baselines: coarse upsample, input IDW(k4,p2,150km),M0,M0+innovation IDW,M2F,M2J.
Metrics: macro/micro RMSE,MAE,bias,coverage,CRPS,NLL,80/90/95 intervals;
pair differences0/25/50/100/200km; Z0 and surface pairs height difference<=100m;
4x4 reduction/upsampling comparison;2000 paired7-day block bootstrap and group audit.
Never call grid spacing effective resolution or station RMSE area-wide truth.

Required gates: leakage, units, height, coordinates, permutation/padding/all-missing,
mosaic unity/constants/all-contributor gradients, publication reload<=.001mm,
paired initialization, faithful mean-equivalence,100+resume100=200, fail nonfinite.

Monitor localhost only,5s polling,100-step logs, separated loss terms/RMSE/LR/speed/
memory/best delta and full-year table. Handoff only after M0 selected/frozen,
M2F actual updates and finite step200 evaluation, live visible logs/curves and
verified F->J queue pause/resume. Queue ends after J and registered evaluations.
