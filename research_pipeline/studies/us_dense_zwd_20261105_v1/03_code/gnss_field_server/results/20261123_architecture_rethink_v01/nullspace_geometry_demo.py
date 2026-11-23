"""Exploratory identifiability demonstration; geometry only, no labels or training."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parent.parent
config=json.loads((PROJECT/'results/20261122_full_generative_design_v01/config.local.json').read_text())
root=Path('\\\\?\\'+config['data']['packed_root'])
stations=pd.read_csv(root/'stations.csv')
stations['role']=stations.role.map({'input':'A','train':'B','development':'C','retrospective':'D'})
with np.load(root/'geometry.npz',allow_pickle=False) as g:
    shape=g['height'].shape; origin=g['origin']; land=g['region'].astype(bool)
    grid_xy=np.c_[g['x'].ravel(),g['y'].ravel()]
h,w=shape
q=(stations[['x','y']].to_numpy()-origin)/25000-.5
inside=np.isfinite(q).all(1)&(q>=0).all(1)&(q<=[w-1,h-1]).all(1)
safe=np.nan_to_num(q).clip([0,0],[w-1,h-1])
x=np.floor(safe[:,0]).astype(int).clip(0,w-2);y=np.floor(safe[:,1]).astype(int).clip(0,h-2)
dx,dy=safe[:,0]-x,safe[:,1]-y
indices=np.stack([y*w+x,y*w+x+1,(y+1)*w+x,(y+1)*w+x+1],-1)
weights=np.stack([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy],-1).astype('f4').astype('f8')
fixed=inside&land.ravel()[indices].all(1)
scale=np.exp(-stations.height.to_numpy()/3000.)
# Smooth isotropic RBF functions on fixed50km centres. This is a mathematical
# hypothesis class, not an empirical claim about atmospheric admissibility.
centres=grid_xy.reshape(h,w,2)[::2,::2].reshape(-1,2)
phi=np.exp(-.5*cdist(grid_xy,centres,'sqeuclidean')/50000.**2)
station_phi=(phi[indices]*weights[:,:,None]).sum(1)*scale[:,None]
fit=np.flatnonzero(fixed&stations.role.isin(['A','B']).to_numpy())
target=np.flatnonzero(fixed&(stations.role=='C').to_numpy())
A=station_phi[fit]
u,s,vt=np.linalg.svd(A,full_matrices=False)
rank=int((s>s[0]*1e-11).sum())
rng=np.random.default_rng(20261123)
a=rng.normal(size=len(centres)); a-=vt[:rank].T@(vt[:rank]@a)
field=phi@a
normalizer=5/np.sqrt(np.mean(field[land.ravel()]**2))
a*=normalizer; field*=normalizer
station_delta=station_phi@a
report=dict(scope='constructive geometry-only nullspace example; not overfitting causality or model skill',
    labels_loaded=False,training_performed=False,seed=20261123,grid_shape=list(shape),grid_cells=h*w,
    land_cells=int(land.sum()),rbf_centres=len(centres),rbf_length_scale_km=50,
    fixed_A_constraints=int((stations.iloc[fit].role=='A').sum()),fixed_B_constraints=int((stations.iloc[fit].role=='B').sum()),
    numerical_rank=rank,coefficient_nullity=len(centres)-rank,
    smallest_retained_singular_value=float(s[rank-1]),rank_threshold_relative=1e-11,
    common_height_land_perturbation_rms_mm=float(np.sqrt(np.mean(field[land.ravel()]**2))),
    maximum_A_B_query_change_mm=float(np.abs(station_delta[fit]).max()),
    C_geometry_count=len(target),C_query_change_rms_mm=float(np.sqrt(np.mean(station_delta[target]**2))),
    maximum_C_query_change_mm=float(np.abs(station_delta[target]).max()),
    interpretation='A and B queries cannot distinguish f and f+delta; A-only coarse conditions stay equal; C geometry used without labels',
    limitations=['field is a smooth constructed perturbation, not a learned posterior',
    'no claim that all meteorological priors allow this perturbation','all geometrically valid A/B constraints enforced; no current-label support filtering',
    'random subsets of these fixed measurements cannot add information in their common nullspace'])
(OUT/'nullspace_geometry_demo.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
np.savez_compressed(OUT/'nullspace_geometry_demo.npz',delta_Z0=field.reshape(shape),land=land,station_delta=station_delta,fit_ids=fit,C_ids=target)
print(json.dumps(report,ensure_ascii=False))
