"""Independent array-level recomputation; does not call the metric implementation."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import HERE,CACHE,SOURCE,ENGINE,sha,write_json


def run():
    a=dict(np.load(CACHE/'data.npz'));summary=json.loads((HERE/'comparison.json').read_text(encoding='utf-8'));meta=pd.read_csv(HERE/'stations.csv')
    assert np.array_equal(meta.station_phys.to_numpy(),a['stations'])
    assert np.array_equal(meta.role.to_numpy(),a['role'])
    assert len(meta)==411 and int((a['role']=='train').sum())==273
    # Every old physical group has exactly one scientific role.
    for g in np.unique(a['groups']):
        roles=np.unique(a['role'][(a['groups']==g)&(a['role']!='new2024')]);assert len(roles)<=1
    audit=json.loads((HERE/'data_audit.json').read_text(encoding='utf-8'))
    source=[dict(path=row['source'],sha256=sha(row['source'])) for row in audit['sources']]
    assert all(x['sha256']==y['sha256'] for x,y in zip(source,audit['sources']))
    assert sha(CACHE/'data.npz')==audit['pack_sha256']
    for seed in (42,17):
        for name in ('base_v2','joint_v2'):
            root=HERE/f'runs/{name}_s{seed}';lock=json.loads((root/'lock.json').read_text(encoding='utf-8'));complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
            assert lock['version']==2 and complete['steps']==3000
            assert complete['best_sha256']==sha(root/'best.pt')
            if name=='base_v2':
                assert lock['neural_years']==[2020,2021,2022,2023]
                candidates=[HERE/'train_base.py',root/'train_base_source.py']
                assert any(p.exists() and sha(p)==lock['code_sha256'] for p in candidates)
            else:
                assert sha(root/'native_train_source.py')==lock['native_code_sha256']
        old=np.load(CACHE/f'original/predictions_s{seed}.npz');keys=old['hours']*411+old['station']
        assert np.all(old['station']>=0) and len(keys)==len(np.unique(keys))
    arrays=np.load(CACHE/'comparison_arrays.npz');rawtarget=a['values'][np.searchsorted(a['hours'],arrays['hours']),:,0]
    np.testing.assert_array_equal(arrays['target'],rawtarget)
    checks={};group_train=np.unique(a['groups'][a['role']=='train'])
    masks=dict(confirm=a['role']=='confirm',dev=a['role']=='dev',new_unseen_site=(a['role']=='new2024')&~np.isin(a['groups'],group_train),all_spatial_holdout=(a['role']!='train')&~np.isin(a['groups'],group_train))
    for cohort,station in masks.items():
        valid=arrays['valid']&station[None];count=int(np.count_nonzero(valid));assert count==summary['cohorts'][cohort]['records']
        checks[cohort]={}
        for key,metric in [('original','original_ensemble'),('ours','ours_ensemble')]:
            residual=arrays[key][valid].astype('f8')-arrays['target'][valid].astype('f8')
            rmse=float(np.sqrt(np.dot(residual,residual)/len(residual)))
            expected=summary['cohorts'][cohort]['models'][metric]['micro_rmse_mm'];assert abs(rmse-expected)<1e-9
            checks[cohort][metric]=rmse
    # Source-array reconstruction of the ensemble matches delivered arrays.
    chosen=summary['selection'];pred=[]
    for seed in (42,17):
        raw=np.load(CACHE/f'native_v2_s{seed}/final_predictions.npz');v=raw[chosen][:,:,0] if chosen=='joint' else raw[chosen]
        dense=np.full(arrays['target'].shape,np.nan);dense[:,raw['query']]=v;pred.append(dense)
    np.testing.assert_array_equal((pred[0]+pred[1])/2,arrays['ours'])
    body=(HERE/'REPORT.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)',body):
        if not target.startswith('https://'):assert (HERE/target).exists(),target
    figures=json.loads((HERE/'figure_manifest.json').read_text(encoding='utf-8'))
    assert len(figures)==14
    for name,expected in figures.items():assert sha(HERE/'figures'/name)==expected
    architecture={name:sha(ENGINE/'tools/terrain_diffusion'/name) for name in ['fixed_support.py','p14_fusion.py','temporal_encoders.py','native_weather_models.py','native_physics_lm.py','native_mean_fusion_v2.py','native_directed_lag.py','native_directed_lag_v2.py']}
    write_json(HERE/'verification.json',dict(status='passed',source_hashes_unchanged=source,old_site_roles_disjoint=True,formal_version=2,models_3000_steps_and_weight_hashes_verified=True,independent_rmse_recalculation=checks,selected_ensemble_reconstructed=True,report_links_exist=True,architecture_source_sha256=architecture))
    print(json.dumps(dict(status='passed',metrics=checks),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
