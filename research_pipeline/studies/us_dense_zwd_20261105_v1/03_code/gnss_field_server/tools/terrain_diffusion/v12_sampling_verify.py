import json
import numpy as np
from .data import Dataset,sha,write_json
from .residual_experiment import ROOT,load_pack
from .v12_ensemble_scores import crps_fields

def run():
    root=ROOT/'E1208_frozen_sampling'
    completion=json.loads((root/'completion.json').read_text())
    assert completion['status']=='completed'
    data=Dataset(device='cpu');pack=load_pack(ROOT/'cache/validation.npz')
    truth=data.a['val_y'][pack['row'],12+pack['hour']]
    records=[]
    for seed in (42,17):
        for name in ('diffusion_k64','diffusion_nested_k16','post_four_k16','post_first_k16'):
            folder=root/f's{seed}'/name;p=load_pack(folder/'evaluation.npz')
            np.testing.assert_allclose(p['truth'],truth,atol=3e-5,rtol=1e-6,equal_nan=True)
            assert np.isfinite(p['samples']).all()
            empirical,fair=crps_fields(p['samples'],p['truth'])
            rmse=[];ec=[];fc=[]
            for li in range(3):
                ix=p['layout']==li
                rmse.append(np.nanmean(np.sqrt(np.nanmean((p['samples'].mean(0)[ix]-p['truth'][ix])**2,axis=0))))
                ec.append(np.nanmean(np.nanmean(empirical[ix],axis=0)));fc.append(np.nanmean(np.nanmean(fair[ix],axis=0)))
            m=json.loads((folder/'metrics.json').read_text())
            np.testing.assert_allclose([np.mean(rmse),np.mean(fc)],[m['macro_rmse_mm'],m['crps_mm']],atol=2e-5)
            records.append(dict(seed=seed,method=name,macro_rmse_mm=float(np.mean(rmse)),empirical_crps_mm=float(np.mean(ec)),fair_crps_mm=float(np.mean(fc))))
    receipts=json.loads((ROOT/'checkpoint_receipts.json').read_text())
    for item in receipts['weights']:assert sha(ROOT/item['path'])==item['sha256']
    for r in completion['receipts']:
        for key,path in [('gaussian_sha256','E1205_correlated_gaussian/s{}/best.pt'),('diffusion_sha256','E1206_diffusion/s{}/best.pt'),('post_tree_sha256','E1207_diffusion_xgb/s{}/tree.ubj')]:
            assert sha(ROOT/path.format(r['seed']))==r[key]
    write_json(root/'verification.json',dict(status='passed',canonical_truth_checked=True,checkpoint_hashes_checked=True,evaluations=records))
    print(json.dumps(records,indent=2))

if __name__=='__main__':run()
