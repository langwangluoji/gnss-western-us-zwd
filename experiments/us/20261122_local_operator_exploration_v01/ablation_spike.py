"""B-only attribution probes after the initial exploration; no new C products."""
import time
import numpy as np
from operator_spike import ROOT,SOURCE,PROJECT,ridge,metric
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256


def main():
    immutable_json(ROOT/'ablation_plan.lock.json',dict(code=sha256(__file__),
        arms={'means_only':[0,4,8,12,16,20],'horizontal_direction':[0,1,2,8,9,10,16,17,18]},
        selection='same B inner ridge; no new C inference; descriptive post-screen attribution',server_jobs=False))
    started=time.monotonic();a=dict(np.load(SOURCE/'data.npz'));m=np.load(ROOT/'moments.npy')
    lock=read_json(SOURCE/'experiment.lock.json');scale=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')['grid_scale']
    x=np.concatenate([a['features'],a['innovation']/scale],-1);output={}
    for fold,split in enumerate(lock['splits']):
        output[str(fold)]={};oi=np.searchsorted(a['hours'],lock['outer_hours'])
        for name,cols in read_json(ROOT/'ablation_plan.lock.json')['arms'].items():
            r=ridge(a,np.concatenate([x,m[:,:,cols]/scale],-1),split,scale)
            output[str(fold)][name]=dict(inner=r['inner'],outer=metric(a,r['prediction'][oi],oi,np.asarray(split['outer'],bool)),ridge=r['ridge'])
    write_json(ROOT/'ablation_results.json',dict(folds=output,seconds=time.monotonic()-started))
    print('ABLATIONS_COMPLETE',{f:{n:[v['inner']['station_macro_rmse_mm'],v['outer']['station_macro_rmse_mm']] for n,v in row.items()} for f,row in output.items()},flush=True)


if __name__=='__main__':main()
