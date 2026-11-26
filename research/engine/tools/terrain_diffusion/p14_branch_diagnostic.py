"""Quantify learned branch corrections on fixed development dates; no selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .p14_fusion import FusionData,P14Fusion
from .data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4);torch.set_grad_enabled(False)
    data=FusionData(a.data,72);results=[]
    for kind in ('features','mlp','gru','lstm','transformer'):
        path=a.out/f'{kind}_s{a.seed}/last.pt'
        if not path.exists():continue
        model=P14Fusion(kind,72).cuda().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        vectors={k:[] for k in ('feature','temporal','base_error','full_error')}
        for date in ('2020-01-08','2020-07-08'):
            start=int(np.searchsorted(data.events.hours,np.datetime64(date,'h').astype('int64')))
            for offset in (0,12):
                b=data.batch(np.arange(start+offset,start+offset+12),data.events.layouts[0],True,kind!='features',True)
                base=model.base(b['point'],b['support'],b['uv'])
                full=model(b)
                model.kind='features';spatial=model(b);model.kind=kind
                values={'feature':spatial-base,'temporal':full-spatial,
                        'base_error':b['target']-base,'full_error':b['target']-full}
                for name,v in values.items():vectors[name].append((v[b['mask']]*data.std[0]).cpu().numpy())
        v={k:np.concatenate(x) for k,x in vectors.items()}
        row=dict(kind=kind,seed=a.seed,feature_correction_rms_mm=float(np.sqrt(np.mean(v['feature']**2))),
            temporal_correction_rms_mm=float(np.sqrt(np.mean(v['temporal']**2))),
            trained_base_micro_rmse_mm=float(np.sqrt(np.mean(v['base_error']**2))),
            full_micro_rmse_mm=float(np.sqrt(np.mean(v['full_error']**2))))
        if kind!='features':
            row['temporal_vs_base_error_correlation']=float(np.corrcoef(v['temporal'],v['base_error'])[0,1])
        results.append(row);print(json.dumps(row),flush=True)
        del model
    write_json(a.out/f'branch_diagnostic_s{a.seed}.json',dict(status='POSTHOC_FIXED_DEVELOPMENT_DIAGNOSTIC',
        dates=['2020-01-08','2020-07-08'],hours_per_date=24,layout=0,
        interpretation='Trained base is from each joint model, not the frozen source; additive branch RMS is not net causal improvement',results=results))


if __name__=='__main__':main()
