"""Frozen C-gate refinement: analytic Gaussian versus K32 direct diffusion."""
import json
import math
import numpy as np
import torch
from scipy.special import ndtr
from .data import Dataset,write_json,sha
from .transport import TransportModel
from .transport_train import ROOT
from .metrics import evaluate


def gaussian_crps(mean,scale,truth):
    z=(truth-mean)/scale
    return scale*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))


@torch.no_grad()
def run():
    torch.set_num_threads(4);data=Dataset();records=[]
    for seed in (42,17):
        result=dict(seed=seed,models={})
        for arm in ('gaussian','diffusion'):
            path=ROOT/f'B_{arm}_s{seed}'/'best.pt';pack=torch.load(path,map_location='cuda',weights_only=False);conf=pack['config']
            model=TransportModel(conf['variant'],data.mean[0],data.std[0]).cuda().eval();model.load_state_dict(pack['model'])
            gen=torch.Generator(device='cuda').manual_seed(19021);layouts=[];stored_samples=[];stored_truth=[];stored_scale=[]
            for li,c in enumerate(data.layouts()):
                ps=[];ss=[];ys=[]
                for row in (0,3,6,9):
                    for t in range(12,16):
                        d=data.batch(row,t,c,np.arange(137),'validation')
                        ys.append(np.where(d['mask'].cpu()[0],d['target'].cpu()[0]*data.std[0]+data.mean[0],np.nan))
                        if arm=='gaussian':
                            mean,scale=model.baseline(d);ps.append((mean*data.std[0]+data.mean[0]).cpu().numpy());ss.append((scale[0]*data.std[0]).cpu().numpy())
                        else:
                            d={k:v.expand(32,*v.shape[1:]) for k,v in d.items()}
                            p=model.sample(d,torch.randn(d['target'].shape,device='cuda',generator=gen))
                            ps.append((p*data.std[0]+data.mean[0]).cpu().numpy())
                samples=np.stack(ps,1);truth=np.stack(ys);score=evaluate(samples,truth)
                stored_samples.append(samples);stored_truth.append(truth)
                if arm=='gaussian':stored_scale.append(np.stack(ss))
                if arm=='gaussian':crps=float(np.nanmean(np.nanmean(gaussian_crps(samples[0],np.stack(ss),truth),axis=0)))
                else:crps=score['fair_crps_mm']
                layouts.append(dict(layout=li,macro_rmse_mm=score['station_macro_rmse_mm'],crps_mm=crps))
            arrays=dict(samples=np.stack(stored_samples),truth=np.stack(stored_truth))
            if stored_scale:arrays['scale']=np.stack(stored_scale)
            np.savez_compressed(ROOT/f'gate_refinement_{arm}_s{seed}.npz',**arrays)
            result['models'][arm]=dict(checkpoint_sha256=sha(path),layouts=layouts,
                macro_rmse_mm=float(np.mean([r['macro_rmse_mm'] for r in layouts])),crps_mm=float(np.mean([r['crps_mm'] for r in layouts])),
                estimator='analytic mean and Gaussian CRPS' if arm=='gaussian' else 'K32 mean and fair CRPS')
        variant=json.loads((ROOT/'A_gate.json').read_text())['selected_variant']
        baseline=json.loads((ROOT/f'A_{variant}_s{seed}'/'metrics.json').read_text())['macro_rmse_mm']
        result['point_margin_mm']=result['models']['diffusion']['macro_rmse_mm']-baseline
        result['crps_margin_vs_gaussian_mm']=result['models']['diffusion']['crps_mm']-result['models']['gaussian']['crps_mm']
        result['passes']=result['point_margin_mm']<=.2 and result['crps_margin_vs_gaussian_mm']<0
        records.append(result);print(json.dumps(result),flush=True)
    write_json(ROOT/'C_gate.json',dict(passed=all(r['passes'] for r in records),records=records,
        rule='Both seeds: diffusion RMSE <= paired deterministic+0.2mm and CRPS better than paired Gaussian',
        protocol='Frozen selected checkpoints, quarterly4h3layouts, independent noise19021, no target history',
        note='Conservative gate for further diffusion temporal training; does not rule out a separately designed temporal-only mechanism'))


if __name__=='__main__':run()
