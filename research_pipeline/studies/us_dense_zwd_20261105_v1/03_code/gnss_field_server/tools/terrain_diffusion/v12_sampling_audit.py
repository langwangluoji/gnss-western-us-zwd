"""E1208: frozen K64 sampling check; post calibration retains16-member features."""
import json,time
import numpy as np
import torch
import xgboost as xgb
from .data import Dataset,write_json,sha
from .residual_field import GaussianResidual,ResidualDenoiser
from .residual_experiment import ROOT,load_pack,downstream_features,probability_predict,save_evaluation
from .transport_gate import gaussian_crps


def run():
    torch.set_num_threads(4);start=time.monotonic();root=ROOT/'E1208_frozen_sampling';root.mkdir(parents=True,exist_ok=False)
    data=Dataset(device='cpu');val=load_pack(ROOT/'cache/validation.npz');chosen=json.loads((ROOT/'foundation_selection.json').read_text())['selected']['experiment'];receipts=[]
    for seed in (42,17):
        base=np.load(ROOT/chosen/f's{seed}/validation_base.npy');features=downstream_features(val,base);sd=root/f's{seed}';sd.mkdir()
        gp=ROOT/f'E1205_correlated_gaussian/s{seed}';model=GaussianResidual().cuda();model.load_state_dict(torch.load(gp/'best.pt',weights_only=True));model.eval();scale=float(np.load(gp/'normalization.npz')['scale']);means=[];sigmas=[]
        with torch.no_grad():
            for i in range(0,len(base),16):
                mu,sigma=model(torch.as_tensor(features[i:i+16],device='cuda'));means.append((base[i:i+16]+scale*mu.cpu().numpy())*data.std[0]+data.mean[0]);sigmas.append(scale*sigma.cpu().numpy()*data.std[0])
        mean=np.concatenate(means);sigma=np.concatenate(sigmas);truth=np.where(val['mask'],val['y']*data.std[0]+data.mean[0],np.nan);scores=[]
        for li in range(3):
            keep=val['layout']==li;err=mean[keep]-truth[keep]
            scores.append(dict(macro_rmse_mm=float(np.nanmean(np.sqrt(np.nanmean(err**2,axis=0)))),crps_mm=float(np.nanmean(np.nanmean(gaussian_crps(mean[keep],sigma[keep],truth[keep]),axis=0)))))
        write_json(sd/'gaussian_analytic.json',dict(macro_rmse_mm=float(np.mean([v['macro_rmse_mm'] for v in scores])),crps_mm=float(np.mean([v['crps_mm'] for v in scores])),layouts=scores))
        dp=ROOT/f'E1206_diffusion/s{seed}';model=ResidualDenoiser().cuda();model.load_state_dict(torch.load(dp/'best.pt',weights_only=True));model.eval();scale=float(np.load(dp/'normalization.npz')['scale'])
        samples=probability_predict(model,'diffusion',features,val['geo'],base,scale,data,64,12471);save_evaluation(sd/'diffusion_k64',samples,val,data)
        tree=xgb.Booster();tp=ROOT/f'E1207_diffusion_xgb/s{seed}/tree.ubj';tree.load_model(tp);parts=[]
        for start_member in range(0,64,16):
            group=samples[start_member:start_member+16];mu=(group.mean(0)-data.mean[0])/data.std[0];sigma=group.std(0)/data.std[0]
            x=np.concatenate([features,mu[...,None],sigma[...,None]],-1);correction=tree.predict(xgb.DMatrix(x.reshape(-1,x.shape[-1]))).reshape(base.shape)*data.std[0]
            parts.append(group+correction[None])
        save_evaluation(sd/'post_four_k16',np.concatenate(parts),val,data)
        receipts.append(dict(seed=seed,gaussian_sha256=sha(gp/'best.pt'),diffusion_sha256=sha(dp/'best.pt'),post_tree_sha256=sha(tp)))
        print(json.dumps(dict(stage='K64_audit_completed',seed=seed)),flush=True)
    write_json(root/'completion.json',dict(status='completed',seconds=time.monotonic()-start,receipts=receipts,
        protocol='Frozen checkpoints, independent noise12471,12month-start days24h3layouts; diffusion64; post applied independently to four16-member groups; no reselection or fitting',
        caution='Post group-aggregated ensemble members are coupled within groups; fair CRPS is descriptive here, not asserted unbiased for independent output draws'))


if __name__=='__main__':run()
