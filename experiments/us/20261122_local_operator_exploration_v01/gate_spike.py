"""Exploratory weather-gated lattice regression; no server actions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import math
import numpy as np
import torch
from torch import nn
from operator_spike import ROOT,SOURCE,PROJECT,query,metric
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.contracts import ScienceConfig
from gnss_field.training.engine import TrainEngine,StageSpec,atomic_torch
from local_innovation_components import query_nodes,station_macro_mse
from gnss_field.runtime import configure


def scene_features(features):
    return np.stack([features[:,:,59],features[:,:,0]-features[:,:,12],features[:,:,56],features[:,:,57],
                     features[:,:,53],features[:,:,65],features[:,:,60],features[:,:,61]],-1)


class Gate(nn.Module):
    def __init__(self,center,std,amplitude):
        super().__init__()
        self.register_buffer('center',torch.tensor(center,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
        self.register_buffer('offset',torch.tensor(math.log(amplitude/(1.5-amplitude))))
        self.net=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,field):
        return 1.5*torch.sigmoid(self.offset+self.net((x-self.center)/self.std).squeeze(-1))*field


def main():
    configure(type('R',(),{'cpus':2})(),42)
    a=dict(np.load(SOURCE/'data.npz'));lock=read_json(SOURCE/'experiment.lock.json');selection=read_json(ROOT/'selection.lock.json')
    immutable_json(ROOT/'gate_training.lock.json',dict(code=sha256(__file__),plan=sha256(ROOT/'GATE_TRAINING_PLAN.md'),
        source_selection=sha256(ROOT/'selection.lock.json'),data=sha256(SOURCE/'data.npz'),max_updates=1008,seed=42,LR=.001,selection='raw inner; step0 included',server_jobs=False))
    scene=scene_features(a['features']);tx=torch.tensor(scene,device='cuda');qi=torch.tensor(a['query_indices'],device='cuda');qw=torch.tensor(a['query_weights'],device='cuda')
    base=torch.tensor(a['mean'],device='cuda');truth=torch.tensor(a['truth'],device='cuda');valid=torch.tensor(a['valid'],device='cuda')
    results={};preds={}
    for fold,split in enumerate(lock['splits']):
        directory=ROOT/f'gate_fold_{fold}';directory.mkdir(exist_ok=True)
        hours=np.asarray(split['train_hours']);ti=np.searchsorted(a['hours'],hours);inner=np.asarray(split['inner_hours']);ii=np.searchsorted(a['hours'],inner)
        center=scene[ti].mean((0,1));std=scene[ti].std((0,1));std[std<1e-4]=1.
        alpha=selection['coefficients'][str(fold)]['kernel'];kernel=np.load(ROOT/f"{alpha['tag']}.npy");tk=torch.tensor(kernel,device='cuda')
        torch.manual_seed(42);model=Gate(center,std,alpha['amplitude']).cuda()
        fit=torch.tensor(split['fit'],device='cuda');counts=(valid[ti]&fit[None]).sum(0)
        science=ScienceConfig(learning_rate=.001,refinement_lr=.001,minimum_lr=.001,warmup=0,ema=2**(-1/72),
            evaluate_every=72,startup_evaluation=200,minimum_refinement_steps=288,patience=4,log_every=24,gradient_clip=1.,weight_decay=1e-4)
        engine=TrainEngine(model,directory,hours,StageSpec('gate_refine',1008,1008,8,8,42,dict(lock=sha256(ROOT/'gate_training.lock.json'),fold=fold)),science,
            pause_paths=[ROOT/'PAUSE'],batch_telemetry=lambda h:dict(supervised_station_hours=int((valid[np.searchsorted(a['hours'],h)]&fit[None]).sum())))
        @torch.no_grad()
        def predict(current,ix):
            return torch.cat([base[j]+query_nodes(current(tx[j],tk[j]),qi,qw) for j in np.array_split(ix,max(1,math.ceil(len(ix)/16)))]).cpu().numpy()
        def objective(batch,current,step):
            ix=np.searchsorted(a['hours'],batch)
            p=base[ix]+query_nodes(current(tx[ix],tk[ix]),qi,qw)
            mse=station_macro_mse(p,truth[ix],valid[ix]&fit[None],counts,len(hours))
            return dict(loss=mse/100,MSE_mm2=mse)
        def evaluator(ema,step):
            r=metric(a,predict(engine.model,ii),ii,np.asarray(split['inner'],bool));e=metric(a,predict(ema,ii),ii,np.asarray(split['inner'],bool))
            return {'point':dict(value=r['station_macro_rmse_mm'],rmse_mm=r['station_macro_rmse_mm'],EMA_rmse_mm=e['station_macro_rmse_mm'])}
        initial=directory/'initial.pt'
        if not initial.exists():
            assert engine.step==0
            engine.products={'point':[dict(**evaluator(engine.ema,0)['point'],step=0,checkpoint=str(initial.resolve()))]}
            atomic_torch(initial,engine.state());engine.save()
        done=engine.run(objective,evaluator=evaluator)
        assert done['status']=='COMPLETE'
        best=engine.products['point'][0];model.load_state_dict(torch.load(best['checkpoint'],map_location='cpu',weights_only=False)['model'])
        preds[fold]=predict(model,np.arange(len(a['hours'])))
        atomic_torch(directory/'selected.pt',dict(model=model.cpu().state_dict(),selection=best,amplitude=alpha))
        results[str(fold)]=dict(selected=best,actual_steps=engine.step,inner=metric(a,preds[fold][ii],ii,np.asarray(split['inner'],bool)))
    immutable_json(ROOT/'gate_selection.lock.json',results)
    oi=np.searchsorted(a['hours'],lock['outer_hours'])
    for fold,split in enumerate(lock['splits']):
        results[str(fold)]['outer']=metric(a,preds[fold][oi],oi,np.asarray(split['outer'],bool))
        np.savez_compressed(ROOT/f'gate_fold_{fold}/prediction.npz',hours=a['hours'],prediction=preds[fold])
    write_json(ROOT/'gate_results.json',results)
    print('GATE_COMPLETE',{f:dict(step=r['selected']['step'],updates=r['actual_steps'],inner=r['inner']['station_macro_rmse_mm'],outer=r['outer']['station_macro_rmse_mm']) for f,r in results.items()},flush=True)


if __name__=='__main__':main()
