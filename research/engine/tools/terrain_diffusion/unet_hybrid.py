"""E1304: paired point features plus a U-Net spatial residual branch."""
import argparse,csv,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .unet_foundation import Experiment,GridUNet,ResidualMLP,ROOT
from .data import write_json,sha
from .residual_experiment import load_pack,save_evaluation

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__();self.point=ResidualMLP();self.spatial=GridUNet(16)
    def forward(self,point,grid,uv,height_grid,height_query):
        delta=self.spatial(grid)-grid[:,52:53]
        spatial=F.grid_sample(delta*torch.exp(height_grid/3),uv,mode='bilinear',padding_mode='border',align_corners=True).flatten()*torch.exp(-height_query/3)
        return self.point(point)+spatial

class HybridExperiment(Experiment):
    def predict(self,model,e,kind='hybrid',validation=False):
        point=self.get_features(e,'mlp',validation);grid=self.get_features(e,'unet',validation).T.reshape(1,113,*self.shape);geo=self.val_features.geo if validation else self.train_features.geo
        h,w=self.shape;uv=(geo[:,:2]-self.origin)/25;uv=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1).reshape(1,1,-1,2)
        return model(point,grid,uv,self.gridheight,geo[:,2])
    @torch.no_grad()
    def evaluate(self,model,folder):
        preds=[]
        for li,c in enumerate(self.data.layouts()):
            for row in range(12):
                for hour in range(24):
                    e=dict(row=row,hour=hour,context=c);preds.append((self.predict(model,e,validation=True)*self.data.std[0]+self.data.mean[0]).cpu().numpy())
                    if li==0 and row==6 and hour==12:
                        f=self.get_features(e,'unet',True);g=f.T.reshape(1,113,*self.shape);field=model.point(f).reshape(1,1,*self.shape)+model.spatial(g)-g[:,52:53];np.save(folder/'native_preview_field_mm.npy',(field*self.data.std[0]+self.data.mean[0]).cpu().numpy())
            print(json.dumps(dict(stage='hybrid_evaluation',layout=li)),flush=True)
        pack=load_pack(ROOT.parent/'20261124_residual_field_v12/cache/validation.npz');save_evaluation(folder/'expanded',np.stack(preds)[None],pack,self.data)
        return json.loads((folder/'expanded/metrics.json').read_text())

def fit(ex,seed,folder,budget):
    folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);model=Hybrid().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(seed);order=rng.permutation(len(ex.fit));start=time.monotonic();last=start;best=float('inf');best_step=0;step=0;bad=0;rows=[]
    while time.monotonic()-start<budget or step==0:
        # Corrected epoch boundary: permutation is renewed before event selection.
        if step and step%len(order)==0:order=rng.permutation(len(ex.fit))
        e=ex.fit[order[step%len(order)]];model.train();opt.zero_grad(set_to_none=True);pred=ex.predict(model,e);y=(ex.raw['train'][e['row'],12+e['hour'],:,0]-ex.data.mean[0])/ex.data.std[0];mask=torch.as_tensor(e['target'],device='cuda')&torch.isfinite(y);loss=(pred[mask]-y[mask]).square().mean();loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(loss) or not torch.isfinite(gn):raise FloatingPointError('nonfinite hybrid training')
        opt.step();step+=1
        if step==1 or time.monotonic()-last>=60 or time.monotonic()-start>=budget:
            score=float(ex.monitor_score(model,'hybrid'));last=time.monotonic();row=dict(step=step,seconds=last-start,train_batch_rmse_mm=float(loss.detach().sqrt()*ex.data.std[0]),monitor_rmse_mm=score,gradient_norm=float(gn));rows.append(row)
            if score<best:best=score;bad=0;best_step=step;torch.save(model.state_dict(),folder/'best.pt')
            else:bad+=1
            torch.save(model.state_dict(),folder/'last.pt');write_json(folder/'progress.json',row);print(json.dumps(dict(stage='hybrid',seed=seed,**row)),flush=True)
            if bad>=5:break
    with (folder/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    write_json(folder/'fit.json',dict(kind='hybrid',width=16,seed=seed,parameters=sum(p.numel() for p in model.parameters()),steps=step,best_step=best_step,best_monitor_rmse_mm=best,seconds=time.monotonic()-start,budget_seconds=budget,early_stopped=bad>=5,epochs=step/len(ex.fit)))
    ex.plot_history(folder,rows);model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));return model.eval()

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=float,default=600);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not args.smoke:assert json.loads((ROOT/'status.json').read_text())['state']=='completed','Original queue must finish before E1304'
    root=ROOT/'E1304_point_unet' if not args.smoke else ROOT/'hybrid_smoke';root.mkdir(exist_ok=False);ex=HybridExperiment();start=time.monotonic()
    if args.smoke:ex.monitor=ex.monitor[:2]
    write_json(root/'protocol.json',dict(adaptive_followup=True,reason='Preserve exact point features while adding a spatial U-Net branch; motivated by first pure-grid result',source_sha256=sha(__file__),foundation_source_sha256=sha(__file__.replace('unet_hybrid.py','unet_foundation.py')),same_events_sha256=sha(ROOT/'events.npz'),budget_seconds=args.budget,smoke=args.smoke,field_readout='Native point-feature decoder plus bilinearly queried spatial correction; a coarse preview grid is not its exact arbitrary-point query operator',epoch_boundary='Corrected one-event-per-epoch ordering issue; earlier arms retain their recorded executed source'))
    for seed in ([42] if args.smoke else [42,17]):
        if not args.smoke:write_json(ROOT/'overall_status.json',dict(state='training_E1304',seed=seed,primary_runs_completed=6))
        folder=root/f's{seed}';write_json(root/'status.json',dict(state='training',seed=seed));model=fit(ex,seed,folder,args.budget)
        if args.smoke:
            e=dict(row=0,hour=0,context=ex.data.layouts()[0]);pred=ex.predict(model,e,validation=True);assert torch.isfinite(pred).all();metrics=dict(finite=True)
        else:metrics=ex.evaluate(model,folder)
        write_json(folder/'completion.json',dict(status='completed',best_sha256=sha(folder/'best.pt'),metrics=metrics));del model;torch.cuda.empty_cache()
    write_json(root/'status.json',dict(state='completed',seconds=time.monotonic()-start))
    if not args.smoke:write_json(ROOT/'overall_status.json',dict(state='completed',training_runs=8,primary_status='status.json',extension_status='E1304_point_unet/status.json'))

if __name__=='__main__':run()
