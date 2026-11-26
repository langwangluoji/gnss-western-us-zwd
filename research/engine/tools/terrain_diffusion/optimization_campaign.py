"""Bounded P1/P3 campaign for spatial support and height-scale hypotheses."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .fixed_support import FixedSupportModel, masked_multiscale_summary
from .unet_foundation import Experiment


class Campaign:
    def __init__(self, sampling: str, height_scale: str):
        self.ex = Experiment(); self.sampling = sampling; self.height_scale = height_scale
        self.events = self.make_events(); self.fit_events = [e for e in self.events if e['row'] % 6 != 5]; self.monitor = [e for e in self.events if e['row'] % 6 == 5 and e['hour'] in (0, 6, 12, 18)]

    def make_events(self):
        a=self.ex.data.a; rng=np.random.default_rng(20261124); events=[]
        for row in range(72):
            for hour in range(24):
                for layout in (0,1):
                    if self.sampling == 'random': pool=np.arange(410)
                    else:
                        center=int(rng.integers(410)); d=np.linalg.norm(a['geo'][:,:2]-a['geo'][center,:2],axis=1)
                        threshold=100 if self.sampling == 'hole100' else 150
                        pool=np.flatnonzero(d>threshold)
                    if self.sampling == 'distance_stratified':
                        center=int(rng.integers(410)); d=np.linalg.norm(a['geo'][:,:2]-a['geo'][center,:2],axis=1); bins=[np.flatnonzero(d<=50),np.flatnonzero((d>50)&(d<=100)),np.flatnonzero((d>100)&(d<=200)),np.flatnonzero(d>200)]; c=np.concatenate([rng.choice(b,min(len(b),20),False) for b in bins if len(b)]); c=np.unique(c)[:82]
                    else: c=np.sort(rng.choice(pool,82,False))
                    target=~np.isin(a['groups'],a['groups'][c]); events.append(dict(row=row,hour=hour,layout=layout,context=c.tolist(),target=target.tolist()))
        return events

    def support(self,e,validation=False):
        row,hour,c=e['row'],e['hour'],e['context']; t=12+hour; raw=self.ex.raw['val_context' if validation else 'train'][row,t-11:t+1,c]; cond=self.ex.train_features.geo; grid=self.ex.grid.geo
        scale=3.0
        summary=masked_multiscale_summary(grid[:,:2],grid[:,2],cond[c,:2],cond[c,2],raw[:,:,0],torch.isfinite(raw[:,:,0]),scale_km=scale)
        return summary.T.reshape(1,30,*self.ex.shape)

    def predict(self,model,e,validation=False):
        point=self.ex.get_features(e,'mlp',validation); geo=self.ex.val_features.geo if validation else self.ex.train_features.geo; h,w=self.ex.shape; uv=(geo[:,:2]-self.ex.origin)/25; uv=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1).reshape(1,1,-1,2); return model(point,self.support(e,validation),uv)

    def fit(self,seed,seconds,out):
        torch.manual_seed(seed); rng=np.random.default_rng(seed); model=FixedSupportModel().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); order=rng.permutation(len(self.fit_events)); start=time.monotonic(); step=0; best=1e9; history=[]; out.mkdir(parents=True,exist_ok=True)
        while time.monotonic()-start<seconds or step==0:
            e=self.fit_events[order[step%len(order)]]; model.train(); opt.zero_grad(set_to_none=True); p=self.predict(model,e); y=(self.ex.raw['train'][e['row'],12+e['hour'],:,0]-self.ex.data.mean[0])/self.ex.data.std[0]; m=torch.as_tensor(e['target'],device='cuda')&torch.isfinite(y); loss=(p[m]-y[m]).square().mean(); loss.backward(); gn=nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); step+=1
            if step==1 or step%50==0:
                model.eval(); scores=[]
                with torch.no_grad():
                    for q in self.monitor[:8]:
                        yy=(self.ex.raw['train'][q['row'],12+q['hour'],:,0]-self.ex.data.mean[0])/self.ex.data.std[0]; mm=torch.as_tensor(q['target'],device='cuda')&torch.isfinite(yy); scores.append(float((((self.predict(model,q)[mm]-yy[mm]).square().mean().sqrt())*self.ex.data.std[0]).cpu()))
                score=float(np.mean(scores)); history.append({'step':step,'monitor_rmse_mm':score,'train_rmse_mm':float((loss.sqrt()*self.ex.data.std[0]).detach())});
                if score<best: best=score; torch.save(model.state_dict(),out/'best.pt')
                print(json.dumps({'seed':seed,'step':step,'monitor_rmse_mm':score}),flush=True)
        model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.eval(); (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        result={'seed':seed,'steps':step,'best_monitor_rmse_mm':best}; (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--sampling',choices=['random','hole100','hole150','distance_stratified'],default='random'); p.add_argument('--height-scale',default='fixed3000'); p.add_argument('--seconds',type=float,default=120); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); c=Campaign(a.sampling,a.height_scale); a.out.mkdir(parents=True,exist_ok=False); (a.out/'protocol.json').write_text(json.dumps({'sampling':a.sampling,'height_scale':a.height_scale,'seconds':a.seconds,'seeds':[42,17],'confirmation_C_D_read':False},indent=2),encoding='utf-8'); rows=[c.fit(s,a.seconds,a.out/f's{s}') for s in (42,17)]; (a.out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__': main()
