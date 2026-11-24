"""v13: fixed-grid U-Net foundations under the v12 allowed-feature contract."""
import argparse,copy,csv,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .data import Dataset,STUDY,PROJECT,static_geometry,sha,write_json
from .residual_experiment import load_pack,save_evaluation

ROOT=PROJECT/'results/20261124_unet_foundation_v13'
GEOMETRY=STUDY/'02_data/processed/gnss_field_server_v1/geometry.npz'

class Features:
    def __init__(self,data,static,geo):
        self.data=data;self.device='cuda';self.static=torch.as_tensor(static,device='cuda');self.geo=torch.as_tensor(geo,device='cuda')
        self.cg=torch.as_tensor(data.a['geo'],device='cuda');self.mean=torch.as_tensor(data.mean,device='cuda');self.std=torch.as_tensor(data.std,device='cuda')
        self.dist=torch.cdist(self.geo[:,:2],self.cg[:,:2]);self.weights=[torch.exp(-.5*(self.dist/r)**2)*(self.dist<=3*r) for r in (50,150,300)]
        phase=self.static[:,:2,None]*1000/torch.tensor([50,100,200,400],device='cuda')*2*torch.pi
        self.fourier=torch.cat([phase.sin(),phase.cos()],-1).reshape(len(static),16)
    @torch.no_grad()
    def __call__(self,raw,context,hour):
        c=torch.as_tensor(context,device='cuda');valid=torch.isfinite(raw);norm=torch.where(valid,(raw-self.mean)/self.std,0)
        frames=[];weather=[]
        for weights in self.weights:
            w=weights[:,c];den=w@valid[:,:,0].float().T;safe=den.clamp_min(1e-8)
            z=norm[:,:,0];avg=(w@z.T)/safe;sd=((w@(z*z).T)/safe-avg.square()).clamp_min(0).sqrt()
            # Height transport via common-height first/second moments, masked per observation.
            value=torch.where(valid[:,:,0],(z*self.std[0]+self.mean[0])*torch.exp(self.cg[c,2]/3),0)
            tm=(w@value.T)/safe;ts=((w@value.square().T)/safe-tm.square()).clamp_min(0).sqrt();factor=torch.exp(-self.geo[:,2:3]/3)
            ta=(tm*factor-self.mean[0])/self.std[0];ta=torch.where(den>0,ta,0);ts=ts*factor/self.std[0]
            vals=[avg,sd,den.log1p(),ta,ts,den.log1p()]
            current=[];counts=[]
            for j in (1,2):
                d=w@valid[:,:,j].float().T;a=(w@norm[:,:,j].T)/d.clamp_min(1e-8);vals.append(a);current.append(a[:,-1]);counts.append(d[:,-1].log1p())
            frames.append(torch.stack(vals,-1));weather.extend(current+counts)
        history=torch.cat(frames,-1)
        h=torch.as_tensor(hour,device='cuda',dtype=torch.float64)
        times=torch.stack([(h%24/24*2*torch.pi).sin(),(h%24/24*2*torch.pi).cos(),(h/(24*365.25)*2*torch.pi).sin(),(h/(24*365.25)*2*torch.pi).cos()]).float()
        active=valid[-1,:,0];dist=self.dist[:,c];near=dist.masked_fill(~active[None],float('inf')).min(1).values.clamp_max(600)/100
        support=((dist<150)&active[None]).sum(1).float().log1p()
        static=torch.cat([self.static,self.fourier,times.expand(len(self.geo),4),torch.stack(weather,-1),near[:,None],support[:,None]],-1)
        flat=torch.cat([static,history[:,-1],history.mean(1),history[:,-1]-history[:,0]],-1)
        if not torch.isfinite(flat).all():raise FloatingPointError('nonfinite features')
        return flat

class DoubleConv(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x):return self.net(x)

class GridUNet(nn.Module):
    def __init__(self,width):
        super().__init__();self.proj=nn.Conv2d(113,width,1);self.a=DoubleConv(width,width);self.b=DoubleConv(width,width*2);self.c=DoubleConv(width*2,width*4)
        self.ub=DoubleConv(width*6,width*2);self.ua=DoubleConv(width*3,width);self.head=nn.Conv2d(width,1,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        a=self.a(self.proj(x));b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));b=self.ub(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));a=self.ua(torch.cat([F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return x[:,52:53]+self.head(a)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(113,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return x[:,52]+self.net(x).squeeze(-1)

class Experiment:
    def __init__(self,data_path=None):
        self.data=Dataset(path=data_path,device='cpu') if data_path is not None else Dataset(device='cpu');a=self.data.a;g=np.load(GEOMETRY);self.shape=g['height'].shape
        static,geo=static_geometry(pd.DataFrame(dict(x=g['x'].ravel(),y=g['y'].ravel(),height=g['height'].ravel())))
        self.grid=Features(self.data,static,geo);self.train_features=Features(self.data,a['static'],a['geo']);self.val_features=Features(self.data,a['val_static'],a['val_geo'])
        self.raw={k:torch.as_tensor(a[k],device='cuda') for k in ('train','val_context')};self.valtruth=torch.as_tensor(a['val_y'],device='cuda')
        self.origin=torch.as_tensor(g['origin']/1000+12.5,device='cuda',dtype=torch.float32);self.gridheight=self.grid.geo[:,2].reshape(1,1,*self.shape)
        self.events=self.make_events();self.fit=[e for e in self.events if e['row']%6!=5];self.monitor=[e for e in self.events if e['row']%6==5 and e['hour'] in (0,6,12,18)]
    def make_events(self):
        rng=np.random.default_rng(13024);a=self.data.a;events=[]
        for row in range(len(a['train'])):
            for hour in range(24):
                for layout in (0,1):
                    if layout:
                        center=int(rng.integers(410));d=np.linalg.norm(a['geo'][:,:2]-a['geo'][center,:2],axis=1);pool=np.flatnonzero(d>150)
                    else:pool=np.arange(410)
                    c=np.sort(rng.choice(pool,82,False));target=~np.isin(a['groups'],a['groups'][c])
                    events.append(dict(row=row,hour=hour,layout=layout,context=c.tolist(),target=target.tolist()))
        return events
    def get_features(self,e,kind,validation=False):
        row=e['row'];t=12+e['hour'];c=e['context'];raw=self.raw['val_context' if validation else 'train'][row,t-11:t+1,c];hours=self.data.a['val_hours' if validation else 'train_hours'][row,t]
        feat=self.grid if kind!='mlp' else self.val_features if validation else self.train_features
        return feat(raw,c,hours)
    def query(self,field,geo):
        h,w=self.shape;uv=(geo[:,:2]-self.origin)/25;uv=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1).reshape(1,1,-1,2)
        common=(field*self.data.std[0]+self.data.mean[0])*torch.exp(self.gridheight/3)
        sampled=F.grid_sample(common,uv,mode='bilinear',padding_mode='border',align_corners=True).flatten()*torch.exp(-geo[:,2]/3)
        return (sampled-self.data.mean[0])/self.data.std[0]
    def predict(self,model,e,kind,validation=False):
        f=self.get_features(e,kind,validation)
        if kind=='mlp':return model(f)
        field=model(f.T.reshape(1,113,*self.shape));return self.query(field,self.val_features.geo if validation else self.train_features.geo)
    @torch.no_grad()
    def monitor_score(self,model,kind):
        model.eval();square=0.;count=0
        for e in self.monitor:
            y=(self.raw['train'][e['row'],12+e['hour'],:,0]-self.data.mean[0])/self.data.std[0];mask=torch.as_tensor(e['target'],device='cuda')&torch.isfinite(y)
            p=self.predict(model,e,kind);square+=float((p[mask]-y[mask]).square().sum());count+=int(mask.sum())
        return (square/count)**.5*self.data.std[0]
    def fit_model(self,kind,width,seed,folder,budget,patience):
        folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);model=(ResidualMLP() if kind=='mlp' else GridUNet(width)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        rng=np.random.default_rng(seed);order=rng.permutation(len(self.fit));start=time.monotonic();last_eval=start;step=0;best=float('inf');bad=0;history=[];best_step=0
        while time.monotonic()-start<budget or step==0:
            if step and step%len(order)==0:order=rng.permutation(len(self.fit))
            e=self.fit[order[step%len(order)]]
            model.train();opt.zero_grad(set_to_none=True);p=self.predict(model,e,kind);y=(self.raw['train'][e['row'],12+e['hour'],:,0]-self.data.mean[0])/self.data.std[0];mask=torch.as_tensor(e['target'],device='cuda')&torch.isfinite(y)
            loss=(p[mask]-y[mask]).square().mean();loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),2.)
            if not torch.isfinite(loss) or not torch.isfinite(gn):raise FloatingPointError('training nonfinite')
            opt.step();step+=1
            if step==1 or time.monotonic()-last_eval>=60 or time.monotonic()-start>=budget:
                score=float(self.monitor_score(model,kind));last_eval=time.monotonic();row=dict(step=step,seconds=last_eval-start,train_batch_rmse_mm=float(loss.detach().sqrt()*self.data.std[0]),monitor_rmse_mm=score,gradient_norm=float(gn));history.append(row)
                if score<best:best=score;bad=0;best_step=step;torch.save(model.state_dict(),folder/'best.pt')
                else:bad+=1
                torch.save(model.state_dict(),folder/'last.pt');write_json(folder/'progress.json',row);print(json.dumps(dict(stage=str(folder.name),kind=kind,seed=seed,**row)),flush=True)
                if bad>=patience:break
        with (folder/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,history[0],lineterminator='\n');w.writeheader();w.writerows(history)
        write_json(folder/'fit.json',dict(kind=kind,width=width,seed=seed,parameters=sum(p.numel() for p in model.parameters()),steps=step,best_step=best_step,best_monitor_rmse_mm=best,seconds=time.monotonic()-start,budget_seconds=budget,early_stopped=bad>=patience,epochs=step/len(self.fit)))
        self.plot_history(folder,history);model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));return model.eval()
    def plot_history(self,folder,rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        for a,key,title in zip(ax,['train_batch_rmse_mm','monitor_rmse_mm'],['Training batch (varying events)','Fixed temporal monitor']):a.plot([r['step'] for r in rows],[r[key] for r in rows],marker='.');a.set(xlabel='Updates',ylabel='RMSE (mm)',title=title);a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/'learning_curve.png',dpi=140);plt.close(fig)
    @torch.no_grad()
    def evaluate(self,model,kind,folder):
        pred=[];example=None
        for li,c in enumerate(self.data.layouts()):
            for row in range(12):
                for hour in range(24):
                    e=dict(row=row,hour=hour,context=c.tolist());p=self.predict(model,e,kind,True);pred.append((p*self.data.std[0]+self.data.mean[0]).cpu().numpy())
                    if kind!='mlp' and li==0 and row==6 and hour==12:
                        f=self.get_features(e,kind,True);example=(model(f.T.reshape(1,113,*self.shape))*self.data.std[0]+self.data.mean[0]).cpu().numpy();np.save(folder/'example_field_mm.npy',example)
            print(json.dumps(dict(stage='evaluation',kind=kind,layout=li)),flush=True)
        samples=np.stack(pred)[None];pack=load_pack(PROJECT/'results/20261124_residual_field_v12/cache/validation.npz');save_evaluation(folder/'expanded',samples,pack,self.data)
        return json.loads((folder/'expanded/metrics.json').read_text())

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=float,default=600);parser.add_argument('--patience',type=int,default=5);parser.add_argument('--smoke',action='store_true');parser.add_argument('--data-path',type=Path,default=None);args=parser.parse_args()
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT;root.mkdir(parents=True,exist_ok=False);start=time.monotonic();exp=Experiment(args.data_path)
    if args.smoke:exp.monitor=exp.monitor[:2]
    write_json(root/'protocol.json',dict(inputs=['ZWD','T','P','existing terrain and coordinates','12h causal history'],features=113,data_sha256=exp.data.contract['data_sha256'],geometry_sha256=sha(GEOMETRY),code_sha256=sha(__file__),train_events=len(exp.fit),all_events=len(exp.events),monitor_events=len(exp.monitor),budget_per_seed=args.budget,patience=args.patience,seeds=[42,17],confirmation_C_D_read=False,smoke=args.smoke))
    np.savez_compressed(root/'events.npz',row=[e['row'] for e in exp.events],hour=[e['hour'] for e in exp.events],layout=[e['layout'] for e in exp.events],context=[e['context'] for e in exp.events],target=[e['target'] for e in exp.events]);write_json(root/'events_receipt.json',dict(sha256=sha(root/'events.npz'),events=len(exp.events),fit_events=len(exp.fit),monitor_events=len(exp.monitor)));results=[]
    for eid,kind,width in [('E1301_mlp','mlp',0),('E1302_unet16','unet',16),('E1303_unet24','unet',24)]:
        for seed in ([42] if args.smoke else [42,17]):
            folder=root/eid/f's{seed}';write_json(root/'status.json',dict(state='training',experiment=eid,seed=seed));model=exp.fit_model(kind,width,seed,folder,args.budget,args.patience)
            if args.smoke:
                e=dict(row=0,hour=0,context=exp.data.layouts()[0]);p=exp.predict(model,e,kind,True);assert torch.isfinite(p).all();metrics=dict(finite=True)
            else:metrics=exp.evaluate(model,kind,folder)
            write_json(folder/'completion.json',dict(status='completed',best_sha256=sha(folder/'best.pt'),metrics=metrics));results.append(dict(experiment=eid,seed=seed,**metrics));write_json(root/'summary.json',results);del model;torch.cuda.empty_cache()
    write_json(root/'status.json',dict(state='completed',seconds=time.monotonic()-start,runs=len(results)))

if __name__=='__main__':run()
