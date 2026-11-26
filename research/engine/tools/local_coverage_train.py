"""Finite, resumable local monthly-coverage mean experiment; no Slurm actions."""
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from local_coverage_data import ROOT,PROJECT,prepare,CoverageData,progress,local_runtime
from local_innovation_components import query_nodes,station_macro_mse
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.contracts import ScienceConfig
from gnss_field.training.engine import TrainEngine,StageSpec,PauseRequested,atomic_torch
from gnss_field.training.observable import _same
from gnss_field.runtime import configure
from local_coverage_provenance import dependency_snapshot,lock_dependencies


class Corrector(nn.Module):
    def __init__(self,center,std,scale):
        super().__init__();self.scale=float(scale)
        self.register_buffer('center',torch.as_tensor(center).detach().clone())
        self.register_buffer('std',torch.as_tensor(std).detach().clone())
        self.input=nn.Linear(69,32);self.residual=nn.Linear(32,32);self.output=nn.Linear(32,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def hidden(self,x):
        h=torch.nn.functional.silu(self.input((x-self.center)/self.std))
        return h+.1*torch.nn.functional.silu(self.residual(h))

    def forward(self,x):return self.scale*self.output(self.hidden(x)).squeeze(-1)

    def set_readout(self,beta):
        with torch.no_grad():
            self.output.weight.copy_(torch.as_tensor(beta[:-1],device=self.output.weight.device)[None])
            self.output.bias.copy_(torch.as_tensor(beta[-1:],device=self.output.bias.device))


def raw_features(data):
    if 'x69' not in data.t:data.t['x69']=torch.cat([data.t['features'],data.t['innovation']/data.settings['grid_scale']],-1)
    return data.t['x69']


@torch.no_grad()
def predictions(data,model,hours):
    model.eval();rows=[]
    for ix in np.array_split(data.indices(hours),max(1,math.ceil(len(hours)/16))):
        if (ROOT/'PAUSE').exists():raise PauseRequested('Paused during field evaluation')
        delta=query_nodes(model(raw_features(data)[ix]),data.t['query_indices'],data.t['query_weights'])
        rows.append((data.t['mean'][ix]+delta).cpu().numpy())
    return np.concatenate(rows)


@torch.no_grad()
def fit_readouts(data,fold):
    directory=ROOT/f'fold_{fold}';directory.mkdir(exist_ok=True)
    statefile=directory/'readouts.pt'
    if statefile.exists():return torch.load(statefile,map_location='cpu',weights_only=False)
    split=data.splits[fold];train=data.indices(split['train_hours']);inner=data.indices(split['inner_hours'])
    x=raw_features(data);center=x[train].mean((0,1));std=x[train].std((0,1),correction=0);std[std<1e-4]=1.
    torch.manual_seed(42);model=Corrector(center,std,data.settings['grid_scale']).cuda()
    mask=data.a['valid'][train]&split['fit'][None];counts=mask.sum(0);use=counts>0
    weights=np.where(mask,1/np.maximum(counts,1)[None]/use.sum(),0.)
    residual=np.where(mask,(data.a['truth'][train]-data.a['mean'][train])/model.scale,0.)
    base_metric=data.metric(data.a['mean'][inner],split['inner_hours'],split['inner'])
    saved=dict(center=center.cpu(),std=std.cpu(),base_inner=base_metric,initial_hidden=model.cpu().state_dict())
    model.cuda()
    for name in ('linear','nonlinear'):
        rows=[]
        for ix in np.array_split(np.arange(len(x)),math.ceil(len(x)/16)):
            feature=(x[ix]-center)/std if name=='linear' else model.hidden(x[ix])
            aug=torch.cat([feature,torch.ones_like(feature[...,:1])],-1)
            rows.append(((aug[:,data.t['query_indices']]*data.t['query_weights'][None,:,:,None]).sum(2)).cpu().numpy().astype('f8'))
        design=np.concatenate(rows);xx=design[train][mask];ww=weights[mask];yy=residual[mask]
        gram=xx.T@(ww[:,None]*xx);rhs=xx.T@(ww*yy);candidates=[]
        for ridge in (1e-4,.01,1.):
            beta=np.linalg.solve(gram+ridge*np.eye(len(rhs)),rhs)
            if abs((gram+ridge*np.eye(len(rhs)))@beta-rhs).max()>1e-10:raise ValueError('Ridge stationary solve failed')
            prediction=data.a['mean'][inner]+model.scale*(design[inner]@beta)
            metric=data.metric(prediction,split['inner_hours'],split['inner'])
            candidates.append(dict(ridge=ridge,beta=beta,inner=metric))
        best=min(candidates,key=lambda r:r['inner']['station_macro_rmse_mm'])
        saved[name]=best
        saved[name+'_candidates']=[{k:v for k,v in c.items() if k!='beta'} for c in candidates]
        if name=='linear':
            beta=best['beta'] if best['inner']['station_macro_rmse_mm']<base_metric['station_macro_rmse_mm'] else np.zeros(len(rhs))
            np.savez_compressed(directory/'linear_predictions.npz',hours=data.a['hours'],prediction=data.a['mean']+model.scale*(design@beta),beta=beta,center=center.cpu(),std=std.cpu())
            saved['linear_selected_zero']=bool(not np.any(beta))
    atomic_torch(statefile,saved);return saved


def make_engine(data,fold,directory,engineering_budget=None):
    lock_dependencies(ROOT,dependency_snapshot(PROJECT,local_runtime()))
    readouts=fit_readouts(data,fold);torch.manual_seed(42)
    model=Corrector(readouts['center'],readouts['std'],data.settings['grid_scale']).cuda()
    model.load_state_dict(readouts['initial_hidden']);model.set_readout(readouts['nonlinear']['beta'])
    hours=data.splits[fold]['train_hours'];epoch=len(hours)//8
    if len(hours)%8:raise ValueError('Registered full-hour pool must contain complete batches')
    budget=engineering_budget or 14*epoch
    science=ScienceConfig(learning_rate=1e-4,refinement_lr=1e-4,minimum_lr=1e-4,warmup=0,
        ema=2**(-1/epoch),gradient_clip=1.,weight_decay=1e-4,evaluate_every=epoch,startup_evaluation=200,
        minimum_refinement_steps=4*epoch,patience=4,log_every=24,budgets={'coverage_refine':budget})
    deps=dict(data=sha256(ROOT/'data_complete.json'),runner=sha256(__file__),readouts=sha256(ROOT/f'fold_{fold}/readouts.pt'),
        provenance=sha256(ROOT/'dependencies.lock.json'),
        engine=sha256(PROJECT/'src/gnss_field/training/engine.py'),fold=fold,selection='raw weights only; EMA diagnostic',
        objective='station macro normalized real B MSE plus fixed ridge readout penalty')
    fit=torch.tensor(data.splits[fold]['fit'],device='cuda');ti=data.indices(hours)
    counts=(data.t['valid'][ti]&fit[None]).sum(0)
    engine=TrainEngine(model,directory,hours,StageSpec('coverage_refine',budget,budget,8,8,42,deps),science,
        pause_paths=[ROOT/'PAUSE'],batch_telemetry=lambda h:dict(supervised_station_hours=int((data.t['valid'][data.indices(h)]&fit[None]).sum())))
    def objective(batch,current,step):
        ix=data.indices(batch);field=current(raw_features(data)[ix])/current.scale
        prediction=data.t['mean'][ix]/current.scale+query_nodes(field,data.t['query_indices'],data.t['query_weights'])
        mse=station_macro_mse(prediction,data.t['truth'][ix]/current.scale,data.t['valid'][ix]&fit[None],counts,len(hours))
        penalty=readouts['nonlinear']['ridge']*(current.output.weight.square().sum()+current.output.bias.square().sum())
        return dict(loss=mse+penalty,MSE_mm2=mse*current.scale**2,readout_penalty=penalty)
    def evaluator(ema,step):
        h=data.splits[fold]['inner_hours'];role=data.splits[fold]['inner']
        # TrainEngine supplies its EMA by convention. This experiment explicitly
        # selects raw; preserve both values and load `model`, never `ema`, later.
        raw=data.metric(predictions(data,engine.model,h),h,role)
        moving=data.metric(predictions(data,ema,h),h,role)
        return {'point':dict(value=raw['station_macro_rmse_mm'],rmse_mm=raw['station_macro_rmse_mm'],
            mae_mm=raw['station_macro_mae_mm'],bias_mm=raw['station_macro_bias_mm'],EMA_rmse_mm=moving['station_macro_rmse_mm'],
            n_stations=raw['n_stations'],n_station_hours=raw['n_station_hours'],selection_weights='raw')}
    return engine,objective,evaluator


def engineering(data):
    path=ROOT/'engineering/verification.json'
    if path.exists():return read_json(path)
    a,obj,_=make_engine(data,0,ROOT/'engineering/continuous',200)
    result=a.run(obj)
    if result['step']!=200:raise PauseRequested('Engineering continuous run incomplete')
    expected=copy.deepcopy(a.state());del a
    b,obj,_=make_engine(data,0,ROOT/'engineering/split',200)
    if b.step<100:b.run(obj,stop_after=100)
    if b.step<100:raise PauseRequested('Engineering split checkpoint before100')
    midpoint=ROOT/'engineering/split_100.pt'
    if b.step==100 and not midpoint.exists():atomic_torch(midpoint,b.state())
    if not midpoint.exists():raise ValueError('Missing actual engineering step100 evidence')
    proof=torch.load(midpoint,map_location='cpu',weights_only=False)
    if proof['step']!=100 or proof['signature']!=b.signature:raise ValueError('Invalid engineering midpoint evidence')
    del b
    b,obj,_=make_engine(data,0,ROOT/'engineering/split',200);result=b.run(obj)
    if result['step']!=200:raise PauseRequested('Engineering restore incomplete')
    actual=b.state()
    for key in ('model','ema','optimizers','cursor','rng','step'):_same(expected[key],actual[key])
    write_json(path,dict(passed=True,continuous=200,resumed=[100,100],bitwise=True,scientific_updates=False))


def train_fold(data,fold):
    directory=ROOT/f'fold_{fold}/nonlinear';engine,obj,evaluator=make_engine(data,fold,directory)
    initial=directory/'initial.pt'
    if not initial.exists():
        if engine.step!=0:raise ValueError('Missing original ridge checkpoint')
        metric=evaluator(engine.ema,0)['point']
        engine.products={'point':[dict(**metric,step=0,checkpoint=str(initial.resolve()))]}
        atomic_torch(initial,engine.state());engine.save()
    progress('TRAINING',fold=fold,step=engine.step,budget=engine.spec.budget,epoch_hours=len(data.splits[fold]['train_hours']))
    result=engine.run(obj,evaluator=evaluator)
    if result['status']!='COMPLETE':raise PauseRequested(f'Training fold{fold}: {result["status"]}')
    if engine.step<engine.spec.budget and not (engine.step>=engine.science.minimum_refinement_steps and engine.stagnant>=engine.science.patience):
        raise ValueError('Premature completion without registered early-stop condition')
    best=engine.products['point'][0];readouts=fit_readouts(data,fold)
    zero=readouts['base_inner']['station_macro_rmse_mm']
    selected=dict(method='zero',step=0,value=zero,checkpoint=None) if zero<=best['value'] else dict(method='nonlinear',**best)
    immutable_json(directory/'selection.lock.json',dict(selected=selected,rule='raw inner station macro RMSE; zero fallback; no outer selection',
        actual_updates=engine.step,budget=engine.spec.budget,early_stop=engine.step<engine.spec.budget))
    if selected['method']=='zero':engine.model.set_readout(np.zeros(33))
    else:engine.model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)['model'])
    mean=predictions(data,engine.model,data.a['hours'])
    np.savez_compressed(ROOT/f'fold_{fold}/nonlinear_predictions.npz',hours=data.a['hours'],prediction=mean)
    atomic_torch(ROOT/f'fold_{fold}/selected_weights.pt',dict(model=engine.model.cpu().state_dict(),scale=engine.model.scale,selected=selected))
    return dict(**selected,actual_updates=engine.step,early_stop=engine.step<engine.spec.budget)


def finalize(data,selected):
    result={};outer=np.asarray(data.lock['outer_hours']);oi=data.indices(outer)
    for fold in (0,1):
        split=data.splits[fold];inner=split['inner_hours'];ii=data.indices(inner);rows={}
        arrays={'zero':data.a['mean'],'linear':np.load(ROOT/f'fold_{fold}/linear_predictions.npz')['prediction'],
            'nonlinear':np.load(ROOT/f'fold_{fold}/nonlinear_predictions.npz')['prediction']}
        for name,prediction in arrays.items():
            rows[name]=dict(inner=data.metric(prediction[ii],inner,split['inner']),outer=data.metric(prediction[oi],outer,split['outer']))
        winner=min(('zero','linear','nonlinear'),key=lambda name:rows[name]['inner']['station_macro_rmse_mm'])
        result[str(fold)]=dict(selected=winner,rows=rows,nonlinear_selection=selected[str(fold)],
            inner_gain=(rows['zero']['inner']['station_macro_rmse_mm']-rows[winner]['inner']['station_macro_rmse_mm'])/rows['zero']['inner']['station_macro_rmse_mm'])
    gate=all(r['inner_gain']>=.005 and r['rows'][r['selected']]['outer']['station_macro_rmse_mm']<r['rows']['zero']['outer']['station_macro_rmse_mm'] for r in result.values())
    nonlinear_gate=gate and all(r['selected']=='nonlinear' for r in result.values())
    write_json(ROOT/'results.json',dict(folds=result,mean_direction_gate=gate,nonlinear_direction_gate=nonlinear_gate,
        C2019_evaluated=False,D2020=False,server_jobs=False))
    progress('COMPLETE',mean_direction_gate=gate,nonlinear_direction_gate=nonlinear_gate,
        updates={k:v['actual_updates'] for k,v in selected.items()},server_jobs=False)


def main():
    if (ROOT/'status.json').exists() and read_json(ROOT/'status.json')['stage']=='COMPLETE':
        print('ALREADY_COMPLETE: no training or lock mutation; use local_coverage_report.py for evaluation-only reporting',flush=True)
        return
    configure(type('R',(),{'cpus':2})(),42)
    prepare();data=CoverageData()
    immutable_json(ROOT/'training.lock.json',dict(runner=sha256(__file__),plan=sha256(ROOT/'PLAN.md'),data=sha256(ROOT/'data_complete.json'),
        epochs_max=14,learning_rate=1e-4,effective_hours=8,seed=42,primary_weights='raw',EMA_half_life_hours=576,
        criterion='station macro RMSE on 12 monthly weather blocks x fixed held station groups',
        stage_order='readouts -> real engineering resume -> fold0 -> fold1 -> outer',server_jobs=False))
    engineering(data);selected={}
    for fold in (0,1):selected[str(fold)]=train_fold(data,fold)
    finalize(data,selected)


if __name__=='__main__':
    try:main()
    except PauseRequested as exc:progress('PAUSED',reason=str(exc))
