"""Evaluate visible-station increment heads; future observations are labels only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def increment_scores(prediction,truth):
    valid=np.isfinite(prediction)&np.isfinite(truth)
    error=np.where(valid,prediction-truth,0).astype('float64')
    count=valid.sum(tuple(range(valid.ndim-2)))
    axes=tuple(range(error.ndim-2))
    def rmse(square):return np.sqrt(np.divide(square.sum(axes),count,out=np.full(count.shape,np.nan),where=count>0)).tolist()
    return dict(prediction_rmse_by_lead_variable=rmse(error**2),
        persistence_rmse_by_lead_variable=rmse(np.where(valid,truth,0).astype('float64')**2),
        valid_count_by_lead_variable=count.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['transformer72_aux','transformer72_physics','transformer72_physics_aux','gru72_physics_aux','lstm72_physics_aux'])
    args=p.parse_args();sys.path.insert(0,str(ENGINE))
    import torch
    from tools.terrain_diffusion.native_weather_experiment import CachedData
    from tools.terrain_diffusion.native_physics_lm import mean_model
    from tools.terrain_diffusion.data import sha,write_json
    torch.set_num_threads(3);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.3)
    cache=ENGINE/f'results/20261126_native_weather_cache_v2/calendar8_s{args.seed}'
    data=CachedData(cache)
    if not data.development:raise ValueError('diagnostic requires a registered development calendar')
    original=Path(data.protocol['config']['data']);hours=np.load(original/'hours.npy')
    raw=np.load(original/'values.npy',mmap_mode='r')
    outcomes=[]
    for name in args.models:
        folder=ENGINE/f'results/20261126_native_weather_means/{name}_s{args.seed}'
        if not (folder/'status.json').exists():continue
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'));config=protocol['config']
        if config['variant']=='baseline':raise ValueError('baseline has no node increment head')
        for filename,expected in protocol['extra_model_sources'].items():
            if sha(ENGINE/'tools/terrain_diffusion'/filename)!=expected:raise ValueError('frozen variant changed')
        if sha(ENGINE/'tools/terrain_diffusion/native_weather_models.py')!=protocol['model_code_sha256']:raise ValueError('frozen base changed')
        model=mean_model(config['kind'],config['length'],config['variant']).cuda().eval()
        model.load_state_dict(torch.load(folder/'last.pt',weights_only=True));predictions=[];truths=[]
        for i in range(len(data)):
            inputs,_,_,_=data.batch(i,config['length'])
            _,_,aux=model(**inputs,return_aux=True)
            prediction=aux[:,:,-1].cpu().numpy()*np.array([10.,2.,2.])
            origins=data.a['origins'][i];indices=np.searchsorted(hours,origins);context=data.a['context_ids'][i].astype(int)
            if not np.array_equal(hours[indices],origins):raise ValueError('source hour missing')
            now=raw[indices[:,None],context[None]]
            future=raw[(indices[:,None]+np.arange(1,7)[None])[:,:,None],context[None,None]].transpose(0,2,1,3)
            truth=future-now[:,:,None]
            predictions.append(prediction);truths.append(truth)
        predictions=np.stack(predictions);truths=np.stack(truths)
        outcome=dict(name=name,**increment_scores(predictions,truths),weights_sha256=sha(folder/'last.pt'),
            mean_protocol_sha256=sha(folder/'protocol.json'))
        outcomes.append(outcome);print(json.dumps(dict(name=name,**increment_scores(predictions,truths))),flush=True)
        del model
    if not outcomes:raise ValueError('no completed node-head model to audit')
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'station_forecast_audit.json',dict(seed=args.seed,outcomes=outcomes,
        variables=['ZWD_mm','T_K','P_hPa'],lead_hours=list(range(1,7)),
        evaluation_role='Visible training-pool stations, future observations used only as diagnostic labels. Not hidden-query scores.',
        future_condition_use=False,cache_manifest_sha256=sha(cache/'manifest.json'),code_sha256=sha(Path(__file__)),
        strict_oof=False,confirmation_C_D_read=False))


if __name__=='__main__':main()
