"""Run a source-locked joint-v2 / G265 / local-diffusion fusion comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
LEGACY=ROOT/'maintenance/weather_native_20261126'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.native_directed_lag_v2_experiment import code_hashes as mean_codes
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import codes as probability_codes
from tools.terrain_diffusion.native_model_seal import validate_model_seal,seal_completed_model,evaluation_marker
from fusion_contract import validate_embedding_identity,require_statistics,require_equal_receipt,require_recipe,trajectory_clock

RESULTS=file_path(ENGINE/'results')
OUTPUT=RESULTS/'20261126_best_base_diffusion_fusion'
HERE=Path(__file__).resolve().parent
LOCK=HERE/'lock.json'
STOP=HERE/'STOP'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def paths(seed,day=0):
    old=RESULTS/'20261126_directed_lag_v2'
    cache=RESULTS/'20261126_native_weather_cache_v2'/(f'train_s{seed}' if not day else f'calendar{day}_s{seed}')
    emb=(OUTPUT/f'embedding_train_s{seed}' if not day else old/f'joint_last_dev_s{seed}' if day==8
         else old/f'supplement/joint_last_s{seed}/embedding')
    return cache,emb,old/f'joint_s{seed}'


def codes():
    values={**mean_codes(),**probability_codes()}
    for name in ('native_copula.py','native_weather_evaluation.py','native_paths.py'):
        values[name]=sha(ENGINE/'tools/terrain_diffusion'/name)
    for p in (Path(__file__),HERE/'fusion_contract.py',HERE/'summary.py',LEGACY/'probability_joint_audit.py',
              LEGACY/'event_audit.py',LEGACY/'event_audit_lock.json',LEGACY/'mean_hs_summary.py',
              LEGACY/'weather_controls_summary.py',LEGACY/'inducing_calibration_audit.py'):
        values[p.relative_to(ROOT).as_posix()]=sha(p)
    return values


def inputs(seed):
    train_cache,train,mean=paths(seed)
    seal=validate_model_seal(mean,'COMPLETE_DIRECTED_LAG_V2',mean_codes())
    protocol=read(mean/'protocol.json')
    if protocol['config']['seed']!=seed or protocol['config']['mode']!='joint':raise ValueError('wrong frozen joint mean')
    tm=read(train/'manifest.json');stats=read(train/'statistics.json');require_statistics(stats)
    result=dict(mean_weights_sha256=seal['weights_sha256'],mean_protocol_sha256=seal['protocol_sha256'],
                mean_seal_sha256=sha(mean/'model_artifact_seal.json'),calendars={})
    for day in (0,8,15):
        cache,emb,_=paths(seed,day);manifest=read(emb/'manifest.json')
        validate_embedding_identity(manifest,tm,sha(cache/'manifest.json'),seal['weights_sha256'],seal['protocol_sha256'])
        if manifest['code_sha256']!=mean_codes():raise ValueError('joint export code changed')
        cp=read(cache/'protocol.json')
        if cp['config']['seed']!=seed or cp['config']['calendar_day']!=day:raise ValueError('wrong cache role or seed')
        for name,expected in manifest['files'].items():
            if sha(emb/name)!=expected:raise ValueError('joint embedding array changed')
        result['calendars'][str(day)]=dict(cache_manifest_sha256=sha(cache/'manifest.json'),
            embedding_manifest_sha256=sha(emb/'manifest.json'),statistics_sha256=sha(emb/'statistics.json'))
    result['normalization']=stats
    return result


def freeze():
    if LOCK.exists():raise FileExistsError('fusion lock already sealed')
    data=dict(code_sha256=codes(),sources={str(s):inputs(s) for s in (42,17)},
        neural_steps=3000,primary='local minus affine F6 fairCRPS; each calendar separately; common months; two fixed seeds',
        sample=dict(members=32,seed=261126,steps=20,solver='heun0'),
        paired=dict(members=16,mechanism='disjoint independent pairs centered at frozen joint'),
        calendar_days=[8,15],strict_oof=False,confirmation_C_D_read=False,
        spec_sha256=sha(ROOT/'docs/superpowers/specs/2026-11-26-best-base-diffusion-fusion.md'))
    write_json(LOCK,data);print('sealed',sha(LOCK),flush=True)


def validate_lock(seed=None):
    lock=read(LOCK)
    if lock['code_sha256']!=codes():raise ValueError('fusion frozen code changed')
    require_recipe(lock,dict(spec_sha256=sha(ROOT/'docs/superpowers/specs/2026-11-26-best-base-diffusion-fusion.md')))
    if seed is not None and lock['sources'][str(seed)]!=inputs(seed):raise ValueError('fusion source changed')
    return lock


def call(module,arguments,label,helper=False):
    if STOP.exists():raise SystemExit('fusion stopped by user flag')
    cmd=[sys.executable,str(LEGACY/module)] if helper else [sys.executable,'-m','tools.terrain_diffusion.'+module]
    cmd+=list(map(str,arguments));env=os.environ.copy();env['PYTHONPATH']=str(ENGINE)
    env['GNSS_RESEARCH_STOP_FILE']=str(STOP);env['PYTHONIOENCODING']='utf-8'
    print('starting',label,flush=True)
    with (HERE/f'{label}.log').open('a',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT if helper else ENGINE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def verify_training(folder,seed,kind):
    status=read(folder/'status.json');p=read(folder/'protocol.json')
    expected={'gaussian':'COMPLETE_PROBABILITY','gp':'COMPLETE_WEATHER_GP','affine':'COMPLETE_WEATHER_TRANSPORT','local':'COMPLETE_WEATHER_TRANSPORT'}[kind]
    if status['status']!=expected or status['steps']!=3000 or p['config']['seed']!=seed:raise ValueError('fusion training incomplete')
    cache,emb,_=paths(seed)
    if p['embedding_manifest_sha256']!=sha(emb/'manifest.json'):raise ValueError('fusion training uses wrong embedding')
    if kind=='gaussian':
        if p['normalization']!=read(emb/'statistics.json') or p['config']['condition_view']!='skip':raise ValueError('wrong Gaussian normalization/condition')
    else:
        from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
        seal_completed_model(folder,expected,gp_codes() if kind=='gp' else probability_codes())
    state=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
    weights=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    if state['step']!=3000 or state['model'].keys()!=weights.keys() or any(not torch.equal(v,state['model'][k]) for k,v in weights.items()):
        raise ValueError('fusion last weights differ from final3000')
    updates=status['supervised_updates'];clock=trajectory_clock(np.load(cache/'truth.npy',mmap_mode='r'))
    final_curve=read(folder/'curve.json')[-1]
    supervised_cells=status.get('fully_observed_trajectories',final_curve.get('total_trajectories'))
    if updates!=clock['supervised_updates'] or supervised_cells!=clock['complete_trajectories']:
        raise ValueError('actual training label support differs from recorded supervised clock')
    opt=state['optimizer'];ids={i for g in opt['param_groups'] for i in g['params']}
    if not opt['state'] or not set(opt['state'])<=ids or any(int(v['step'])!=updates for v in opt['state'].values()) or state['scheduler']['last_epoch']!=updates:
        raise ValueError('fusion optimizer clock differs')
    receipt=dict(seed=seed,kind=kind,lock_sha256=sha(LOCK),protocol_sha256=sha(folder/'protocol.json'),
        weights_sha256=sha(folder/'last.pt'),checkpoint_sha256=sha(folder/'checkpoint_3000.pt'),
        status_sha256=sha(folder/'status.json'),supervised_updates=updates,
        optimizer_states=len(opt['state']),optimizer_parameters=len(ids),actual_label_clock=clock)
    target=folder/'fusion_receipt.json'
    if target.exists():require_equal_receipt(read(target),receipt)
    else:write_json(target,receipt)
    return receipt


def train(seed):
    validate_lock(seed);cache,emb,_=paths(seed);root=OUTPUT/f's{seed}';root.mkdir(parents=True,exist_ok=True)
    gaussian=root/'S175';gp=root/'G265'
    common=['--cache',cache,'--embeddings',emb,'--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed]
    gauss_args=['--phase','probability','--cache',cache,'--embeddings',emb,'--domain','gaussian','--condition-view','skip','--width',64,'--seed',seed,'--lr',3e-4]
    if seed==42 and not (root/'smoke_S175/status.json').exists():call('native_weather_experiment',[*gauss_args,'--out',root/'smoke_S175','--steps',4],f's{seed}_smoke_S175')
    if not (gaussian/'status.json').exists():call('native_weather_experiment',[*gauss_args,'--out',gaussian,'--steps',3000],f's{seed}_S175')
    verify_training(gaussian,seed,'gaussian')
    if not (gaussian/'copula.json').exists():call('native_copula',['--cache',cache,'--embeddings',emb,'--gaussian',gaussian],f's{seed}_copula')
    if seed==42 and not (root/'smoke_G265/status.json').exists():call('native_weather_gp_experiment',[*common,'--phase','train','--out',root/'smoke_G265','--view',265,'--steps',4,'--smoke-cache'],f's{seed}_smoke_G265')
    if not (gp/'status.json').exists():call('native_weather_gp_experiment',[*common,'--phase','train','--out',gp,'--view',265,'--steps',3000],f's{seed}_G265')
    verify_training(gp,seed,'gp')
    for mode in ('affine','local'):
        if seed==42 and not (root/f'smoke_{mode}/status.json').exists():
            call('native_weather_gp_diffusion_experiment',[*common,'--gp',gp,'--phase','train','--mode',mode,'--out',root/f'smoke_{mode}','--steps',4,'--smoke-cache'],f's{seed}_smoke_{mode}')
        target=root/mode
        if not (target/'status.json').exists():call('native_weather_gp_diffusion_experiment',[*common,'--gp',gp,'--phase','train','--mode',mode,'--out',target,'--steps',3000],f's{seed}_{mode}')
        verify_training(target,seed,mode)
    print('all four formal models completed',seed,flush=True)


def evaluate(seed):
    validate_lock(seed);root=OUTPUT/f's{seed}';gaussian=root/'S175';gp=root/'G265'
    for mode in ('gaussian','gp','affine','local'):verify_training(root/({'gaussian':'S175','gp':'G265'}.get(mode,mode)),seed,mode)
    for day in (8,15):
        cache,emb,_=paths(seed,day)
        common=['--cache',cache,'--embeddings',emb,'--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed,'--members',32,'--sample-seed',261126]
        for mode in ('G265','affine','local'):
            folder=OUTPUT/f'calendar{day}/{mode}_s{seed}';model=gp if mode=='G265' else root/mode
            from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
            seal=validate_model_seal(model,'COMPLETE_WEATHER_GP' if mode=='G265' else 'COMPLETE_WEATHER_TRANSPORT',gp_codes() if mode=='G265' else probability_codes())
            if not folder.exists():
                if mode=='G265':call('native_weather_gp_experiment',[*common,'--phase','evaluate','--model',gp,'--out',folder],f's{seed}_day{day}_G265')
                else:call('native_weather_gp_diffusion_experiment',[*common,'--gp',gp,'--phase','evaluate','--process',model,'--out',folder,'--sample-steps',20,'--solver','heun0'],f's{seed}_day{day}_{mode}')
            evaluation_marker(folder,seal);lineage=read(folder/'lineage.json')
            require_equal_receipt(lineage,dict(cache_manifest_sha256=sha(cache/'manifest.json'),embedding_manifest_sha256=sha(emb/'manifest.json'),
                seed=261126,members=32,independent_iid_member_noise=True))
            gaussian_p=read(gaussian/'protocol.json')
            expected_source=dict(gaussian_protocol_sha256=sha(gaussian/'protocol.json'),gaussian_weights_sha256=sha(gaussian/'last.pt'),
                copula_sha256=sha(gaussian/'copula.json'),mean_weights_sha256=read(emb/'manifest.json')['mean_weights_sha256'],normalization=gaussian_p['normalization'])
            if mode!='G265':
                gp_seal=read(gp/'model_artifact_seal.json')
                expected_source.update(weather_gp_protocol_sha256=gp_seal['protocol_sha256'],weather_gp_weights_sha256=gp_seal['weights_sha256'],weather_gp_seal_sha256=sha(gp/'model_artifact_seal.json'))
                require_recipe(lineage,dict(solver='heun0',sampling_steps=20))
            require_equal_receipt(lineage['source'],expected_source)
            if not (folder/'joint_metrics.json').exists():call('probability_joint_audit.py',['--cache',cache,'--evaluation',folder,'--out',folder],f's{seed}_day{day}_{mode}_joint',helper=True)
    print('two calendars evaluated',seed,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['lock','train','evaluate','all'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,17]);args=parser.parse_args();torch.set_num_threads(2)
    if args.phase=='lock':freeze();return
    if args.seed is None:parser.error('--seed required')
    if args.phase in ('train','all'):train(args.seed)
    if args.phase in ('evaluate','all'):evaluate(args.seed)


if __name__=='__main__':main()
