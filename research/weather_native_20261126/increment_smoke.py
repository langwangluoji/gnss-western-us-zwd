"""Actual four-update equivalence against the immutable H trainer."""
import json
import torch
from campaign import RESULTS,read,run
from tools.terrain_diffusion.data import write_json,sha


def equal_state(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('changed tensor '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('changed keys '+path)
        for key in a:equal_state(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('changed length '+path)
        for i,(x,y) in enumerate(zip(a,b)):equal_state(x,y,path+'.'+str(i))
    elif a!=b:raise AssertionError('changed value '+path)


def main():
    seed=42;root=RESULTS/'20261126_increment_smoke';cache=RESULTS/'20261126_native_weather_cache_v2/train_s42'
    source=RESULTS/'20261126_native_weather_means/transformer72_physics_aux_s42'
    for name,module,extra in [('H_legacy','native_mean_fusion_v2_experiment',['--arm','H']),
        ('H_zero','native_increment_experiment',['--increment-weight',0]),
        ('H_delta','native_increment_experiment',['--increment-weight',.5])]:
        folder=root/f'{name}_s{seed}'
        run(module,['--phase','train','--cache',cache,'--source',source,'--out',folder,
            '--seed',seed,'--steps',4,'--smoke-cache',*extra],folder/'status.json')
        if read(folder/'status.json')['status']!='ENGINEERING_SMOKE_COMPLETE':raise ValueError('smoke incomplete')
    legacy=torch.load(root/'H_legacy_s42/checkpoint_4.pt',map_location='cpu',weights_only=True)
    zero=torch.load(root/'H_zero_s42/checkpoint_4.pt',map_location='cpu',weights_only=True)
    equal_state(legacy,zero)
    delta=torch.load(root/'H_delta_s42/checkpoint_4.pt',map_location='cpu',weights_only=True)
    if all(torch.equal(v,delta['model'][k]) for k,v in zero['model'].items()):raise ValueError('new objective has no effect')
    protocols={name:read(root/f'{name}_s42/protocol.json') for name in ['H_legacy','H_zero','H_delta']}
    for field in ['cache_manifest_sha256','source_checkpoint_sha256','source_protocol_sha256','parameters','warm_start_max_difference']:
        if not all(protocols[n][field]==protocols['H_legacy'][field] for n in ['H_zero','H_delta']):raise ValueError('smoke contract differs '+field)
    receipt=dict(complete=True,zero_vs_legacy_all_checkpoint_state_exact=True,new_objective_changes_weights=True,
        matched_fields=['model','ema','optimizer','scheduler','torch_rng','cuda_rng','step'],
        source_checkpoint_sha256=protocols['H_legacy']['source_checkpoint_sha256'],
        cache_manifest_sha256=protocols['H_legacy']['cache_manifest_sha256'],
        checkpoint_sha256={n:sha(root/f'{n}_s42/checkpoint_4.pt') for n in protocols},
        code_sha256={n:protocols[n]['code_sha256'] for n in protocols},
        per_arm_curves={n:read(root/f'{n}_s42/curve.json') for n in protocols},engineering_only=True)
    write_json(root/'equivalence.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['per_arm_curves','code_sha256']}),flush=True)


if __name__=='__main__':main()
