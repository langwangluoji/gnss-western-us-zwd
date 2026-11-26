"""Real disabled-path replay and both active four-update transport arms."""
import torch
from campaign import RESULTS,read,run
from tools.terrain_diffusion.data import sha,write_json
from increment_smoke import equal_state


def named_optimizer_state(state,names):
    ids=[i for g in state['optimizer']['param_groups'] for i in g['params']]
    if len(ids)!=len(names):raise ValueError('optimizer name order differs')
    return {n:state['optimizer']['state'].get(i,{}) for n,i in zip(names,ids)}


def main():
    from tools.terrain_diffusion.native_physics_lm import NativePhysicsLM
    from tools.terrain_diffusion.native_mean_fusion_v2 import NativeMeanFusionV2
    from tools.terrain_diffusion.native_directed_lag import NativeDirectedLag
    cache=RESULTS/'20261126_native_weather_cache_v2/train_s42';source=RESULTS/'20261126_native_weather_means/transformer72_physics_aux_s42'
    root=RESULTS/'20261126_directed_lag_smoke'
    for name,mode,disabled in [('disabled','joint',True),('factorized','factorized',False),('joint','joint',False)]:
        out=root/f'{name}_s42'
        run('native_directed_lag_experiment',['--phase','train','--cache',cache,'--source',source,'--out',out,
            '--seed',42,'--mode',mode,'--steps',4,'--smoke-cache',*(['--disabled-transport'] if disabled else [])],out/'status.json')
        if read(out/'status.json')['status']!='ENGINEERING_SMOKE_COMPLETE':raise ValueError('smoke incomplete')
    legacy_path=RESULTS/'20261126_increment_smoke/H_delta_s42/checkpoint_4.pt'
    legacy=torch.load(legacy_path,map_location='cpu',weights_only=True)
    new=torch.load(root/'disabled_s42/checkpoint_4.pt',map_location='cpu',weights_only=True)
    for name in ['model','ema']:
        for key,value in legacy[name].items():equal_state(value,new[name][key],name+'.'+key)
    for name in ['scheduler','torch_rng','cuda_rng','step']:equal_state(legacy[name],new[name],name)
    old_names=[n for n,_ in NativePhysicsLM().named_parameters()]
    def ordered(model):return old_names+[n for n,_ in model.named_parameters() if n in model.new_parameter_names()]
    before=named_optimizer_state(legacy,ordered(NativeMeanFusionV2('H')))
    after=named_optimizer_state(new,ordered(NativeDirectedLag('joint')))
    for name,state in before.items():equal_state(state,after[name],'optimizer.'+name)
    active={n:read(root/f'{n}_s42/curve.json') for n in ['factorized','joint']}
    if any(c[-1]['transport_key_gradient_norm']<=0 or c[-1]['transport_readout_norm']<=0 for c in active.values()):raise ValueError('inactive new transport')
    receipt=dict(complete=True,common_H_delta_state_exact=True,common_optimizer_by_name_exact=True,source_rng_exact=True,
        legacy_checkpoint_sha256=sha(legacy_path),disabled_checkpoint_sha256=sha(root/'disabled_s42/checkpoint_4.pt'),
        active_curves=active,engineering_only=True,strict_oof=False,confirmation_C_D_read=False)
    write_json(root/'equivalence.json',receipt);print('Directed smoke exact replay and active gradients verified',flush=True)


if __name__=='__main__':main()
