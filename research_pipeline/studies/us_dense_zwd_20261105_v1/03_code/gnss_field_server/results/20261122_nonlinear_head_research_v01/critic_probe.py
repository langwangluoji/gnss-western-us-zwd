"""Archived critic diagnostics, not a new GAN trial or parameter-gradient claim."""
import json
from pathlib import Path
import numpy as np
import torch
from local_innovation_pilot import Pilot,local_runtime
from gnss_field.data import PackedStore
from gnss_field.models.grid_generation import ObservationCritic
from gnss_field.objectives.scores import energy_score
from gnss_field.runtime import configure
from gnss_field.artifacts import write_json,sha256


def main():
    configure(type('R',(),{'cpus':2})(),42)
    root=Path(__file__).resolve().parent;project=root.parents[1]
    source=project/'results/20261122_generation_postrun_research_v01/downloaded'
    data=Pilot();store=PackedStore(local_runtime()['data_root'],purpose='context')
    grid=store.grid;xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    nodeXY=(2*(xy-xy.min(0))/np.ptp(xy,axis=0)-1)[data.a['global_nodes']].astype('f4')
    coordinates=torch.tensor(np.c_[data.a['xy']/400000,data.a['height']/3000],dtype=torch.float32)
    height=np.exp(-data.a['height']/3000)
    geometry=data.a['query_weights']/height[:,None]
    rng=np.random.default_rng(42);out={}
    original_folds=np.load(project/'results/20261120_local_context_three_arm_v01/dataset/arrays.npz')['folds']
    for fold in (0,1):
        reference=json.loads((source/f'fold_{fold}/gan/joint_weights_reference.json').read_text())
        path=source/reference['path'];state=torch.load(path,map_location='cpu',weights_only=False)
        critic=ObservationCritic().eval()
        critic.load_state_dict({k.removeprefix('critic.'):v for k,v in state['ema'].items() if k.startswith('critic.')})
        for p in critic.parameters():p.requires_grad_(False)
        archive=dict(np.load(source/f'fold_{fold}/gan/joint_outer_predictions.npz'))
        rows=[]
        for t,hour in enumerate(archive['hours']):
            if hour not in data.a['hours']:continue
            ix=int(data.indices([hour])[0]);features=np.c_[data.a['features'][ix],nodeXY]
            condition=torch.tensor((features[data.a['query_indices']]*geometry[:,:,None]).sum(1),dtype=torch.float32)
            real=archive['truth'][t];base=data.a['mean'][ix];members=archive['samples'][t]
            # Read the frozen mean directly from the original Gaussian archive,
            # since GAN archive mean includes its learned correction.
            gaussian=np.load(source/f'fold_{fold}/gaussian_outer.npz')
            gt=int(np.flatnonzero(gaussian['hours']==hour)[0]);base=gaussian['mean'][gt]
            if np.max(abs(base-data.a['mean'][ix]))>.001:raise ValueError('Frozen base join mismatch')
            shuffled=np.stack([rng.permutation(members[:,j]) for j in range(members.shape[1])],1)
            if np.max(abs(np.sort(shuffled,axis=0)-np.sort(members,axis=0)))>0:raise ValueError('Shuffling changed margins')
            for population,role in [('fitted',data.a['valid'][ix]&(original_folds!=fold)),('outer',archive['population'][t])]:
                mask=torch.tensor(role[None]);k=len(members)
                cond=condition[None].expand(k,-1,-1)
                masks=mask.expand(k,-1)
                with torch.no_grad():
                    real_score=float(critic(torch.tensor(np.where(role,(real-base)/10,0.)[None],dtype=torch.float32),coordinates,condition[None],mask)[0])
                    fake_score=float(critic(torch.tensor((members-base[None])/10,dtype=torch.float32),coordinates,cond,masks).mean())
                    shuffled_score=float(critic(torch.tensor((shuffled-base[None])/10,dtype=torch.float32),coordinates,cond,masks).mean())
                z=torch.tensor(members[:4][None],requires_grad=True)
                y=torch.tensor(np.where(role,real,0.)[None])
                es=energy_score(z,y,mask)
                adv=-.1*critic((z[0]-torch.tensor(base)[None])/10,coordinates,condition[None].expand(4,-1,-1),mask.expand(4,-1)).mean()
                g1=torch.autograd.grad(es,z,retain_graph=True)[0]
                g2=torch.autograd.grad(adv,z)[0]
                norm1=float(g1.norm());norm2=float(g2.norm())
                cosine=float((g1*g2).sum()/(g1.norm()*g2.norm()).clamp_min(1e-30))
                rows.append(dict(hour=int(hour),population=population,stations=int(role.sum()),D_real=real_score,D_fake=fake_score,
                    D_shuffled=shuffled_score,fake_minus_shuffled=fake_score-shuffled_score,
                    ES_gradient_norm=norm1,weighted_adversarial_gradient_norm=norm2,
                    weighted_adversarial_to_ES_gradient_ratio=norm2/norm1,gradient_cosine=cosine))
        out[str(fold)]=dict(weights_sha256=sha256(path),rows=rows,interpretation='Observation-value gradients, not generator-parameter gradients. Scores do not establish causality or a tuning rule.')
    write_json(root/'critic_diagnostics.json',out)
    for f,v in out.items():
        for population in ['fitted','outer']:
            r=[x for x in v['rows'] if x['population']==population]
            print(f,population,len(r),dict(shuffle_gap=np.mean([x['fake_minus_shuffled'] for x in r]),gradient_ratio=np.median([x['weighted_adversarial_to_ES_gradient_ratio'] for x in r]),cosine=np.mean([x['gradient_cosine'] for x in r])),flush=True)


if __name__=='__main__':main()
