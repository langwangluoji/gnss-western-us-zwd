"""Native FP32, full-epoch observation-supervised generative stages.

Only atomic serialization and RNG helpers are shared with the historical engine.
The published field excludes independent observation discrepancy; losses add it at
station queries exactly once. Latent pretraining never treats dense fields as truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import hashlib
import json
import math
import random
import signal
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from gnss_field.artifacts import fingerprint, model_digest, sha256, write_json
from gnss_field.models.diffusion import keyed_noise
from gnss_field.training.engine import atomic_torch, capture_rng, restore_rng
from .architecture import (ConditionalFieldVAE, LatentGenerator, LatentCritic,
                           LatentDenoiser, ObservationCritic)
from .observations import query_surface, query_features, variational_loss, sample_surface
from .evaluation import EvaluationPaused


STAGE_EPOCHS = dict(deterministic_control=10, vae_representation=30,
                    posterior_cache=1, gan_pretrain=20, diffusion_pretrain=20,
                    prior_es=3, complete_model_es=3)


@dataclass
class FullTrainingConfig:
    stage: str = 'vae_representation'
    model_family: str = 'vae'
    seed: int = 42
    epochs: int | None = None
    min_epochs: int | None = None
    batch_size: int = 4
    lr: float = 1e-4
    minimum_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_updates: int = 100
    kl_warmup_updates: int = 0
    patience: int = 6
    output_dir: str = 'full_generative_run'
    sigma_eff_mm: float = 6.36
    latent_dim: int = 64
    grid_shape: tuple[int, int] = (144, 96)
    data_config: dict = field(default_factory=dict)
    dependency_paths: dict = field(default_factory=dict)
    budget_seconds: float | None = None
    device: str = 'cpu'
    max_updates: int | None = None
    checkpoint_every_updates: int = 100
    evaluate_every_epochs: int = 1
    formal_selection_start_step: int = 1
    evaluation_check_updates: tuple[int, ...] = (200,)
    development_hours: list | None = None
    eval_members: int = 16
    point_members: int = 64
    evaluation_seed: int = 20261123
    point_seed: int = 20261124
    evaluation_chunk: int = 4
    evaluation_hours_batch: int = 1
    evaluation_population_manifest: str | None = None
    evaluation_callback: Callable | None = field(default=None, repr=False)
    ema_decay: float = .999
    reference_count: float | None = None
    es_members: int = 4
    sampling_steps: int = 18
    adversarial_weight: float = 1.
    es_weight: float = 1.
    gradient_clip: float = 10.
    late_decay_fraction: float = .7

    def __post_init__(self):
        if self.stage not in STAGE_EPOCHS or self.model_family not in ('vae','gan','diffusion'):
            raise ValueError('Unknown stage or model family')
        self.epochs = STAGE_EPOCHS[self.stage] if self.epochs is None else self.epochs
        if self.stage == 'posterior_cache':
            self.epochs = 1  # A cache enumerates its population once, rather than optimizing it.
        self.min_epochs = self.epochs if self.min_epochs is None else self.min_epochs
        if not 1 <= self.min_epochs <= self.epochs or self.batch_size < 1:
            raise ValueError('Invalid full epoch limits or batch size')
        if self.sigma_eff_mm <= 0 or self.es_members < 2 or self.sampling_steps < 2:
            raise ValueError('Invalid observation discrepancy or sampling configuration')
        if not 0 < self.minimum_lr <= self.lr or not 0 <= self.ema_decay < 1:
            raise ValueError('Invalid learning rate or EMA')
        if self.formal_selection_start_step < 1 or self.evaluate_every_epochs < 1 or any(step < 1 for step in self.evaluation_check_updates):
            raise ValueError('Formal selection and diagnostic evaluation updates must be positive')
        if self.kl_warmup_updates < 0:
            raise ValueError('KL warmup updates must be nonnegative')
        if not 0 < self.late_decay_fraction < 1 or self.checkpoint_every_updates < 1:
            raise ValueError('Invalid checkpoint or late-decay configuration')
        if self.stage == 'gan_pretrain' and self.model_family != 'gan':
            raise ValueError('GAN pretraining requires GAN family')
        if self.stage == 'diffusion_pretrain' and self.model_family != 'diffusion':
            raise ValueError('Diffusion pretraining requires diffusion family')
        if self.stage in ('deterministic_control','vae_representation','posterior_cache') and self.model_family != 'vae':
            raise ValueError('Representation stages require VAE family')


def _config_dict(config):
    return {k:v for k,v in asdict(config).items() if k != 'evaluation_callback'}


def _science_signature(config):
    omitted = {'output_dir','dependency_paths','budget_seconds','device','max_updates',
               'checkpoint_every_updates','evaluation_callback'}
    return fingerprint({k:v for k,v in _config_dict(config).items() if k not in omitted})


class FullGenerativeModel(nn.Module):
    """A complete conditional field distribution and its train-only critics."""
    def __init__(self, config):
        super().__init__()
        self.family = config.model_family
        self.deterministic = config.stage == 'deterministic_control'
        self.latent_dim = config.latent_dim
        self.vae = ConditionalFieldVAE(latent_dim=config.latent_dim,grid_shape=tuple(config.grid_shape))
        self.prior_model = (LatentGenerator(latent_dim=config.latent_dim) if self.family=='gan' else
                            LatentDenoiser(latent_dim=config.latent_dim) if self.family=='diffusion' else None)
        self.latent_critic = LatentCritic(latent_dim=config.latent_dim) if self.family=='gan' else None
        self.observation_critic = ObservationCritic() if self.family=='gan' else None
        self.register_buffer('latent_mean',torch.zeros(config.latent_dim))
        self.register_buffer('latent_std',torch.ones(config.latent_dim))

    def encode(self,batch):
        return self.vae.encode(batch)

    def sample(self,encoded,background,noise):
        if self.family == 'vae':
            return self.vae.sample(encoded,background,noise)
        if self.family != 'gan':
            raise ValueError('Use sample_fields for the diffusion sampling chain')
        z=self.prior_model(noise,encoded.global_features)*self.latent_std+self.latent_mean
        return self.vae.decoder(encoded,z,background)

    def sample_fields(self,batch,members,seed,steps=18):
        encoded=self.encode(batch)
        if self.deterministic:
            z=batch['background'].new_zeros(len(batch['hour_ids']),members,self.latent_dim)
            return self.vae.decoder(encoded,z,batch['background'])
        if self.family=='diffusion':
            z=self.prior_model.sample(encoded.global_features,batch['hour_ids'],members,seed,steps=steps)
            return self.vae.decoder(encoded,z*self.latent_std+self.latent_mean,batch['background'])
        noise=keyed_noise(batch['hour_ids'],members,seed,(self.latent_dim,),
                          device=batch['background'].device)
        return self.sample(encoded,batch['background'],noise)


def read_training_checkpoint(path):
    """Keep CPU RNG byte tensors on CPU even when the model trains on CUDA."""
    return torch.load(path,map_location='cpu',weights_only=False)


def _validate_best_candidates(state):
    # Standalone candidates are immutable inference artifacts. Their own file hash
    # is registered in the publishing last.pt, rather than recursively in themselves.
    if state.get('status') == 'candidate':
        return
    selections=list(state.get('best',{}).values())+list(state.get('baseline',{}).values())
    selections.extend(selection for group in state.get('candidates',{}).values() for selection in group)
    verified=set()
    for selection in selections:
        path=Path(selection['checkpoint'])
        identity=(str(path),selection.get('sha256'))
        if identity in verified:
            continue
        if not path.is_file() or selection.get('sha256')!=sha256(path):
            raise ValueError('Selected development candidate is missing or its SHA256 differs')
        verified.add(identity)



def _plain_metrics(value):
    """Retain complete evaluator outputs in portable CPU/JSON-compatible values."""
    if torch.is_tensor(value):
        value=value.detach().cpu()
        return value.item() if value.ndim==0 else value.tolist()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {str(key):_plain_metrics(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain_metrics(item) for item in value]
    if isinstance(value,Path):
        return str(value)
    return value


def _candidate_key(selection):
    return (selection['value'],selection['step'],0 if selection['weight_source']=='raw' else 1)


def _training_label_identity(data,hours,batch_size):
    provider=getattr(data,'_label_identity',None)
    if callable(provider):
        return provider('B')
    manifest=getattr(getattr(data,'store',None),'manifest',{})
    if manifest.get('training_hash'):
        return manifest['training_hash']
    # Small injected fixtures may expose only the public batch API. Hash their
    # actual B labels/masks; production FullFieldData uses its audited manifest.
    digests={key:hashlib.sha256(key.encode()) for key in ('hour_ids','target','target_mask')}
    for offset in range(0,len(hours),batch_size):
        batch=data.batch(hours[offset:offset+batch_size],role='B',device='cpu')
        for key in ('hour_ids','target','target_mask'):
            value=batch[key].detach().cpu().contiguous().numpy()
            digests[key].update(value.tobytes())
    return fingerprint({key:digest.hexdigest() for key,digest in digests.items()})


def load_stage_model(path,device='cpu',use_ema=None):
    state=read_training_checkpoint(path)
    _validate_best_candidates(state)
    config=FullTrainingConfig(**state['config'])
    model=FullGenerativeModel(config).to(device).float()
    source=state.get('selected_weight_source','raw') if use_ema is None else ('ema' if use_ema else 'raw')
    model.load_state_dict(state[source])
    return model,state


def energy_score(samples,target,mask,scale_mm=10.):
    """IID U-statistic energy score, equal hours, sqrt(valid N) normalization."""
    if samples.shape[1]<2 or (mask.sum(-1)==0).any():
        raise ValueError('ES requires >=2 IID members and at least one valid target')
    clean=torch.where(mask[:,None,:],samples,torch.zeros_like(samples))
    truth=torch.where(mask,target,torch.zeros_like(target))
    norm=mask.sum(-1).sqrt()*scale_mm
    first=torch.linalg.vector_norm(clean-truth[:,None,:],dim=-1).mean(-1)
    pair=torch.cdist(clean,clean).sum((-1,-2))/(samples.shape[1]*(samples.shape[1]-1))
    return ((first-.5*pair)/norm).mean()


def adversarial_gradient_scale(es, adversarial, parameters, weight):
    """Audit the two actual generator gradients; keep the locked weight fixed."""
    norms=[]
    for objective in (es,adversarial):
        gradients=torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True)
        squares=[g.detach().square().sum() for g in gradients if g is not None]
        norm=torch.stack(squares).sum().sqrt() if squares else objective.new_zeros(())
        if not torch.isfinite(norm) or norm<=1e-12:
            raise FloatingPointError('Nonfinite or zero ES/adversarial generator gradient')
        norms.append(norm)
    ratio=abs(float(weight))*norms[1]/norms[0]
    if ratio>10:
        raise FloatingPointError('Adversarial generator gradient dominates ES by more than10x')
    return dict(es_generator_gradient_norm=norms[0],adversarial_generator_gradient_norm=norms[1],
                adversarial_gradient_ratio=ratio)


def edm_dsm_loss(denoiser,posterior_draw,condition,sigma=None,noise=None):
    """EDM sigma_data=1 weighted DSM against a posterior *draw*."""
    if sigma is None:
        sigma=(torch.randn(len(posterior_draw),device=posterior_draw.device)*1.2-1.2).exp()
    if noise is None:
        noise=torch.randn_like(posterior_draw)
    estimate=denoiser.denoise(posterior_draw+sigma[:,None]*noise,sigma,condition)
    weight=(sigma.square()+1)/sigma.square()
    return (weight[:,None]*(estimate-posterior_draw).square()).mean()


def _surface(fields,batch):
    return query_surface(fields,batch['query_indices'],batch['query_weights'],batch['target_geometry'][:,2])


def _residual(batch):
    return torch.where(batch['target_mask'],batch['target']-_surface(batch['background'],batch)[:,0],0.)


def _vae_loss(model,batch,config,encoded=None,beta=1.):
    encoded=model.encode(batch) if encoded is None else encoded
    noise=torch.randn(len(batch['hour_ids']),1,config.latent_dim,device=batch['background'].device)
    fields,mq,lq,mp,lp=model.vae.posterior_sample(encoded,batch['background'],_residual(batch),
        batch['query_indices'],batch['query_weights'],batch['target_geometry'],batch['target_mask'],noise)
    # Analytic Gaussian likelihood integrates independent station discrepancy.
    result=variational_loss(_surface(fields,batch),batch['target'],batch['target_mask'],
                            mq,lq,mp,lp,config.sigma_eff_mm,beta=beta,reference_count=config.reference_count)
    reference=float(batch['target'].shape[-1] if config.reference_count is None else config.reference_count)
    return {**result,'beta':result['loss'].new_tensor(beta),'kl_sum':result['kl']*reference,
            'negative_elbo':result['nll']+result['kl']}


def _draw_fields(model,batch,config,encoded=None):
    encoded=model.encode(batch) if encoded is None else encoded
    if model.family=='diffusion':
        # Independent RNG-derived key; checkpointed global RNG makes resumed paths identical.
        seed=int(torch.randint(0,2**31-1,()).item())
        z=model.prior_model.sample(encoded.global_features,batch['hour_ids'],config.es_members,
                                  seed,steps=config.sampling_steps)
        return model.vae.decoder(encoded,z*model.latent_std+model.latent_mean,batch['background'])
    noise=torch.randn(len(batch['hour_ids']),config.es_members,config.latent_dim,device=batch['background'].device)
    return model.sample(encoded,batch['background'],noise)


def _observation_samples(fields,batch,config):
    noise=torch.randn(len(batch['hour_ids']),fields.shape[1],batch['target'].shape[1],device=fields.device)
    return sample_surface(fields,batch,noise,config.sigma_eff_mm)


def make_model_and_objective(config,batch=None):
    config=FullTrainingConfig(**config) if isinstance(config,dict) else config
    model=FullGenerativeModel(config).to(config.device).float()
    _load_dependencies(model,config)
    _set_trainable(model,config)

    def objective(current,values,*,step=0):
        if config.stage=='deterministic_control':
            encoded=current.encode(values)
            z=values['background'].new_zeros(len(values['hour_ids']),1,config.latent_dim)
            pred=_surface(current.vae.decoder(encoded,z,values['background']),values)[:,0]
            error=torch.where(values['target_mask'],pred-values['target'],0.)
            loss=(error.square().sum(-1)/values['target_mask'].sum(-1)).mean()/100
            return {'loss':loss,'mse_mm2':loss*100}
        if config.stage=='vae_representation':
            beta=min(1.,step/config.kl_warmup_updates) if config.kl_warmup_updates else 1.
            return _vae_loss(current,values,config,beta=beta)
        if config.stage in ('prior_es','complete_model_es'):
            encoded=current.encode(values)
            fields=_draw_fields(current,values,config,encoded)
            samples=_observation_samples(fields,values,config)
            es=energy_score(samples,values['target'],values['target_mask'])
            result={'loss':config.es_weight*es,'es':es}
            if config.stage=='complete_model_es' and current.family=='vae':
                elbo=_vae_loss(current,values,config,encoded)
                result={**result,**{k:v for k,v in elbo.items() if k!='loss'},
                        'loss':elbo['loss']+config.es_weight*es}
            if config.stage=='complete_model_es' and current.family=='gan':
                local=query_features(encoded.pyramid[0],values['query_indices'],values['query_weights']).detach()
                base=_surface(values['background'],values)[:,0]
                fake=(samples-base[:,None,:])/10
                scores=[current.observation_critic(fake[:,i],local,values['target_geometry'],values['target_mask'])
                        for i in range(config.es_members)]
                adv=-torch.stack(scores).mean()
                result['loss']=result['loss']+config.adversarial_weight*adv
                result['adversarial']=adv
            return result
        raise ValueError('Latent pretraining uses posterior-cache objectives')
    return model,objective


def _representation_hashes(model):
    return {key:model_digest(getattr(model.vae,key)) for key in ('encoder','decoder','posterior')}


def _load_dependencies(model,config):
    paths=config.dependency_paths
    if config.stage not in ('vae_representation','deterministic_control') and 'representation' not in paths:
        raise ValueError('A frozen representation dependency is required')
    if 'representation' in paths:
        source,_=load_stage_model(paths['representation'],config.device)
        model.vae.load_state_dict(source.vae.state_dict())
    if 'prior' in paths:
        source,_=load_stage_model(paths['prior'],config.device)
        if source.family!=model.family:
            raise ValueError('Prior model family mismatch')
        model.load_state_dict(source.state_dict())
    if 'posterior_cache' in paths and config.stage in ('gan_pretrain','diffusion_pretrain','prior_es'):
        cache=torch.load(paths['posterior_cache'],map_location='cpu',weights_only=False)
        if cache['representation_hashes']!=_representation_hashes(model):
            raise ValueError('Latent cache representation hashes differ')
        model.latent_mean.copy_(cache['standardization']['mean'].to(config.device))
        model.latent_std.copy_(cache['standardization']['std'].to(config.device))
    if config.stage in ('gan_pretrain','diffusion_pretrain') and 'posterior_cache' not in paths:
        raise ValueError('Latent pretraining requires a posterior cache')
    if config.stage in ('prior_es','complete_model_es') and model.family!='vae' and 'prior' not in paths:
        raise ValueError('ES refinement requires a pretrained prior')


def _set_trainable(model,config):
    model.requires_grad_(False)
    if config.stage in ('deterministic_control','vae_representation','complete_model_es'):
        model.vae.encoder.requires_grad_(True)
        model.vae.decoder.requires_grad_(True)
    if config.stage in ('vae_representation','prior_es','complete_model_es') and model.family=='vae':
        model.vae.prior.requires_grad_(True)
    if config.stage in ('vae_representation','complete_model_es') and model.family=='vae':
        model.vae.posterior.requires_grad_(True)
    if model.prior_model is not None and config.stage!='posterior_cache':
        model.prior_model.requires_grad_(True)


def _gp(critic,real,fake,condition,mask=None,geometry=None):
    alpha=torch.rand(len(real),1,device=real.device)
    mixed=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    score=(critic(mixed,condition) if mask is None else critic(mixed,condition,geometry,mask))
    gradient=torch.autograd.grad(score.sum(),mixed,create_graph=True)[0]
    if mask is not None:
        gradient=torch.where(mask,gradient,0.)
    return (gradient.norm(2,dim=-1)-1).square().mean()


def _learning_rate(config,step,total):
    if config.warmup_updates and step<config.warmup_updates:
        return config.lr*(step+1)/config.warmup_updates
    start=max(config.warmup_updates,total*config.late_decay_fraction)
    if step<start or total<=start:
        return config.lr
    fraction=min(1.,(step-start)/max(1.,total-start-1))
    return config.minimum_lr+(config.lr-config.minimum_lr)*.5*(1+math.cos(math.pi*fraction))


def _cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def run_stage(config,data=None,evaluate=None):
    """Run/recover one bounded local stage; never allocate or submit a server job."""
    started=time.monotonic()
    config=FullTrainingConfig(**config) if isinstance(config,dict) else config
    if data is None:
        from .data import FullFieldData
        data=FullFieldData(config.data_config,purpose='train')
    hours=np.asarray(data.train_hours,dtype=np.int64)
    if not len(hours) or len(np.unique(hours))!=len(hours):
        raise ValueError('Training hours must be nonempty and unique')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if config.device.startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    label_identity=_training_label_identity(data,hours,config.batch_size)
    data_identity=fingerprint(dict(metadata=getattr(data,'metadata',{}),train_hours=hours,
                                   b_label_identity=label_identity))
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'last.pt'
    dependencies={k:sha256(v) for k,v in config.dependency_paths.items() if k!='resume'}
    if config.evaluation_population_manifest is not None:
        dependencies['evaluation_population_manifest']=sha256(config.evaluation_population_manifest)
    for name in ('representation','prior'):
        if name in config.dependency_paths:
            source=read_training_checkpoint(config.dependency_paths[name])
            if source.get('data_identity')!=data_identity:
                raise ValueError('Dependency data contract differs from this training population')
    model,objective=make_model_and_objective(config)
    generator_parameters=[p for p in model.parameters() if p.requires_grad]
    optimizers={}
    if generator_parameters:
        optimizers['generator']=torch.optim.AdamW(generator_parameters,lr=config.lr,weight_decay=config.weight_decay)
    critic=None
    if config.stage=='gan_pretrain': critic=model.latent_critic
    if config.stage=='complete_model_es' and config.model_family=='gan': critic=model.observation_critic
    if critic is not None:
        optimizers['critic']=torch.optim.AdamW(critic.parameters(),lr=config.lr,betas=(0.,.9),weight_decay=0.)
    cursor_rng=np.random.default_rng(config.seed)
    order=cursor_rng.permutation(hours)
    position=epoch=step=0
    history=[];evaluation_history=[];best={};baseline={};candidates={'point':[],'joint':[]}
    bad_evaluations=0;pending=False
    telemetry=dict(elapsed_seconds=0.,data_seconds=0.,model_seconds=0.,checkpoint_seconds=0.,evaluation_seconds=0.)
    prior_elapsed=0.
    optimizer_steps={key:0 for key in optimizers}
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    cache_parts=[]
    cache=None
    if config.stage in ('gan_pretrain','diffusion_pretrain'):
        cache=torch.load(config.dependency_paths['posterior_cache'],map_location='cpu',weights_only=False)
        if cache.get('data_identity')!=data_identity:
            raise ValueError('Posterior cache data contract differs')
        if not np.array_equal(np.sort(cache['hour_ids'].numpy()),np.sort(hours)):
            raise ValueError('Posterior cache training-hour population differs')
        cache_index={int(h):i for i,h in enumerate(cache['hour_ids'])}
    resume=Path(config.dependency_paths.get('resume',checkpoint))
    if resume.exists():
        state=read_training_checkpoint(resume)
        _validate_best_candidates(state)
        if state.get('data_identity')!=data_identity:
            raise ValueError('Resume data contract changed')
        if state['science_signature']!=_science_signature(config) or state['dependency_hashes']!=dependencies:
            raise ValueError('Resume scientific configuration or dependency hashes changed')
        if not np.array_equal(np.sort(state['cursor']['order']),np.sort(hours)):
            raise ValueError('Resume training-hour manifest changed')
        model.load_state_dict(state['raw']);ema={k:v.to(config.device) for k,v in state['ema'].items()}
        for key,opt in optimizers.items(): opt.load_state_dict(state['optimizers'][key])
        cursor=state['cursor'];order=cursor['order'];position=cursor['position'];epoch=state['epochs']
        cursor_rng.bit_generator.state=cursor['rng']
        step=state['step'];history=state['history'];best=state['best'];bad_evaluations=state['bad_evaluations']
        evaluation_history=state.get('evaluation_history',[])
        baseline=state.get('baseline',{})
        candidates=state.get('candidates',{'point':[],'joint':[]})
        pending=state['pending_evaluation'];optimizer_steps=state['optimizer_steps'];cache_parts=state.get('cache_parts',[])
        restore_rng(state['rng'])
        telemetry=state.get('telemetry',telemetry)
        prior_elapsed=telemetry['elapsed_seconds']
    stop_requested=False
    previous_handlers={}

    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True

    for name in ('SIGTERM','SIGINT','SIGUSR1'):
        signum=getattr(signal,name,None)
        if signum is not None:
            try:
                previous_handlers[signum]=signal.signal(signum,request_stop)
            except ValueError: pass  # Python requires signal registration on the main thread.

    def state(status):
        return dict(format_version=3,config=_config_dict(config),science_signature=_science_signature(config),
            dependency_hashes=dependencies,data_identity=data_identity,b_label_identity=label_identity,raw=_cpu_state(model),
            ema={k:v.detach().cpu().clone() for k,v in ema.items()},
            optimizers={k:v.state_dict() for k,v in optimizers.items()},optimizer_steps=optimizer_steps,
            rng=capture_rng(),cursor=dict(order=order.copy(),position=position,rng=copy.deepcopy(cursor_rng.bit_generator.state)),
            step=step,epochs=epoch,pending_evaluation=pending,best=best,bad_evaluations=bad_evaluations,
            history=history,evaluation_history=evaluation_history,candidates=candidates,baseline=baseline,
            status=status,cache_parts=cache_parts,
            telemetry={**telemetry,'elapsed_seconds':prior_elapsed+time.monotonic()-started},
            processed_hours=epoch*len(hours)+position,
            latent_cache_active=config.stage in ('gan_pretrain','diffusion_pretrain'),
            representation_hashes=_representation_hashes(model),
            objective_name=('station_elbo_plus_es' if config.stage=='complete_model_es' and model.family=='vae' else
                            'real_station_es' if config.stage in ('prior_es','complete_model_es') else config.stage))

    def save(status):
        write_started=time.monotonic()
        atomic_torch(checkpoint,state(status))
        telemetry['checkpoint_seconds']+=time.monotonic()-write_started
        progress=dict(status=status,stage=config.stage,family=config.model_family,step=step,epoch=epoch,
            hours_in_epoch=position,total_training_hours=len(hours),updated_unix=time.time(),
            elapsed_seconds=prior_elapsed+time.monotonic()-started,last_loss=history[-1] if history else None,
            best=best,pending_evaluation=pending,
            development=evaluation_history[-2:] if evaluation_history else [],
            peak_memory_bytes=torch.cuda.max_memory_allocated() if config.device.startswith('cuda') else 0)
        write_json(output/'progress.json',progress)
        with (output/'events.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(progress,ensure_ascii=False,allow_nan=False)+'\n')
        print('TRAIN_PROGRESS',json.dumps({key:progress[key] for key in ('status','stage','family','step','epoch','elapsed_seconds')},allow_nan=False),flush=True)

    def stage_result(status):
        return dict(status=status,checkpoint=str(checkpoint),step=step,epochs=epoch,
                    cursor=dict(position=position),dependency_hashes=dependencies,best=best,baseline=baseline,
                    candidates=candidates,history=history,evaluation_history=evaluation_history,
                    selection_status='formal' if any(candidates.values()) else 'baseline_only_no_formal_candidates',
                    raw=str(checkpoint),ema=str(checkpoint),
                    telemetry={**telemetry,'elapsed_seconds':prior_elapsed+time.monotonic()-started})

    def evaluate_pending():
        nonlocal pending,bad_evaluations
        if not pending: return
        callback=evaluate or config.evaluation_callback
        eval_hours=(config.development_hours if config.development_hours is not None else
                    getattr(data,'development_hours',[]))
        if not len(eval_hours):
            pending=False;save('running');return
        if callback is None:
            from .evaluation import evaluate_model
            callback=evaluate_model
        purpose=('baseline' if step==0 else 'startup' if step in config.evaluation_check_updates else
                 'preselection' if step<config.formal_selection_start_step else 'formal')
        eligible=purpose=='formal'
        evaluation_started=time.monotonic()
        rng=capture_rng()
        raw_state=_cpu_state(model)
        was_training=model.training
        original_best=copy.deepcopy(best)
        original_baseline=copy.deepcopy(baseline)
        original_candidates=copy.deepcopy(candidates)
        original_evaluation_history=copy.deepcopy(evaluation_history)
        original_bad_evaluations=bad_evaluations

        def rollback():
            best.clear();best.update(original_best)
            baseline.clear();baseline.update(original_baseline)
            candidates.clear();candidates.update(original_candidates)
            evaluation_history[:]=original_evaluation_history

        try:
            model.eval()
            for source in ('raw','ema'):
                model.load_state_dict(raw_state if source=='raw' else ema)
                restore_rng(rng)
                missing=object()
                previous_stop=getattr(config,'_evaluation_should_stop',missing)
                config._evaluation_should_stop=lambda: stop_requested or (
                    config.budget_seconds is not None and time.monotonic()-started>=config.budget_seconds)
                try:
                    with torch.no_grad():
                        raw=callback(model,data,eval_hours,config)
                finally:
                    if previous_stop is missing:
                        config.__dict__.pop('_evaluation_should_stop',None)
                    else:
                        config._evaluation_should_stop=previous_stop
                metrics=(raw.as_dict() if hasattr(raw,'as_dict') else raw.to_dict() if hasattr(raw,'to_dict') else raw)
                if not isinstance(metrics,dict):
                    raise TypeError('Evaluation must return metrics mapping')
                metrics=_plain_metrics(metrics)
                values={'point':metrics.get('point',metrics.get('rmse_macro_mm')),
                        'joint':metrics.get('joint',metrics.get('es_mm'))}
                if any(value is None for value in values.values()):
                    raise ValueError('Development evaluation requires both point and joint metrics')
                if any(not math.isfinite(float(value)) for value in values.values()):
                    raise ValueError('Nonfinite development metric')
                evaluation_history.append(dict(step=step,epoch=epoch,weight_source=source,
                    purpose=purpose,selection_eligible=eligible,evaluation_hours=len(eval_hours),metrics=metrics))
                if not eligible and purpose!='baseline':
                    continue
                prefix='candidate' if eligible else 'baseline'
                path=str(output/f'{prefix}_step{step:09d}_{source}.pt')
                for product,value in values.items():
                    selection=dict(value=float(value),step=step,epoch=epoch,weight_source=source,
                                   purpose=purpose,checkpoint=path,metrics=copy.deepcopy(metrics))
                    if eligible:
                        pool=[item for item in candidates[product]
                              if (item['step'],item['weight_source'])!=(step,source)]
                        candidates[product]=sorted(pool+[selection],key=_candidate_key)[:3]
                    elif product not in baseline or _candidate_key(selection)<_candidate_key(baseline[product]):
                        baseline[product]=selection
            for product in ('point','joint'):
                if candidates[product]:
                    best[product]=candidates[product][0]
                elif product in baseline:
                    best[product]=baseline[product]
        except Exception:
            rollback();raise
        finally:
            model.load_state_dict(raw_state)
            model.train(was_training);restore_rng(rng)
            telemetry['evaluation_seconds']+=time.monotonic()-evaluation_started
        try:
            # Publish all finalists before last.pt commits their paths and hashes.
            # A common step/source artifact can serve both product rankings, so
            # the formal point/joint top-three union contains at most six files.
            selections=list(baseline.values())+[item for group in candidates.values() for item in group]
            published={}
            for selection in selections:
                path=selection['checkpoint']
                if 'sha256' not in selection:
                    if path not in published:
                        selected=state('candidate')
                        selected.update(best={},baseline={},candidates={'point':[],'joint':[]},
                                        pending_evaluation=False,selected_weight_source=selection['weight_source'],
                                        selected_step=step,selection_purpose=selection['purpose'])
                        atomic_torch(path,selected)
                        published[path]=sha256(path)
                    selection['sha256']=published[path]
            # best entries alias their winning pool records; copy hashes even if
            # recovery loaded dictionaries without those aliases.
            for product,selection in best.items():
                winner=candidates[product][0] if candidates[product] else baseline[product]
                selection.update(winner)
            improved=eligible and any(not original_candidates[product] or
                candidates[product][0]['value']<original_candidates[product][0]['value']
                for product in ('point','joint'))
            if eligible:
                bad_evaluations=0 if improved else bad_evaluations+1
            pending=False
            save('running')
        except Exception:
            pending=True
            rollback()
            bad_evaluations=original_bad_evaluations
            raise

    total=math.ceil(len(hours)/config.batch_size)*config.epochs
    status='running'
    try:
        if not checkpoint.exists():
            # Step0 records a control baseline; formal finalist slots start after the registered boundary.
            pending=config.stage != 'posterior_cache'
            save('running')
        evaluate_pending()
        while epoch<config.epochs:
            if (stop_requested or (config.max_updates is not None and step>=config.max_updates) or
                (config.budget_seconds is not None and time.monotonic()-started>=config.budget_seconds)):
                status='paused';break
            selected=order[position:min(position+config.batch_size,len(order))]
            data_started=time.monotonic()
            if config.stage in ('gan_pretrain','diffusion_pretrain'):
                indices=torch.tensor([cache_index[int(h)] for h in selected])
                cond=cache['global_features'][indices].to(config.device)
                mu=cache['posterior_mu'][indices].to(config.device)
                logstd=cache['posterior_logstd'][indices].to(config.device)
                telemetry['data_seconds']+=time.monotonic()-data_started
                model_started=time.monotonic()
                def posterior_draw():
                    return (mu+logstd.exp()*torch.randn_like(mu)-model.latent_mean)/model.latent_std
                for opt in optimizers.values():
                    for group in opt.param_groups: group['lr']=_learning_rate(config,step,total)
                if config.stage=='gan_pretrain':
                    critic.requires_grad_(True)
                    for _ in range(3):
                        optimizers['critic'].zero_grad(set_to_none=True)
                        real=posterior_draw()
                        noise=torch.randn(len(selected),1,config.latent_dim,device=config.device)
                        fake=model.prior_model(noise,cond)[:,0].detach()
                        dloss=critic(fake,cond).mean()-critic(real,cond).mean()+10*_gp(critic,real,fake,cond)
                        dloss.backward();optimizers['critic'].step();optimizer_steps['critic']+=1
                    critic.requires_grad_(False)
                    noise=torch.randn(len(selected),1,config.latent_dim,device=config.device)
                    loss=-critic(model.prior_model(noise,cond)[:,0],cond).mean()
                    losses={'loss':loss,'critic':dloss}
                else:
                    losses={'loss':edm_dsm_loss(model.prior_model,posterior_draw(),cond)}
            else:
                batch=data.batch(selected,role='B',device=config.device)
                telemetry['data_seconds']+=time.monotonic()-data_started
                model_started=time.monotonic()
                if (batch['target_mask'].sum(-1)<4).any():
                    raise ValueError('Training hour requires at least four valid B station observations')
                if config.stage=='posterior_cache':
                    with torch.no_grad():
                        encoded=model.encode(batch)
                        mu,logstd=model.vae.posterior_parameters(encoded,_residual(batch),batch['query_indices'],
                            batch['query_weights'],batch['target_geometry'],batch['target_mask'])
                    cache_parts.append(dict(hour_ids=torch.as_tensor(selected.copy()),global_features=encoded.global_features.cpu(),
                                            posterior_mu=mu.cpu(),posterior_logstd=logstd.cpu()))
                    losses={}
                else:
                    for opt in optimizers.values():
                        for group in opt.param_groups: group['lr']=_learning_rate(config,step,total)
                    if critic is not None:
                        critic.requires_grad_(True)
                        with torch.no_grad():
                            encoded=model.encode(batch)
                            fields=_draw_fields(model,batch,config,encoded)
                            fake=(_observation_samples(fields,batch,config)[:,0]-_surface(batch['background'],batch)[:,0])/10
                            local=query_features(encoded.pyramid[0],batch['query_indices'],batch['query_weights'])
                            real=_residual(batch)/10
                        for _ in range(3):
                            optimizers['critic'].zero_grad(set_to_none=True)
                            args=(local,batch['target_geometry'],batch['target_mask'])
                            dloss=critic(fake,*args).mean()-critic(real,*args).mean()+10*_gp(critic,real,fake,local,batch['target_mask'],batch['target_geometry'])
                            dloss.backward();optimizers['critic'].step();optimizer_steps['critic']+=1
                        critic.requires_grad_(False)
                    losses=objective(model,batch,step=step)
            if losses:
                if config.stage=='complete_model_es' and model.family=='gan' and step%100==0:
                    losses.update(adversarial_gradient_scale(config.es_weight*losses['es'],losses['adversarial'],
                                                               generator_parameters,config.adversarial_weight))
                loss=losses['loss']
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
                optimizers['generator'].zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm=nn.utils.clip_grad_norm_(generator_parameters,config.gradient_clip,error_if_nonfinite=True)
                optimizers['generator'].step();optimizer_steps['generator']+=1
                # EMA remains on the model device; serialization copies it only at checkpoints.
                for key,value in model.state_dict().items():
                    value=value.detach()
                    if value.is_floating_point(): ema[key].lerp_(value,1-config.ema_decay)
                    else: ema[key].copy_(value)
                history.append(dict(step=step+1,epoch=epoch,learning_rate=optimizers['generator'].param_groups[0]['lr'],
                                    gradient_norm=float(gradient_norm.detach()),**{k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1}))
            telemetry['model_seconds']+=time.monotonic()-model_started
            step+=1;position+=len(selected)
            if position==len(order):
                epoch+=1;position=0;order=cursor_rng.permutation(hours)
                pending=(config.stage != 'posterior_cache' and
                         (epoch%config.evaluate_every_epochs==0 or step in config.evaluation_check_updates))
                save('running');evaluate_pending()
                if epoch>=config.min_epochs and config.patience>0 and bad_evaluations>=config.patience:
                    status='early_stopped';break
            elif config.stage!='posterior_cache' and step in config.evaluation_check_updates:
                pending=True;save('running');evaluate_pending()
            elif step%config.checkpoint_every_updates==0:
                save('running')
        if status=='running':status='complete'
        result=stage_result(status)
        if config.stage=='posterior_cache' and status=='complete':
            arrays={key:torch.cat([p[key] for p in cache_parts]) for key in cache_parts[0]}
            mu=arrays['posterior_mu'];logstd=arrays['posterior_logstd']
            mean=mu.mean(0)
            std=(mu.square().add((2*logstd).exp()).mean(0)-mean.square()).clamp_min(1e-8).sqrt()
            payload={**arrays,'standardization':dict(mean=mean,std=std),
                     'representation_hashes':_representation_hashes(model),'dependency_hashes':dependencies,
                     'training_only':True,'data_identity':data_identity,'b_label_identity':label_identity}
            path=output/'posterior_cache.pt';atomic_torch(path,payload);result['cache']=str(path)
        save(status)
        result['raw']=str(checkpoint);result['ema']=str(checkpoint)
        result['telemetry']={**telemetry,'elapsed_seconds':prior_elapsed+time.monotonic()-started}
        return result
    except EvaluationPaused:
        # A cooperative allocation/signal pause rolls back the unfinished raw/EMA
        # transaction while retaining every completed optimizer update.
        pending=True
        save('paused')
        result=stage_result('paused')
        result['pause_reason']='development_evaluation'
        return result
    except Exception:
        # A failed evaluation remains pending in the checkpoint and is retried on resume.
        save('error');raise
    finally:
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
