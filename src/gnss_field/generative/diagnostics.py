"""Registered B-internal representation diagnostics, never C/D selection.

The frozen historical background has already seen B. These are training-internal
hyperdiagnostics, not an independent validation claim or sensor calibration.
"""
from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json
from gnss_field.evaluation import crps_ensemble
from gnss_field.models.diffusion import keyed_noise
from .evaluation import _latent_fields, _evaluation_batches, _member_chunk
from .observations import query_surface


SIGMA_CANDIDATES = (3.18, 6.36, 12.72)


def calendar_panels(eligible_hours):
    """Predetermined days 2/4 versus day 15 in each 2016--2018 month."""
    eligible = set(np.asarray(eligible_hours, 'i8').tolist())
    fit, validation = [], []
    for month in np.arange(np.datetime64('2016-01'), np.datetime64('2019-01')):
        start = int(month.astype('datetime64[h]').astype('i8'))
        fit.extend(h for day in (2, 4) for h in range(start+(day-1)*24, start+day*24) if h in eligible)
        validation.extend(h for h in range(start+14*24, start+15*24) if h in eligible)
    if fit and validation:
        # Only neighboring hours in the merged order can be closest.
        tagged = sorted([(h, 0) for h in fit]+[(h, 1) for h in validation])
        if any(b-a < 18 and ka != kb for (a,ka),(b,kb) in zip(tagged, tagged[1:])):
            raise ValueError('Weather-panel separation must be at least 18 hours')
    return dict(fit_hours=fit, validation_hours=validation,
                calendar_rule='2016-2018 monthly days 2+4 all hours fit; day 15 all hours validation',
                nominal_fit_hours=1728, nominal_validation_hours=864,
                eligibility_intersection=True, weather_separation_hours_min=18)


def physical_group_folds(metadata):
    """Seed42 size-balanced 3 folds over connected physical/50km cell unions."""
    table = pd.DataFrame(metadata).reset_index(drop=True)
    if not {'physical_group','dense_group'}.issubset(table):
        raise ValueError('B station metadata requires physical_group and dense_group')
    parent = list(range(len(table)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for column in ('physical_group','dense_group'):
        seen = {}
        for i, key in enumerate(table[column]):
            if pd.isna(key):
                continue  # Missing IDs are singletons, not one artificial group.
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = {}
    for i in range(len(table)):
        components.setdefault(find(i), []).append(i)
    if len(components) < 3:
        raise ValueError('At least three connected physical/dense groups are required')
    groups = list(components.values())
    rng = np.random.default_rng(42)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    groups.sort(key=len, reverse=True)  # stable randomized ties
    counts = np.zeros(3, 'i8'); folds = np.empty(len(table), 'i8')
    for group in groups:
        fold = int(np.argmin(counts)); folds[group] = fold; counts[fold] += len(group)
    return folds


def _station_metadata(data):
    if hasattr(data, 'station_metadata'):
        return pd.DataFrame(data.station_metadata).reset_index(drop=True)
    if hasattr(data, 'store'):
        return data.store.stations.iloc[data.store.ids('B')].reset_index(drop=True)
    raise ValueError('B station metadata is required before diagnostics')


class BTrainingView:
    """Keep the true B role while removing whole held groups from q and loss."""
    purpose = 'train'
    development_hours = ()

    def __init__(self, data, hours, folds, held_fold):
        self.data = data
        self.train_hours = np.asarray(hours, 'i8')
        self.folds = np.asarray(folds, 'i8')
        self.held_fold = int(held_fold)
        self.metadata = {**data.metadata, 'diagnostic_role':'B_training_internal',
                         'fold_signature':fingerprint(dict(folds=self.folds.tolist(),held=self.held_fold)),
                         'panel_hours':self.train_hours.tolist()}

    def batch(self, hour_ids, role='B', device='cpu'):
        if role != 'B':
            raise PermissionError('Training diagnostics can access only B labels')
        out = dict(self.data.batch(hour_ids, role='B', device=device))
        keep = torch.as_tensor(self.folds != self.held_fold, device=out['target'].device)
        if len(keep) != out['target'].shape[1]:
            raise ValueError('B station metadata/order differs from target population')
        out['target_mask'] = out['target_mask'] & keep[None]
        out['target'] = torch.where(out['target_mask'], out['target'], 0.)
        return out


def _condition_only(batch):
    return {key:value for key,value in batch.items() if key not in ('target','target_mask')}


def _surface(fields, batch):
    return query_surface(fields, batch['query_indices'].to(fields.device),
                         batch['query_weights'].to(fields.device), batch['target_geometry'][:,2].to(fields.device))


def _deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError('Training-only diagnostic wall-clock budget exhausted; no selection published')


def _macro(values, masks, *, root=False):
    values = np.asarray(values, 'f8'); masks = np.asarray(masks, bool)
    count = masks.sum(0); valid = count > 0
    if not valid.any():
        raise ValueError('No fixed B diagnostic observations')
    means = np.where(masks, values, 0).sum(0)/np.maximum(count,1)
    if root:
        means = np.sqrt(means)
    return float(means[valid].mean())


@torch.no_grad()
def score_B_prior(model, data, hours, held_mask, *, sigma_eff_mm,
                  device='cpu', deadline=None, evaluation_hours_batch=1, evaluation_chunk=4):
    """Prior-only forecasting: IID K16 CRPS; independent K32 point mean.

    NLL is the K16 Gaussian-kernel predictive mixture with fixed sigma_eff;
    it is a finite-Monte-Carlo working score, not an analytic field likelihood.
    """
    model.eval(); held_mask = np.asarray(held_mask, bool)
    errors, crps, nlls, masks = [], [], [], []
    for batch_hours in _evaluation_batches(hours, evaluation_hours_batch):
        _deadline(deadline)
        batch = data.batch(batch_hours, role='B', device=device)
        condition = _condition_only(batch)
        fields = _latent_fields(model, condition, 16, 2026112242, 18, evaluation_chunk).cpu()
        clean = _surface(fields, batch)
        error = keyed_noise(batch_hours,16,2026112243,(len(held_mask),),device='cpu')
        observed = clean+float(sigma_eff_mm)*error
        point_fields = _latent_fields(model, condition, 32, 2026112244, 18, evaluation_chunk).cpu()
        mean = _surface(point_fields, batch).mean(1)
        truth = batch['target'].cpu()
        mask = batch['target_mask'].cpu().numpy() & held_mask[None]
        crps.extend(crps_ensemble(truth.numpy(),observed.permute(0,2,1).numpy()))
        errors.extend((mean-truth).square().numpy())
        kernels = -.5*((clean-truth[:,None])/sigma_eff_mm).square()-math.log(sigma_eff_mm)-.5*math.log(2*math.pi)
        nlls.extend((-torch.logsumexp(kernels,1)+math.log(16)).numpy())
        masks.extend(mask)
    return dict(prior_rmse_mm=_macro(errors,masks,root=True),
                station_macro_crps_mm=_macro(crps,masks), station_macro_nll=_macro(nlls,masks),
                role='B_training_diagnostic', probability_members=16, point_members=32,
                probability_members_iid=True, point_stream_independent=True,
                prior_only=True, sigma_eff_mm=float(sigma_eff_mm),
                scored_station_hours=int(np.asarray(masks).sum()),
                station_weights='equal eligible B station; fixed per-station hour masks',
                nll_method='K16 finite prior-field Gaussian-discrepancy mixture')


def choose_latent_dimension(baseline, diagnostic):
    """Upgrade only if all three predictive/capability endpoints support it."""
    keys = ('prior_rmse_mm','posterior_withheld_rmse_mm','station_macro_crps_mm')
    if len(baseline) != 3 or len(diagnostic) != 3:
        raise ValueError('Latent decision requires all three completed paired folds')
    improvements = {}
    eligible = True
    for key in keys:
        a = np.asarray([row[key] for row in baseline], 'f8')
        b = np.asarray([row[key] for row in diagnostic], 'f8')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a <= 0).any():
            raise ValueError('Latent decision requires finite positive original outcomes')
        gains = (a-b)/a
        improvements[key] = dict(paired_relative_improvements=gains.tolist(),
                                 mean_relative_improvement=float(gains.mean()),
                                 all_folds_improve=bool((gains > 0).all()))
        eligible &= bool((gains > 0).all() and gains.mean() >= .01)
    return dict(latent_dim=128 if eligible else 64, threshold_relative=.01,
                improvements=improvements, statistical_certainty='not established; training-internal three-fold diagnostic')


def _posterior_capability(model, batch, fit_mask, noise_seed=2026112245, evaluation_chunk=None):
    """q reads only fitB labels; returned predictions can be queried at heldB."""
    encoded = model.encode(_condition_only(batch))
    base = _surface(batch['background'],batch)[:,0]
    residual = torch.where(fit_mask,batch['target']-base,0.)
    mu, logstd = model.vae.posterior_parameters(encoded,residual,batch['query_indices'],
                         batch['query_weights'],batch['target_geometry'],fit_mask)
    mp, lp = model.vae.prior_parameters(encoded)
    noise = keyed_noise(batch['hour_ids'],16,noise_seed,(model.latent_dim,),device=mu.device)
    latent = mu[:,None]+logstd.exp()[:,None]*noise
    if evaluation_chunk is None:
        fields = model.vae.decoder(encoded,latent,batch['background'])
        prediction = _surface(fields,batch).mean(1)
    else:
        chunk = _member_chunk(evaluation_chunk,len(batch['hour_ids']))
        # Preserve the original member-axis mean; only field decoding is tiled.
        surfaces = [_surface(model.vae.decoder(encoded,latent[:,start:start+chunk],batch['background']),batch)
                    for start in range(0,16,chunk)]
        prediction = torch.cat(surfaces,dim=1).mean(1)
    kl = lp-logstd+((2*logstd).exp()+(mu-mp).square())/(2*(2*lp).exp())-.5
    return prediction, kl, encoded, mu


@torch.no_grad()
def _posterior_withheld_rmse(model,data,hours,folds,held_fold,device,deadline,
                             evaluation_hours_batch=1,evaluation_chunk=4):
    errors=[]; masks=[]
    keep = torch.as_tensor(np.asarray(folds)!=held_fold,device=device)
    for batch_hours in _evaluation_batches(hours,evaluation_hours_batch):
        _deadline(deadline)
        batch=data.batch(batch_hours,role='B',device=device)
        fit=batch['target_mask'] & keep[None]
        if not fit.any(1).all():
            raise ValueError('B posterior diagnostic has empty fit population')
        pred,*_= _posterior_capability(model,batch,fit,evaluation_chunk=evaluation_chunk)
        errors.extend((pred-batch['target']).square().cpu().numpy())
        masks.extend((batch['target_mask'] & ~keep[None]).cpu().numpy())
    return _macro(errors,masks,root=True)


def _pilot_signature(config,data,panel,folds,fold,sigma,latent):
    label=data._label_identity('B') if callable(getattr(data,'_label_identity',None)) else None
    return fingerprint(dict(scientific=config.get('scientific',{}),condition=data.metadata,B_labels=label,
        panel=panel,folds=np.asarray(folds).tolist(),fold=fold,sigma=sigma,latent=latent))


def _pilot(config,data,panel,folds,fold,sigma,latent,run_root):
    """All retries debit the same pilot cap, including failed reads/scoring."""
    output=Path(run_root)/'training_only_diagnostics'/f'sigma_{sigma:g}_latent_{latent}_fold_{fold}'
    output.mkdir(parents=True,exist_ok=True)
    committed=output/'outcome.commit.json'
    if committed.exists():
        pointer=read_json(committed)
        if sha256(output/'outcome.json')!=pointer['outcome_sha256'] or sha256(output/'population.lock.json')!=pointer['population_sha256']:
            raise ValueError('Committed pilot outcome or population changed')
        previous=read_json(output/'outcome.json')
        if previous.get('pilot_signature')!=_pilot_signature(config,data,panel,folds,fold,sigma,latent):
            raise ValueError('Committed pilot data or scientific configuration changed')
        if (sha256(previous['checkpoint'])!=previous['checkpoint_sha256'] or
            previous['training_outcome']['status']!='complete' or previous['training_outcome']['epochs']!=5):
            raise ValueError('Committed pilot checkpoint differs or is incomplete')
        return previous
    ledger_path=output/'pilot_budget_ledger.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else dict(charged_seconds=0.,attempts=0)
    budget=float(config['runtime']['diagnostic_pilot_budget_seconds'])
    if not math.isfinite(budget) or budget<=0:
        raise ValueError('Pilot requires a finite positive cumulative budget')
    remaining=min(budget-float(ledger['charged_seconds']),
                  float(config['runtime'].get('_diagnostic_remaining_seconds',math.inf)))
    if remaining<=0:
        raise TimeoutError('Cumulative training-only pilot budget exhausted')
    started=time.perf_counter()
    limited={**config,'runtime':{**config['runtime'],'diagnostic_pilot_budget_seconds':remaining}}
    try:
        return _execute_pilot(limited,data,panel,folds,fold,sigma,latent,run_root)
    finally:
        ledger['charged_seconds']+=time.perf_counter()-started
        ledger['attempts']+=1
        ledger['budget_seconds']=budget
        write_json(ledger_path,ledger)


def _globally_bounded_pilot(config,data,panel,folds,fold,sigma,latent,run_root):
    """Debit every pilot attempt to one shared diagnostics envelope."""
    root=Path(run_root);root.mkdir(parents=True,exist_ok=True)
    path=root/'diagnostics_budget_ledger.json'
    ledger=read_json(path) if path.exists() else dict(charged_seconds=0.,attempts=0)
    cap=float(config['runtime'].get('diagnostics_budget_seconds',
                    config['runtime'].get('diagnostics_remaining_seconds',0)))
    if not math.isfinite(cap) or cap<=0:
        raise ValueError('A measured global diagnostics budget is required')
    remaining=min(cap-float(ledger['charged_seconds']),
                  float(config['runtime'].get('diagnostics_deadline_unix',math.inf))-time.time())
    if remaining<=0:
        raise TimeoutError('Cumulative global diagnostic budget exhausted')
    limited={**config,'runtime':{**config['runtime'],'_diagnostic_remaining_seconds':remaining}}
    started=time.perf_counter()
    try:
        return _pilot(limited,data,panel,folds,fold,sigma,latent,root)
    finally:
        ledger['charged_seconds']+=time.perf_counter()-started
        ledger['attempts']+=1;ledger['budget_seconds']=cap
        write_json(path,ledger)


def _execute_pilot(config,data,panel,folds,fold,sigma,latent,run_root):
    from .training import run_stage, load_stage_model
    runtime=config['runtime']; scientific=config.get('scientific',{})
    budget=float(runtime['diagnostic_pilot_budget_seconds'])
    deadline=time.perf_counter()+budget
    output=Path(run_root)/'training_only_diagnostics'/f'sigma_{sigma:g}_latent_{latent}_fold_{fold}'
    output.mkdir(parents=True,exist_ok=True)
    view=BTrainingView(data,panel['fit_hours'],folds,fold)
    # Validity filtering is fixed once per fold and common to every candidate.
    eligible=[]; excluded=[]
    for hour in view.train_hours:
        _deadline(deadline)
        valid=int(view.batch([int(hour)])['target_mask'].sum())
        (eligible if valid>=4 else excluded).append(int(hour))
    if not eligible:
        raise ValueError('No pilot-fitting hours retain four fitB stations')
    view=BTrainingView(data,eligible,folds,fold)
    identity=dict(fold=fold,folds=folds.tolist(),fit_hours=eligible,excluded_fit_hours=excluded,
                  validation_hours=panel['validation_hours'],role='B_training_internal',
                  sigma_eff_mm=sigma,latent_dim=latent)
    immutable_json(output/'population.lock.json',identity)
    training=dict(stage='vae_representation',model_family='vae',seed=42,epochs=5,min_epochs=5,
                  batch_size=8,latent_dim=latent,sigma_eff_mm=sigma,
                  grid_shape=tuple(scientific.get('grid_shape',data.metadata.get('grid_shape',(144,96)))),
                  output_dir=str(output),device=runtime.get('device','cpu'),
                  lr=float(scientific.get('lr',1e-4)),minimum_lr=float(scientific.get('minimum_lr',1e-6)),
                  reference_count=float(scientific.get('reference_count',442.)),
                  kl_warmup_updates=math.ceil(math.ceil(len(eligible)/8)*5*.1),
                  budget_seconds=max(.001,deadline-time.perf_counter()),patience=0,development_hours=[])
    result=run_stage(training,view)
    write_json(output/'training_outcome.json',result)
    if result['status']=='paused':
        raise TimeoutError('Pilot training paused with a recoverable checkpoint')
    if result['status']!='complete' or result['epochs']!=5:
        raise RuntimeError('Pilot incomplete: original outcome retained; no selection lock')
    model,_=load_stage_model(result['checkpoint'],device=runtime.get('device','cpu'))
    score=score_B_prior(model,data,panel['validation_hours'],folds==fold,sigma_eff_mm=sigma,
                        device=runtime.get('device','cpu'),deadline=deadline,
                        evaluation_hours_batch=runtime.get('evaluation_hours_batch',1),
                        evaluation_chunk=runtime.get('evaluation_chunk',4))
    score['posterior_withheld_rmse_mm']=_posterior_withheld_rmse(model,data,panel['validation_hours'],folds,
                                                fold,runtime.get('device','cpu'),deadline,
                                                evaluation_hours_batch=runtime.get('evaluation_hours_batch',1),
                                                evaluation_chunk=runtime.get('evaluation_chunk',4))
    record={**identity,**score,'checkpoint':result['checkpoint'],'checkpoint_sha256':sha256(result['checkpoint']),
            'pilot_signature':_pilot_signature(config,data,panel,folds,fold,sigma,latent),
            'training_outcome':result}
    write_json(output/'outcome.json',record)
    write_json(output/'outcome.commit.json',dict(outcome_sha256=sha256(output/'outcome.json'),
                                               population_sha256=sha256(output/'population.lock.json')))
    return record


def run_representation_diagnostics(config,data,run_root):
    """Run 9 sigma pilots then 3 latent128 pilots, only in explicit backend mode."""
    runtime=config.get('runtime',{})
    if not runtime.get('backend_runtime') or not config.get('diagnostics',{}).get('execute_pilots'):
        raise RuntimeError('Actual diagnostic pilots require explicit backend runtime and execute_pilots')
    budget=runtime.get('diagnostic_pilot_budget_seconds')
    if budget is None or not math.isfinite(float(budget)) or float(budget)<=0:
        raise ValueError('Every diagnostic pilot requires a finite positive measured budget')
    root=Path(run_root); root.mkdir(parents=True,exist_ok=True)
    panel=calendar_panels(data.train_hours)
    if not panel['fit_hours'] or not panel['validation_hours']:
        raise ValueError('Registered fit and weather-validation panels must both be nonempty')
    meta=_station_metadata(data); folds=physical_group_folds(meta)
    identity=dict(panel=panel,folds=folds.tolist(),seed=42,
                  B_station_metadata=meta[['physical_group','dense_group']].fillna('').astype(str).to_dict('list'),
                  condition_metadata=data.metadata,scientific=config.get('scientific',{}),
                  B_label_identity=data._label_identity('B') if callable(getattr(data,'_label_identity',None)) else None,
                  diagnostics=config.get('diagnostics',{}),diagnostic_source_sha256=sha256(__file__),
                  role='B_training_internal',sigma_candidates_mm=SIGMA_CANDIDATES,
                  fit_epochs=5,min_epochs=5,batch_size=8,KL_warmup_fraction=.1)
    immutable_json(root/'training_only_diagnostic_plan.lock.json',identity)
    signature=fingerprint(identity)
    lock=root/'training_only_selection.lock.json'
    if lock.exists():
        existing=read_json(lock)
        if existing['diagnostic_plan_signature']!=signature:
            raise ValueError('Training-only selection lock does not match registered data/plan')
        return existing
    records={}
    for sigma in SIGMA_CANDIDATES:
        records[str(sigma)]=[_globally_bounded_pilot(config,data,panel,folds,fold,sigma,64,root) for fold in range(3)]
    # Lexicographic CRPS primary and working mixture-NLL secondary, fixed sigma tie order.
    sigma=min(SIGMA_CANDIDATES,key=lambda s:(np.mean([r['station_macro_crps_mm'] for r in records[str(s)]]),
                                           np.mean([r['station_macro_nll'] for r in records[str(s)]]),s))
    latent128=[_globally_bounded_pilot(config,data,panel,folds,fold,sigma,128,root) for fold in range(3)]
    decision=choose_latent_dimension(records[str(sigma)],latent128)
    selection=dict(status='complete',diagnostic_plan_signature=signature,sigma_eff_mm=sigma,
                   latent_dim=decision['latent_dim'],latent_decision=decision,
                   sigma64_original_outcomes=records,latent128_original_outcomes=latent128,
                   role='B_training_internal',selection_rule='mean station-macro CRPS; NLL secondary',
                   observation_kernel='fixed effective discrepancy shared by VAE/GAN/diffusion; not sensor calibration',
                   background_exposure='historically trained B; internal hyperdiagnostics only',
                   independent_development='C2019 unseen in these diagnostics; D sealed',
                   statistical_certainty='not established',negative_gate_fabricates_no_scores=True)
    immutable_json(lock,selection)
    return selection


def representation_metrics(model,data,hours,config,deadline=None):
    """Finite/query/z diagnostics and sparse fitB-only posterior/oracle capability.

    The oracle optimizes latent z against fitB, then queries heldB. It is not
    area truth and does not establish inference skill. All gradients stop at
    frozen network weights; only temporary latent variables are optimized.
    """
    model.eval(); runtime=config.get('runtime',{}); options=config.get('diagnostics',{})
    device=runtime.get('device','cpu'); folds=physical_group_folds(_station_metadata(data))
    steps=int(options.get('oracle_steps',32))
    if not 1<=steps<=128:
        raise ValueError('Oracle diagnostic requires 1..128 bounded latent-only steps')
    prior_errors=[]; background_errors=[]; label_differences=[]
    fit_errors=[]; fit_masks=[]; held_errors=[]; held_masks=[]
    oracle_errors=[]; oracle_masks=[]; masks=[]; kls=[]; field_var=[]; station_var=[]; equivalence=[]
    original_requires=[p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    try:
        for hour in np.asarray(hours,'i8'):
            _deadline(deadline)
            batch=data.batch([int(hour)],role='B',device=device)
            with torch.no_grad():
                fields=_latent_fields(model,_condition_only(batch),16,2026112242,18,4).cpu()
                surface=_surface(fields,batch)
                perturbed=dict(batch)
                perturbed['target']=torch.full_like(batch['target'],98765.4321)
                perturbed['target_mask']=~batch['target_mask']
                changed=_latent_fields(model,perturbed,16,2026112242,18,4).cpu()
                label_differences.append(float((changed-fields).abs().max()))
                if not torch.isfinite(fields).all() or not torch.isfinite(surface).all():
                    raise ValueError('Nonfinite representation field/query')
                field_var.append(float(fields.var(1,unbiased=False).mean()))
                valid=batch['target_mask'][0].cpu()
                station_var.append(float(surface[0].var(0,unbiased=False)[valid].mean()))
                # Independently gather the bilinear stencil, then restore height.
                idx=batch['query_indices'].cpu(); weights=batch['query_weights'].cpu()
                explicit=(fields.flatten(-2)[:,:,0,:,None].expand(-1,-1,-1,4)
                          .gather(2,idx[None,None].expand(fields.shape[0],fields.shape[1],-1,-1))*weights).sum(-1)
                explicit=explicit*torch.exp(-batch['target_geometry'][:,2].cpu()/3000)
                equivalence.append(float((explicit-surface).abs().max()))
                prior_errors.append((surface[0].mean(0)-batch['target'][0].cpu()).square().numpy())
                background_errors.append((_surface(batch['background'],batch)[0,0]-batch['target'][0]).square().cpu().numpy())
                masks.append(valid.numpy())
            for fold in range(3):
                _deadline(deadline)
                keep=torch.as_tensor(folds!=fold,device=device)
                fit=batch['target_mask'] & keep[None]; held=batch['target_mask'] & ~keep[None]
                if not fit.any() or not held.any():
                    raise ValueError('Representation panel requires nonempty fitB and heldB groups')
                with torch.no_grad():
                    posterior,kl,encoded,mu=_posterior_capability(model,batch,fit)
                    kls.append(kl[0].cpu().numpy())
                    squared=(posterior-batch['target']).square()[0].cpu().numpy()
                    fit_errors.append(squared);fit_masks.append(fit[0].cpu().numpy())
                    held_errors.append(squared);held_masks.append(held[0].cpu().numpy())
                z=mu.detach().clone().requires_grad_(True)
                optimizer=torch.optim.Adam([z],lr=.1)
                for _ in range(steps):
                    _deadline(deadline)
                    optimizer.zero_grad()
                    pred=_surface(model.vae.decoder(encoded,z[:,None],batch['background']),batch)[:,0]
                    loss=torch.where(fit,pred-batch['target'],0.).square().sum()/fit.sum()
                    loss.backward();optimizer.step()
                with torch.no_grad():
                    pred=_surface(model.vae.decoder(encoded,z[:,None],batch['background']),batch)[:,0]
                    oracle_errors.append((pred-batch['target']).square()[0].cpu().numpy())
                    oracle_masks.append(held[0].cpu().numpy())
    finally:
        for p,requires in zip(model.parameters(),original_requires): p.requires_grad_(requires)
    if not masks:
        raise ValueError('Representation panel is empty')
    mean_kl=np.asarray(kls).mean(0)
    result=dict(finite=True,grid_query_max_error_mm=max(equivalence),
                label_isolation=bool(max(label_differences)<=1e-6),
                label_perturbation_max_error_mm=max(label_differences),
                field_latent_variance_mm2=float(np.mean(field_var)),
                station_latent_variance_mm2=float(np.mean(station_var)),
                kl_per_dim=mean_kl.tolist(),active_kl_units=int((mean_kl>.01).sum()),
                active_kl_threshold=.01,prior_rmse_mm=_macro(prior_errors,masks,root=True),
                background_rmse_mm=_macro(background_errors,masks,root=True),
                posterior_fit_rmse_mm=_macro(fit_errors,fit_masks,root=True),
                posterior_withheld_rmse_mm=_macro(held_errors,held_masks,root=True),
                oracle_fitB_withheld_rmse_mm=_macro(oracle_errors,oracle_masks,root=True),
                oracle_steps=steps,oracle_scope='latent fit on fitB only; query heldB; not area truth',
                diagnostic_role='B_training_internal',panel_hours=np.asarray(hours,'i8').tolist(),
                posterior_prior_nearness_alone_rejects=False,
                independent_station_noise_in_latent_test=False)
    reasons=[]
    if not result['label_isolation']: reasons.append('inference_target_label_path')
    if result['grid_query_max_error_mm']>.001: reasons.append('grid_query_disagreement')
    if min(result['field_latent_variance_mm2'],result['station_latent_variance_mm2'])<=1e-10:
        reasons.append('latent_path_collapsed')
    if not all(math.isfinite(result[key]) for key in ('field_latent_variance_mm2','station_latent_variance_mm2',
              'prior_rmse_mm','posterior_fit_rmse_mm','posterior_withheld_rmse_mm','oracle_fitB_withheld_rmse_mm')):
        reasons.append('nonfinite_representation')
    result.update(passed=not reasons,reasons=reasons)
    return result


def check_representation(config,data,checkpoint,run_root):
    from .training import load_stage_model
    runtime=config.get('runtime',{})
    budget=runtime.get('representation_check_budget_seconds')
    if not runtime.get('backend_runtime') or budget is None or not math.isfinite(float(budget)) or float(budget)<=0:
        raise RuntimeError('Representation check requires explicit backend runtime and finite measured budget')
    deadline=time.perf_counter()+float(budget)
    panel=calendar_panels(data.train_hours)['fit_hours']
    # A deterministic evenly spread 32-hour subset; no label ranking or resampling.
    if len(panel)<32:
        raise ValueError('Full representation qualification requires its fixed 32-hour B panel')
    hours=np.asarray(panel,'i8')[np.linspace(0,len(panel)-1,32,dtype=int)]
    root=Path(run_root)
    immutable_json(root/'representation_check_panel.lock.json',dict(hours=hours.tolist(),role='B',
                   calendar_source='registered fit panel; 32 evenly spaced time-order indices'))
    model,_=load_stage_model(checkpoint,runtime.get('device','cpu'))
    result=representation_metrics(model,data,hours,config,deadline=deadline)
    result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
    write_json(root/'representation_diagnostics.json',result)
    return result
