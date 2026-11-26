"""Locked-population field evaluation with separate point and IID member streams."""
from pathlib import Path
import io
import signal
import time

import numpy as np
import torch

from gnss_field.artifacts import atomic_bytes, fingerprint, immutable_json, model_digest, read_json, sha256, write_json
from gnss_field.models.diffusion import heun_sample, keyed_noise
from gnss_field.geometry import coarsen_100km
from gnss_field.evaluation import point_metrics
from .observations import query_surface


def fixed_development_hours(hours):
    """Two time-order quantiles within each month and nominal-hour stratum."""
    hours = np.unique(np.asarray(hours, 'i8'))
    dates = hours.astype('datetime64[h]')
    months = dates.astype('datetime64[M]')
    selected = []
    for month in np.unique(months):
        for nominal_hour in range(24):
            group = hours[(months == month) & (hours % 24 == nominal_hour)]
            if len(group):
                selected.extend(group[[int((len(group)-1)/3), int(2*(len(group)-1)/3)]])
    return np.unique(np.asarray(selected, 'i8'))


class LockedPopulation:
    """A precommitted C2019 population, independent of a model's predictions."""
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.spec = read_json(self.path)
        if self.spec['role'] != 'C' or self.spec['year'] != 2019:
            raise ValueError('Only the registered C2019 development manifest is allowed')
        self.loaded = {}

    def mask(self, hour):
        month = str(np.datetime64(int(hour), 'h').astype('datetime64[M]'))
        if month not in self.loaded:
            row = self.spec['monthly'][month]
            path = Path(row['file'])
            if not path.is_absolute():
                path = self.path.parent/path
            if sha256(path) != row['sha256']:
                raise ValueError('Locked evaluation archive hash changed')
            with np.load(path, allow_pickle=False) as archive:
                self.loaded[month] = (archive[row.get('hour_key', 'hours')].copy(),
                                      archive[row.get('mask_key', 'mask')].astype(bool))
        hours, mask = self.loaded[month]
        index = np.searchsorted(hours, int(hour))
        if index >= len(hours) or hours[index] != int(hour):
            raise ValueError('Evaluation hour missing from locked population')
        return mask[index]


def _development_view(data):
    if getattr(data, 'purpose', None) == 'train' and hasattr(data, '_other_view'):
        return data._other_view('development')
    return data


def _evaluation_batches(hours, hours_batch=1):
    """Runtime-only tiling, preserving the requested hour order and identities."""
    size = int(hours_batch)
    if not 1 <= size <= 16:
        raise ValueError('Evaluation hour batch must be between 1 and 16')
    hours = np.asarray(hours, 'i8')
    for start in range(0, len(hours), size):
        yield hours[start:start+size]


def _member_chunk(chunk, batch_size):
    """Bound a decoder launch to at most 256 hour/member pairs."""
    if int(chunk) < 1:
        raise ValueError('Positive decoding chunk required')
    return min(int(chunk), max(1, 256//int(batch_size)))


@torch.no_grad()
def _latent_fields(model, batch, members, seed, steps, chunk, antithetic=False):
    """Decode chunks without changing hourly/member noise identities."""
    if not hasattr(model, 'vae'):
        fields = model.sample_fields(batch, members, seed, steps=steps)
        if not torch.isfinite(fields).all():
            raise ValueError('Nonfinite published field')
        return fields
    chunk = _member_chunk(chunk, len(batch['hour_ids']))
    encoded = model.encode(batch)
    count = (members+1)//2 if antithetic else members
    noise = keyed_noise(batch['hour_ids'], count, seed, (model.latent_dim,),
                        device=batch['background'].device)
    if antithetic:
        noise = torch.stack((noise, -noise), dim=2).flatten(1, 2)[:, :members]
    if model.deterministic:
        latent = torch.zeros_like(noise)
    elif model.family == 'vae':
        mu, logstd = model.vae.prior_parameters(encoded)
        latent = mu[:, None]+logstd.exp()[:, None]*noise
    elif model.family == 'gan':
        latent = model.prior_model(noise, encoded.global_features)*model.latent_std+model.latent_mean
    else:
        condition = encoded.global_features[:, None].expand(-1, members, -1).flatten(0, 1)
        latent = heun_sample(lambda state, sigma: model.prior_model.denoise(state, sigma, condition),
                             noise.flatten(0, 1), steps=steps).reshape_as(noise)
        latent = latent*model.latent_std+model.latent_mean
    fields = []
    for start in range(0, members, chunk):
        values = model.vae.decoder(encoded, latent[:, start:start+chunk], batch['background'])
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite published field')
        fields.append(values.cpu())
    return torch.cat(fields, dim=1)


@torch.no_grad()
def _evaluate_hours(model, data, hours, config, arrays_only=False):
    """Return both selection values and actual metrics; never use posterior q."""
    model.eval();data = _development_view(data)
    hours = np.asarray(hours, 'i8')
    if not len(hours) or len(np.unique(hours)) != len(hours):
        raise ValueError('Evaluation hours must be nonempty and unique')
    members = int(getattr(config, 'eval_members', 16))
    point_members = int(getattr(config, 'point_members', 64))
    if members < 2 or point_members < 4 or point_members % 2:
        raise ValueError('Need IID K>=2 and an even independent point pool K>=4')
    seed = int(getattr(config, 'evaluation_seed', 20261123))
    point_seed = int(getattr(config, 'point_seed', seed+1000000))
    if point_seed == seed:
        raise ValueError('Point and probability noise streams must be distinct')
    chunk = int(getattr(config, 'evaluation_chunk', 4))
    if chunk < 1:
        raise ValueError('Positive decoding chunk required')
    population_path = getattr(config, 'evaluation_population_manifest', None)
    population = LockedPopulation(population_path) if population_path else None
    samples = [];means = [];truths = [];masks = [];coarse_means = [];ses = [];backgrounds=[];nlls=[]
    for batch_hours in _evaluation_batches(hours, getattr(config, 'evaluation_hours_batch', 1)):
        should_stop = getattr(config, '_evaluation_should_stop', None)
        if should_stop is not None and should_stop():
            if arrays_only and samples:
                break  # Return a durable completed prefix to the outer writer.
            raise EvaluationPaused('Evaluation paused at hour-batch boundary')
        batch = data.batch(batch_hours, role='C', device=config.device)
        # Returned field chunks may live on CPU to bound GPU memory. Query on CPU.
        indices = batch['query_indices'].cpu();weights = batch['query_weights'].cpu()
        heights = batch['target_geometry'][:, 2].cpu()
        probability_fields = _latent_fields(model, batch, members, seed, config.sampling_steps, chunk)
        clean_samples = query_surface(probability_fields.cpu(), indices, weights, heights)
        error = keyed_noise(batch_hours, members, seed+2000000, (len(heights),), device='cpu')
        observed = clean_samples+float(config.sigma_eff_mm)*error
        sigma = float(config.sigma_eff_mm)
        components = -.5*((clean_samples-batch['target'].cpu()[:,None])/sigma).square()-np.log(sigma*np.sqrt(2*np.pi))
        observation_nll = -torch.logsumexp(components,dim=1)+np.log(members)
        point_fields = _latent_fields(model, batch, point_members, point_seed,
                                      config.sampling_steps, chunk, antithetic=True).cpu()
        points = query_surface(point_fields, indices, weights, heights)
        mean = points.mean(1)
        pair_means = points.reshape(len(batch_hours), point_members//2, 2, -1).mean(2)
        se = pair_means.std(1, unbiased=True)/np.sqrt(point_members//2)
        # Query and global 4x4 coarsening are linear: coarsen the mean once.
        coarse = coarsen_100km(point_fields.mean(1))[:, None]
        coarse_mean = query_surface(coarse, indices, weights, heights)[:, 0]
        mask = batch['target_mask'].cpu().numpy().astype(bool)
        if population:
            locked = np.stack([population.mask(hour) for hour in batch_hours])
            if locked.shape != mask.shape:
                raise ValueError('Locked evaluation station order/shape changed')
            # Target observations were sanitized to zero outside the current mask;
            # never invent valid labels that the data view did not actually supply.
            if np.any(locked & ~mask):
                raise ValueError('Current data mask cannot reproduce locked score population')
            mask = locked
        if not torch.isfinite(observed).all() or not torch.isfinite(mean).all():
            raise ValueError('Nonfinite station forecast')
        samples.extend(observed.numpy());means.extend(mean.numpy())
        truths.extend(batch['target'].cpu().numpy());masks.extend(mask)
        coarse_means.extend(coarse_mean.numpy());ses.extend(se.numpy())
        backgrounds.extend(query_surface(batch['background'].cpu(),indices,weights,heights)[:,0].numpy())
        nlls.extend(observation_nll.numpy())
    arrays = dict(hours=hours[:len(samples)], samples=np.asarray(samples), mean=np.asarray(means),
                  truth=np.asarray(truths), population=np.asarray(masks),
                  coarse_mean=np.asarray(coarse_means), point_mean_se=np.asarray(ses),
                  background_mean=np.asarray(backgrounds),observation_nll=np.asarray(nlls),
                  xy=batch['target_geometry'][:, :2].cpu().numpy(),heights=heights.numpy())
    if arrays_only:
        return arrays
    return _score_arrays(arrays, config)


def _score_arrays(arrays, config):
    from grid_generation_metrics import score_ensemble
    hours = arrays['hours']
    point_members = int(getattr(config, 'point_members', 64))
    scored = arrays['population'].any(1)
    if not scored.any():
        raise ValueError('No fixed scored station observations')
    metrics = score_ensemble(arrays['truth'][scored], arrays['samples'][scored],
                             arrays['mean'][scored], arrays['population'][scored], arrays['xy'])
    coarse_metrics = point_metrics(arrays['truth'], arrays['coarse_mean'], arrays['population'])
    nll_count=arrays['population'].sum(0)
    macro_nll=(np.where(arrays['population'],arrays['observation_nll'],0.).sum(0)[nll_count>0]/nll_count[nll_count>0]).mean()
    metrics.update(point=metrics['station_macro_rmse_mm'], joint=metrics['group_ES_mm'],
                   point_mean_mc_se_mm=float(arrays['point_mean_se'][arrays['population']].mean()),
                   coarse_station_macro_rmse_mm=coarse_metrics['station_macro_rmse_mm'],
                   station_macro_observation_nll=float(macro_nll),
                   nll_method='finite clean-field mixture with fixed Gaussian effective discrepancy',
                   hours_requested=len(hours), hours_scored=int(scored.sum()),
                   empty_fixed_population_hours=hours[~scored].tolist(),
                   probability_members_iid=True, point_members=point_members,
                   point_mean_estimator='independent antithetic latent stream; analytical zero-mean discrepancy',
                   sigma_eff_mm=float(config.sigma_eff_mm), role='C', selection_year=2019,
                   observation_kernel='one independent effective-discrepancy draw per station/member')
    return metrics


class EvaluationPaused(RuntimeError):
    """Completed evaluation blocks are durable; a fresh allocation can resume."""


@torch.no_grad()
def evaluate_model(model, data, hours, config, output=None):
    """Evaluate a fixed population, optionally resuming immutable weekly blocks."""
    if output is None:
        return _evaluate_hours(model, data, hours, config)
    hours = np.asarray(hours, 'i8')
    if not len(hours) or len(np.unique(hours)) != len(hours):
        raise ValueError('Evaluation hours must be nonempty and unique')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    population = getattr(config, 'evaluation_population_manifest', None)
    identity = dict(schema=1, hours=hours.tolist(),
        model=model_digest(model) if hasattr(model,'state_dict') else str(type(model)),
        condition=fingerprint(getattr(data,'metadata',{})),
        C_label_identity=data._label_identity('C') if callable(getattr(data,'_label_identity',None)) else None,
        population_sha256=sha256(population) if population else None,
        evaluation_source_sha256=sha256(__file__),
        options={k:getattr(config,k,None) for k in ('eval_members','point_members','evaluation_seed',
             'point_seed','sampling_steps','sigma_eff_mm')})
    immutable_json(output/'evaluation.lock.json',identity)
    previous_handlers = {}; paused = False
    def stop(signum, frame):
        nonlocal paused
        paused = True
    for name in ('SIGUSR1','SIGTERM','SIGINT'):
        number = getattr(signal, name, None)
        if number is not None:
            try: previous_handlers[number] = signal.signal(number, stop)
            except ValueError: pass
    started = time.monotonic(); budget = getattr(config,'budget_seconds',None)
    previous_callback = getattr(config,'_evaluation_should_stop',None)
    def should_stop():
        return (paused or (budget is not None and time.monotonic()-started >= budget) or
                (previous_callback is not None and previous_callback()))
    config._evaluation_should_stop = should_stop
    parts = []
    try:
        for offset in range(0,len(hours),168):
            path = output/'blocks'/f'{offset:05d}.npz'
            commit = path.with_suffix('.json')
            prefix = None
            if commit.exists():
                record = read_json(commit)
                path = commit.parent/record.get('file',path.name)
                if sha256(path) != record['sha256']:
                    raise ValueError('Evaluation block checksum changed')
                with np.load(path,allow_pickle=False) as archive:
                    arrays = {key:archive[key].copy() for key in archive.files}
                if not np.array_equal(arrays['hours'],hours[offset:offset+len(arrays['hours'])]):
                    raise ValueError('Evaluation block hours changed')
                if len(arrays['hours']) < len(hours[offset:offset+168]):
                    prefix = arrays
            if not commit.exists() or prefix is not None:
                if paused or (budget is not None and time.monotonic()-started >= budget):
                    raise EvaluationPaused('Evaluation paused at durable block boundary')
                count = len(prefix['hours']) if prefix is not None else 0
                arrays = _evaluate_hours(model,data,hours[offset+count:offset+168],config,arrays_only=True)
                if prefix is not None:
                    arrays = {key:(prefix[key] if key in ('xy','heights') else np.concatenate((prefix[key],arrays[key]))) for key in prefix}
                buffer = io.BytesIO(); np.savez_compressed(buffer,**arrays)
                import hashlib
                digest = hashlib.sha256(buffer.getvalue()).hexdigest()
                # The atomic pointer only ever refers to a complete immutable file.
                path = commit.with_name(f'{offset:05d}.{digest[:16]}.npz')
                atomic_bytes(path,buffer.getvalue())
                write_json(commit,dict(file=path.name,sha256=digest,hour_count=len(arrays['hours'])))
                write_json(output/'progress.json',dict(status='running',completed_hours=offset+len(arrays['hours']),
                                                       requested_hours=len(hours)))
                if len(arrays['hours']) < len(hours[offset:offset+168]):
                    raise EvaluationPaused('Evaluation paused after committing completed hour prefix')
            parts.append(arrays)
        arrays = {key:(parts[0][key] if key in ('xy','heights') else np.concatenate([p[key] for p in parts])) for key in parts[0]}
        metrics = _score_arrays(arrays,config)
        buffer = io.BytesIO();np.savez_compressed(buffer, **arrays)
        atomic_bytes(output/'station_predictions.npz',buffer.getvalue())
        write_json(output/'metrics.json',metrics)
        write_json(output/'progress.json',dict(status='complete',completed_hours=len(hours),requested_hours=len(hours)))
        return metrics
    finally:
        for number,handler in previous_handlers.items(): signal.signal(number,handler)
        if previous_callback is None:
            del config._evaluation_should_stop
        else:
            config._evaluation_should_stop = previous_callback
