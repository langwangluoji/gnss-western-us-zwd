"""Complete local training of independent VAE, GAN and spatial diffusion.

The data population is the audited local experiment, not all server hours.
Diffusion is optimized by observations through its full Heun chain, with no
invented dense target. Evaluation uses free prior samples for every family.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from contextlib import contextmanager
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from simple_weather_unet_pilot import PROJECT, ROOT as DATA_ROOT, PilotData, metrics
from train_enhanced_encoder import ROOT, atomic_checkpoint
from gnss_field.artifacts import immutable_json, write_json, read_json, sha256
from gnss_field.models.conditioned_unet import (FieldVAE, FieldGAN, FieldDiffusion, StationCritic, fair_scores, finite)
from gnss_field.evaluation import variogram_score

FACTORIES = dict(vae=FieldVAE, gan=FieldGAN, diffusion=FieldDiffusion)
COUNTS = dict(vae=1800, gan=1600, diffusion=1000)
SEED = 42
CLOCK_START = time.monotonic()
PRIOR_SPENT = 0.
LOCAL_CAP = 3600


class EvaluationPaused(RuntimeError):
    pass


class LocalBudgetExhausted(RuntimeError):
    pass


def budget_seconds():
    return PRIOR_SPENT+time.monotonic()-CLOCK_START


def commit_budget(status):
    write_json(ROOT/'local_budget.json', dict(consumed_seconds=budget_seconds(), cap_seconds=LOCAL_CAP,
               status=status, updated=time.time(), includes='benchmark, training and evaluation in this process'))


@contextmanager
def single_process():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'active.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit('Another independent-generator process already owns this experiment')
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_random(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def make_model(family):
    seed_everything(SEED)
    model = FACTORIES[family]().cuda()
    source = ROOT/'encoder_seed42/best.pt'
    state = torch.load(source, map_location='cuda', weights_only=False)
    model.core.load_state_dict(state['model'])
    return model


def query_members(data, fields, ids):
    batch, members = fields.shape[:2]
    return data.query(fields.flatten(0, 1)[:, 0], ids).reshape(batch, members, -1)


def posterior_grid(data, rows, ids, base):
    # This function is called only with the locked fitting-B indices.
    if not np.array_equal(np.asarray(ids), data.a['split_fit']):
        raise PermissionError('Posterior may read only this experiment fitting B')
    mask = data.mask[rows][:, ids]
    residual = (data.labels[rows][:, ids]-data.query(base[:, 0], ids))/data.height_factor[ids]/10
    residual = torch.where(mask, residual, 0.)
    indices, weight = data.qi[ids], data.qw[ids]*mask[..., None]
    batch, _, height, width = base.shape
    numerator = base.new_zeros(batch, height*width)
    denominator = base.new_zeros(batch, height*width)
    index = indices[None].expand(batch, -1, -1).flatten(1)
    numerator.scatter_add_(1, index, (residual[..., None]*weight).flatten(1))
    denominator.scatter_add_(1, index, weight.flatten(1))
    value = numerator/denominator.clamp(min=1e-8)
    return torch.stack((value, denominator.log1p()), 1).reshape(batch, 2, height, width)


def elbo(fields, parameters, data, rows, ids):
    pred = query_members(data, fields, ids)
    mask = data.mask[rows][:, ids]
    target = data.labels[rows][:, ids]
    reference = float(np.median(data.a['mask'][data.rows('train')][:, ids].sum(1)))
    clean = torch.where(mask, target, 0.)
    error = torch.where(mask[:, None], pred-clean[:, None], 0.)
    nll = (.5*(error/3.).square()*mask[:, None]).sum(-1).mean()/reference
    mu_q, log_q, mu_p, log_p = parameters
    kl = (log_p-log_q+.5*(torch.exp(2*(log_q-log_p))+(mu_q-mu_p).square()*torch.exp(-2*log_p)-1)).flatten(1).sum(1).mean()/reference
    return nll+kl, nll, kl


def critic_features(data, x, ids):
    return (x[:, :4].flatten(2)[:, :, data.qi[ids]]*data.qw[ids][None, None]).sum(-1).transpose(1, 2).detach()


def update(model, optimizer, data, rows, family, step, critic=None, critic_optimizer=None):
    model.train()
    ids = data.a['split_fit']
    optimizer.zero_grad(set_to_none=True)
    totals = dict(loss=0., es_mm=0., crps_mm=0., nll=0., kl=0., adversarial=0., critic=0.)
    for row in rows:
        batch_rows = np.array([row])
        x, base = data.batch(batch_rows, True)
        hours = data.hours[batch_rows]
        encoded = model.encode(x)
        mask = data.mask[batch_rows][:, ids]
        target = data.labels[batch_rows][:, ids]
        if family == 'vae' and step <= 600:
            post = posterior_grid(data, batch_rows, ids, base)
            fields, params = model.posterior_sample(encoded, base, post, hours, 2, SEED+step*1009)
            loss, nll, kl = elbo(fields, params, data, batch_rows, ids)
            totals['nll'] += float(nll.detach())/len(rows)
            totals['kl'] += float(kl.detach())/len(rows)
        else:
            fields = model.sample_encoded(encoded, base, hours, 4, SEED+step*1009)
            samples = query_members(data, fields, ids)
            es, crps = fair_scores(samples/10, target/10, mask)
            loss = es+.2*crps
            totals['es_mm'] += float(es.detach())*10/len(rows)
            totals['crps_mm'] += float(crps.detach())*10/len(rows)
            if family == 'vae':
                post = posterior_grid(data, batch_rows, ids, base)
                reconstructed, params = model.posterior_sample(encoded, base, post, hours, 1, SEED+step*1013)
                regularizer, nll, kl = elbo(reconstructed, params, data, batch_rows, ids)
                loss = loss+.1*regularizer
                totals['nll'] += float(nll.detach())/len(rows)
                totals['kl'] += float(kl.detach())/len(rows)
            if family == 'gan':
                condition = critic_features(data, x, ids)
                geometry = torch.as_tensor(data.a['xy'][ids]/1e6, dtype=x.dtype, device=x.device)
                geometry = torch.cat([geometry, torch.as_tensor(data.a['heights'][ids, None]/3000, dtype=x.dtype, device=x.device)], -1)
                background = data.query(base[:, 0], ids)
                fake = (samples-background[:, None])/10
                real = (target-background)/10
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                critic_optimizer.zero_grad(set_to_none=True)
                d_real = critic(real[:, None], condition, geometry, mask)
                d_fake = critic(fake.detach(), condition, geometry, mask)
                d_loss = F.relu(1-d_real).mean()+F.relu(1+d_fake).mean()
                d_loss.backward()
                d_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                finite(d_norm)
                critic_optimizer.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                adv = -critic(fake, condition, geometry, mask).mean()
                loss = loss+.05*adv
                totals['adversarial'] += float(adv.detach())/len(rows)
                totals['critic'] += float(d_loss.detach())/len(rows)
        finite(loss)
        (loss/len(rows)).backward()
        totals['loss'] += float(loss.detach())/len(rows)
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    finite(grad)
    optimizer.step()
    totals['gradient_norm'] = float(grad)
    return totals


def distribution_metrics(samples, truth, mask, xy):
    result = metrics(samples.mean(1), truth, mask)
    clean = np.where(mask, truth, 0.)
    z = torch.as_tensor(samples, dtype=torch.float64)
    y, m = torch.as_tensor(clean), torch.as_tensor(mask)
    es, crps = fair_scores(z, y, m)
    result.update(ES_mm=float(es), CRPS_mm=float(crps), members=samples.shape[1])
    for level in (80, 90, 95):
        alpha = (100-level)/200
        lo, hi = np.quantile(samples, [alpha, 1-alpha], axis=1)
        counts = mask.sum(0)
        selected = counts > 0
        result[f'coverage{level}'] = float(((mask*((truth >= lo) & (truth <= hi))).sum(0)[selected]/counts[selected]).mean())
        result[f'width{level}_mm'] = float(((hi-lo)*mask).sum(0)[selected].__truediv__(counts[selected]).mean())
    _, groups = np.unique(np.floor(xy/200000).astype(int), axis=0, return_inverse=True)
    group_es, group_vs, shuffled_vs = [], [], []
    rng = np.random.default_rng(20261123)
    for group in np.unique(groups):
        stations = np.flatnonzero(groups == group)
        if len(stations) < 4:
            continue
        scores, vs, shuffled = [], [], []
        for t in range(len(truth)):
            ids = stations[mask[t, stations]]
            if len(ids) < 4:
                continue
            forecasts, observations = samples[t][:, ids], truth[t, ids]
            value, _ = fair_scores(torch.as_tensor(forecasts[None], dtype=torch.float64),
                                   torch.as_tensor(observations[None]), torch.ones(1, len(ids), dtype=torch.bool))
            distance = np.linalg.norm(xy[ids, None]-xy[None, ids], axis=-1)/1000
            scores.append(float(value))
            vs.append(variogram_score(observations, forecasts, distance))
            permuted = np.stack([rng.permutation(forecasts[:, i]) for i in range(len(ids))], 1)
            shuffled.append(variogram_score(observations, permuted, distance))
        if scores:
            group_es.append(np.mean(scores))
            group_vs.append(np.mean(vs))
            shuffled_vs.append(np.mean(shuffled))
    result.update(group_ES_mm=float(np.mean(group_es)) if group_es else None,
                  group_VS=float(np.mean(group_vs)) if group_vs else None,
                  shuffled_group_VS=float(np.mean(shuffled_vs)) if shuffled_vs else None,
                  eligible_groups=len(group_es), estimator='fair off-diagonal ES/CRPS')
    return result


@torch.no_grad()
def sample_population(model, data, rows, ids, members, seed=20261123):
    model.eval()
    result = []
    for row in rows:
        if (ROOT/'PAUSE').exists():
            raise EvaluationPaused('Paused before next complete hourly ensemble')
        if budget_seconds() >= LOCAL_CAP:
            raise LocalBudgetExhausted('Cumulative local wall-clock cap reached during evaluation')
        x, base = data.batch(np.array([row]), True)
        # Fixed prefix across member chunking; every member is a full field.
        fields = model.sample(x, base, [int(data.hours[row])], members, seed)
        result.append(query_members(data, fields, ids)[0].cpu().numpy())
    return np.stack(result)


def evaluate(model, data, rows, ids, members):
    samples = sample_population(model, data, rows, ids, members)
    truth, mask = data.a['labels'][np.ix_(rows, ids)], data.a['mask'][np.ix_(rows, ids)]
    return distribution_metrics(samples, truth, mask, data.a['xy'][ids]), samples


def science_lock():
    return dict(schema='independent_spatial_generators_v1', seed=SEED, updates=COUNTS,
                train_batch=4, train_members=4, diffusion_steps=6, diffusion_calls=11,
                diffusion_objective='full spatial sampler trained through observations by fair ES/CRPS; no DSM targets',
                vae_likelihood_scale_mm=3., likelihood_scale_role='fixed training assumption, not ensemble coverage inflation',
                widths=[32, 64, 128], decoder=[32, 16], lr=.0002, minimum_lr=.00001,
                input_cache=read_json(DATA_ROOT/'data_complete.json')['sha256'],
                initial_mean=sha256(ROOT/'encoder_seed42/best.pt'),
                model_code=sha256(PROJECT/'src/gnss_field/models/conditioned_unet.py'),
                data_loader=sha256(PROJECT/'tools/simple_weather_unet_pilot.py'),
                sampling_code=sha256(PROJECT/'src/gnss_field/models/diffusion.py'),
                evaluation_code=sha256(PROJECT/'src/gnss_field/evaluation/metrics.py'),
                training_code=sha256(__file__), local_wall_cap_seconds=LOCAL_CAP,
                validation_hours=24, validation_members=8, final_members=32,
                roles='A context; fitting B training; inner B selection; exposed outer B and C development; D sealed')


def benchmark(data):
    result = {}
    rows = data.rows('train')[:4]
    for family in FACTORIES:
        model = make_model(family)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
        critic = StationCritic().cuda() if family == 'gan' else None
        opt_d = torch.optim.AdamW(critic.parameters(), lr=.0002, betas=(.0, .9)) if critic else None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        record = update(model, optimizer, data, rows, family, 601 if family == 'vae' else 1, critic, opt_d)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        finite(torch.tensor(list(record.values())))
        # Exact optimizer/RNG roundtrip on the actual training entry.
        checkpoint = ROOT/f'benchmark_{family}.pt'
        atomic_checkpoint(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         critic=critic.state_dict() if critic else None,
                                         critic_optimizer=opt_d.state_dict() if opt_d else None, rng=random_state()))
        update(model, optimizer, data, rows, family, 602 if family == 'vae' else 2, critic, opt_d)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if critic:
            critic.load_state_dict(saved['critic'])
            opt_d.load_state_dict(saved['critic_optimizer'])
        restore_random(saved['rng'])
        update(model, optimizer, data, rows, family, 602 if family == 'vae' else 2, critic, opt_d)
        max_error = max(float((expected[k]-v).abs().max()) for k, v in model.state_dict().items())
        if max_error != 0:
            raise ValueError(f'Actual {family} optimizer resume differs: {max_error}')
        result[family] = dict(update_seconds=elapsed, peak_mb=torch.cuda.max_memory_allocated()/2**20,
                             parameters=sum(p.numel() for p in model.parameters()),
                             resume_max_parameter_difference=max_error, metrics=record)
        del model, optimizer, critic, opt_d
        torch.cuda.empty_cache()
    write_json(ROOT/'benchmark.json', dict(passed=True, families=result,
               code=dict(model=sha256(PROJECT/'src/gnss_field/models/conditioned_unet.py'), training=sha256(__file__)),
               projected_training_seconds=sum(result[f]['update_seconds']*COUNTS[f] for f in COUNTS)))
    print('GENERATOR_BENCHMARK_COMPLETE', json.dumps(result), flush=True)


def train_family(data, family):
    out = ROOT/family
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        return read_json(out/'complete.json')
    model = make_model(family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    critic = StationCritic().cuda() if family == 'gan' else None
    opt_d = torch.optim.AdamW(critic.parameters(), lr=.0002, betas=(.0, .9)) if critic else None
    rng = np.random.default_rng(SEED)
    allowed = data.rows('train')
    steps = COUNTS[family]
    schedule = np.concatenate([rng.permutation(allowed) for _ in range(math.ceil(steps*4/len(allowed)))])[:steps*4].reshape(steps, 4)
    inner = data.rows('inner').reshape(12, -1)[:, [0, 6]].ravel()
    ids = data.a['split_inner']
    best_point, best_joint, start, spent = float('inf'), float('inf'), 0, 0.
    history = []
    pending_evaluation = False
    contract_sha256 = sha256(ROOT/'generators.lock.json')
    if (out/'last.pt').exists():
        state = torch.load(out/'last.pt', map_location='cuda', weights_only=False)
        if state.get('family') != family or state.get('contract_sha256') != contract_sha256:
            raise ValueError('Checkpoint belongs to another family or scientific release')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if critic:
            critic.load_state_dict(state['critic'])
            opt_d.load_state_dict(state['critic_optimizer'])
        restore_random(state['rng'])
        start, spent = state['step'], state['spent_seconds']
        best_point, best_joint, history = state['best_point'], state['best_joint'], state['history']
        pending_evaluation = state.get('pending_evaluation', False)
    else:
        initial, _ = evaluate(model, data, inner, ids, 8)
        history.append(dict(step=0, phase='initialization_only', validation=initial))
    started = time.monotonic()
    accumulated = []

    def save_progress(step, pending=False):
        atomic_checkpoint(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    critic=critic.state_dict() if critic else None, critic_optimizer=opt_d.state_dict() if opt_d else None,
                    step=step, schedule=schedule, rng=random_state(), spent_seconds=spent+time.monotonic()-started,
                    best_point=best_point, best_joint=best_joint, history=history,
                    family=family, contract_sha256=contract_sha256, pending_evaluation=pending))

    def select_candidates(step, validation):
        nonlocal best_point, best_joint
        if family == 'vae' and step <= 600:
            return
        payload = dict(model=model.state_dict(), step=step, phase='observation_proper_scores',
                       validation=validation, family=family, contract_sha256=contract_sha256)
        if validation['macro_rmse'] < best_point:
            best_point = validation['macro_rmse']
            atomic_checkpoint(out/'point_best.pt', payload)
        if validation['group_ES_mm'] is None:
            raise ValueError('No eligible registered station group for joint selection')
        if validation['group_ES_mm'] < best_joint:
            best_joint = validation['group_ES_mm']
            atomic_checkpoint(out/'joint_best.pt', payload)

    if pending_evaluation:
        validation, _ = evaluate(model, data, inner, ids, 8)
        select_candidates(start, validation)
        history.append(dict(step=start, phase='resumed_pending_evaluation', validation=validation))
        if family == 'vae' and start == 600:
            atomic_checkpoint(out/'elbo_pretrained.pt', dict(model=model.state_dict(), step=start))
        save_progress(start)
        write_json(out/'curve.json', history)
    for step in range(start+1, steps+1):
        pretrain = family == 'vae' and step <= 600
        local_step, duration = (step-1, 600) if pretrain else (step-601, 1200) if family == 'vae' else (step-1, steps)
        lr = .00001+.00019*.5*(1+math.cos(math.pi*max(local_step, 0)/max(duration-1, 1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        record = update(model, optimizer, data, schedule[step-1], family, step, critic, opt_d)
        accumulated.append(record)
        elapsed = spent+time.monotonic()-started
        pause = (ROOT/'PAUSE').exists()
        budget = budget_seconds() >= LOCAL_CAP
        if pause or budget:
            save_progress(step, pending=(step % 200 == 0 or step == steps))
            status = 'PAUSED' if pause else 'BUDGET_EXHAUSTED'
            write_json(ROOT/'status.json', dict(stage=status, family=family, step=step, updated=time.time()))
            commit_budget(status)
            return dict(status=status, step=step)
        if step % 25 == 0 and step % 200:
            write_json(ROOT/'status.json', dict(stage='GENERATIVE_TRAINING', family=family, step=step,
                       updated=time.time(), lr=lr, training=record, last_validation_step=history[-1]['step']))
        if step % 200 == 0 or step == steps:
            save_progress(step, pending=True)
            validation, _ = evaluate(model, data, inner, ids, 8)
            event = dict(step=step, phase='observation_ELBO' if pretrain else 'observation_proper_scores',
                         lr=lr, seconds=elapsed, validation=validation,
                         losses={k: float(np.mean([r[k] for r in accumulated])) for k in record},
                         peak_mb=torch.cuda.max_memory_allocated()/2**20)
            accumulated = []
            history.append(event)
            commit_budget('TRAINING')
            select_candidates(step, validation)
            if family == 'vae' and step == 600:
                atomic_checkpoint(out/'elbo_pretrained.pt', dict(model=model.state_dict(), step=step))
            save_progress(step)
            write_json(out/'curve.json', history)
            with (out/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')
            write_json(ROOT/'status.json', dict(stage='PAUSED' if pause else 'BUDGET_EXHAUSTED' if budget else 'GENERATIVE_TRAINING',
                       family=family, updated=time.time(), **event))
            print(json.dumps(dict(family=family, **event)), flush=True)
            if pause or budget:
                return dict(status='PAUSED' if pause else 'BUDGET_EXHAUSTED', step=step)
    products = {}
    fixed = allowed[np.rint(np.linspace(0, len(allowed)-1, 48)).astype(int)]
    populations = dict(spatial=(fixed, data.a['split_outer']), temporal=(data.rows('outer'), data.a['split_fit']),
                       spatial_temporal=(data.rows('outer'), data.a['split_outer']),
                       C2019_development=(data.rows('development'), data.a['C_ids']))
    for product in ('point', 'joint'):
        checkpoint = out/f'{product}_best.pt'
        selected = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if selected.get('family') != family or selected.get('contract_sha256') != contract_sha256:
            raise ValueError('Selected product differs from the scientific release')
        model.load_state_dict(selected['model'])
        scores = {}
        for cohort, (rows, target_ids) in populations.items():
            if budget_seconds() >= LOCAL_CAP:
                commit_budget('BUDGET_EXHAUSTED_DURING_EVALUATION')
                return dict(status='BUDGET_EXHAUSTED', step=steps, phase='final_evaluation')
            artifact = out/f'{product}_{cohort}.npz'
            receipt_path = out/f'{product}_{cohort}.json'
            if receipt_path.exists():
                existing = read_json(receipt_path)
                if existing['checkpoint_sha256'] != sha256(checkpoint) or existing['data_sha256'] != read_json(DATA_ROOT/'data_complete.json')['sha256']:
                    raise ValueError('Completed evaluation belongs to another product')
                if existing['artifact_sha256'] != sha256(artifact) or existing['members'] != 32:
                    raise ValueError('Completed ensemble evaluation changed')
                scores[cohort] = existing['metrics']
                continue
            value, samples = evaluate(model, data, rows, target_ids, 32)
            scores[cohort] = value
            temporary = artifact.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, samples=samples, mean=samples.mean(1),
                         target=data.a['labels'][np.ix_(rows, target_ids)], mask=data.a['mask'][np.ix_(rows, target_ids)],
                         hours=data.hours[rows], station_ids=target_ids)
            os.replace(temporary, artifact)
            write_json(receipt_path, dict(checkpoint_sha256=sha256(checkpoint), artifact_sha256=sha256(artifact),
                       data_sha256=read_json(DATA_ROOT/'data_complete.json')['sha256'], members=32, metrics=value))
            commit_budget('FINAL_EVALUATION')
        products[product] = dict(step=selected['step'], checkpoint_sha256=sha256(checkpoint), scores=scores)
    receipt = dict(status='COMPLETE', family=family, updates=steps, products=products,
                   seconds=spent+time.monotonic()-started, data_sha256=read_json(DATA_ROOT/'data_complete.json')['sha256'])
    write_json(out/'complete.json', receipt)
    return receipt


def main():
    global CLOCK_START, PRIOR_SPENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['benchmark', 'all', *FACTORIES], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    PRIOR_SPENT = read_json(ROOT/'local_budget.json')['consumed_seconds'] if (ROOT/'local_budget.json').exists() else 0.
    CLOCK_START = time.monotonic()
    if not (ROOT/'encoder_results.json').exists():
        raise ValueError('Both encoder trials must finish before generator training')
    source = read_json(DATA_ROOT/'data_complete.json')
    if sha256(DATA_ROOT/'data.npz') != source['sha256']:
        raise ValueError('Actual local condition cache differs from its audited checksum')
    if sha256(PROJECT/'tools/simple_weather_unet_pilot.py') != source['script_sha256']:
        raise ValueError('Input construction/query code differs from the audited cache release')
    data = PilotData()
    code = dict(model=sha256(PROJECT/'src/gnss_field/models/conditioned_unet.py'), training=sha256(__file__))
    if not (ROOT/'benchmark.json').exists() or read_json(ROOT/'benchmark.json').get('code') != code:
        benchmark(data)
    commit_budget('BENCHMARK_COMPLETE')
    if args.stage == 'benchmark':
        return
    immutable_json(ROOT/'generators.lock.json', science_lock())
    results = {}
    for family in FACTORIES if args.stage == 'all' else [args.stage]:
        result = train_family(data, family)
        results[family] = result
        if result['status'] != 'COMPLETE':
            return
    completed = {f: read_json(ROOT/f/'complete.json') for f in FACTORIES if (ROOT/f/'complete.json').exists()}
    write_json(ROOT/'generators_results.json', dict(status='COMPLETE' if len(completed) == 3 else 'PARTIAL', families=completed))
    write_json(ROOT/'status.json', dict(stage='COMPLETE' if len(completed) == 3 else 'PARTIAL', updated=time.time(), families=list(completed)))
    commit_budget('COMPLETE' if len(completed) == 3 else 'PARTIAL')
    print('INDEPENDENT_GENERATORS_COMPLETE', flush=True)


if __name__ == '__main__':
    with single_process():
        try:
            main()
        except EvaluationPaused as error:
            write_json(ROOT/'status.json', dict(stage='PAUSED', updated=time.time(), reason=str(error)))
            commit_budget('PAUSED')
        except LocalBudgetExhausted as error:
            write_json(ROOT/'status.json', dict(stage='BUDGET_EXHAUSTED', updated=time.time(), reason=str(error)))
            commit_budget('BUDGET_EXHAUSTED')
        except Exception as error:
            write_json(ROOT/'status.json', dict(stage='FAILED', updated=time.time(),
                       error_type=type(error).__name__, error=str(error)))
            commit_budget('FAILED')
            raise
