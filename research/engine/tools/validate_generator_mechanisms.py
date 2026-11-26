"""Paired local mechanism tests; no C/D scoring and no architecture changes."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import train_independent_generators as base
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256
from gnss_field.objectives.ensemble_mean import unbiased_mean_mse

SOURCE = base.ROOT
ROOT = base.PROJECT/'results/20261123_generator_mechanism_validation_v01'
ARMS = ('control', 'intervention')
SOURCE_CODE = Path(base.__file__)


def load_start(family):
    saved = torch.load(SOURCE/family/'last.pt', map_location='cuda', weights_only=False)
    if saved['contract_sha256'] != sha256(SOURCE/'generators.lock.json') or saved['family'] != family:
        raise ValueError('Source checkpoint release mismatch')
    model = base.make_model(family)
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    optimizer.load_state_dict(saved['optimizer'])
    critic = base.StationCritic().cuda() if family == 'gan' else None
    opt_d = torch.optim.AdamW(critic.parameters(), lr=.0002, betas=(0., .9)) if critic else None
    if critic:
        critic.load_state_dict(saved['critic'])
        opt_d.load_state_dict(saved['critic_optimizer'])
    base.restore_random(saved['rng'])
    return model, optimizer, critic, opt_d, saved


def controlled_update(model, optimizer, data, rows, family, step, critic=None, opt_d=None,
                      elbo_weight=.1, adversarial_weight=.05, mean_weight=0., noise_offset=0):
    model.train()
    ids = data.a['split_fit']
    optimizer.zero_grad(set_to_none=True)
    totals = dict(loss=0., es_mm=0., crps_mm=0., nll=0., kl=0., adversarial=0., critic=0., mean_u=0.)
    for row in rows:
        batch_rows = np.array([row])
        x, background = data.batch(batch_rows, True)
        hours = data.hours[batch_rows]
        encoded = model.encode(x)
        mask, target = data.mask[batch_rows][:, ids], data.labels[batch_rows][:, ids]
        fields = model.sample_encoded(encoded, background, hours, 4, base.SEED+step*1009+noise_offset)
        samples = base.query_members(data, fields, ids)
        es, crps = base.fair_scores(samples/10, target/10, mask)
        loss = es+.2*crps
        totals['es_mm'] += float(es.detach())*10/len(rows)
        totals['crps_mm'] += float(crps.detach())*10/len(rows)
        if family == 'vae' and elbo_weight:
            post = base.posterior_grid(data, batch_rows, ids, background)
            reconstructed, params = model.posterior_sample(encoded, background, post, hours, 1, base.SEED+step*1013+noise_offset)
            regularizer, nll, kl = base.elbo(reconstructed, params, data, batch_rows, ids)
            loss = loss+elbo_weight*regularizer
            totals['nll'] += float(nll.detach())/len(rows)
            totals['kl'] += float(kl.detach())/len(rows)
        if family == 'gan':
            condition = base.critic_features(data, x, ids)
            geometry = torch.as_tensor(data.a['xy'][ids]/1e6, dtype=x.dtype, device=x.device)
            geometry = torch.cat([geometry, torch.as_tensor(data.a['heights'][ids, None]/3000, dtype=x.dtype, device=x.device)], -1)
            background_stations = data.query(background[:, 0], ids)
            fake, real = (samples-background_stations[:, None])/10, (target-background_stations)/10
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            d_loss = F.relu(1-critic(real[:, None], condition, geometry, mask)).mean()+F.relu(1+critic(fake.detach(), condition, geometry, mask)).mean()
            d_loss.backward()
            d_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            base.finite(d_norm)
            opt_d.step()
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            adv = -critic(fake, condition, geometry, mask).mean()
            loss = loss+adversarial_weight*adv
            totals['adversarial'] += float(adv.detach())/len(rows)
            totals['critic'] += float(d_loss.detach())/len(rows)
        if mean_weight:
            value = unbiased_mean_mse(samples/10, target/10, mask)
            loss = loss+mean_weight*value
            totals['mean_u'] += float(value.detach())/len(rows)
        base.finite(loss)
        (loss/len(rows)).backward()
        totals['loss'] += float(loss.detach())/len(rows)
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    base.finite(grad)
    optimizer.step()
    totals['gradient_norm'] = float(grad)
    return totals


def flat_gradient(loss, model, retain_graph=True):
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p, g in zip(parameters, values)])


def gradient_audit(data, rows, family):
    model, _, critic, _, saved = load_start(family)
    model.eval()
    ids, records = data.a['split_fit'], []
    if critic:
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
    for index, row in enumerate(rows):
        batch_rows = np.array([row])
        x, background = data.batch(batch_rows, True)
        fields = model.sample(x, background, data.hours[batch_rows], 4, 20261123+index)
        samples = base.query_members(data, fields, ids)
        target, mask = data.labels[batch_rows][:, ids], data.mask[batch_rows][:, ids]
        es, crps = base.fair_scores(samples/10, target/10, mask)
        score = flat_gradient(es+.2*crps, model)
        if family == 'diffusion':
            other = flat_gradient(unbiased_mean_mse(samples/10, target/10, mask), model, False)
        else:
            condition = base.critic_features(data, x, ids)
            geometry = torch.as_tensor(np.column_stack((data.a['xy'][ids]/1e6, data.a['heights'][ids]/3000)), dtype=x.dtype, device=x.device)
            background_stations = data.query(background[:, 0], ids)
            other = flat_gradient(-critic((samples-background_stations[:, None])/10, condition, geometry, mask).mean(), model, False)
        base.finite(score)
        base.finite(other)
        records.append(dict(score_norm=float(score.norm()), other_norm=float(other.norm()),
                            cosine=float(F.cosine_similarity(score[None], other[None])[0])))
    result = dict(family=family, source_step=saved['step'], records=records,
                  role='fixed fitting B only; raw gradients before AdamW, not actual parameter-update attribution')
    if family == 'diffusion':
        result['mean_weight'] = min(.1, .2*np.median([v['score_norm'] for v in records])/(np.median([v['other_norm'] for v in records])+1e-12))
    else:
        result['weighted_adversarial_gradient_ratio'] = [.05*r['other_norm']/max(r['score_norm'], 1e-12) for r in records]
    del model, critic
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def vae_audit(data, rows):
    model, _, _, _, _ = load_start('vae')
    model.eval()
    ids = data.a['split_fit']
    result = []
    for row in rows:
        batch_rows = np.array([row])
        x, background = data.batch(batch_rows, True)
        encoded = model.encode(x)
        prior = model.sample_encoded(encoded, background, data.hours[batch_rows], 8, 20261123)
        post_grid = base.posterior_grid(data, batch_rows, ids, background)
        posterior, params = model.posterior_sample(encoded, background, post_grid, data.hours[batch_rows], 8, 20261123)
        mu_q, log_q, mu_p, log_p = params
        mean_latent = model.decode(encoded, background, mu_p[:, None])
        prior_values, posterior_values = base.query_members(data, prior, ids), base.query_members(data, posterior, ids)
        mask, target = data.a['mask'][batch_rows][:, ids], data.a['labels'][batch_rows][:, ids]
        kl = (log_p-log_q+.5*(torch.exp(2*(log_q-log_p))+(mu_q-mu_p).square()*torch.exp(-2*log_p)-1))
        result.append(dict(hour=int(data.hours[row]), valid_stations=int(mask.sum()), raw_KL=float(kl.sum()),
                   mean_KL_per_latent=float(kl.mean()), latent_dimensions=kl.numel(),
                   active_latents_KL_gt_001=int((kl > .01).sum()),
                   prior_rmse=base.metrics(prior_values.mean(1).cpu().numpy(), target, mask)['macro_rmse'],
                   posterior_reconstruction_rmse=base.metrics(posterior_values.mean(1).cpu().numpy(), target, mask)['macro_rmse'],
                   prior_member_std_mm=float(prior_values.std(1)[torch.as_tensor(mask, device='cuda')].mean()),
                   prior_mean_vs_mean_latent_rms_mm=float((prior_values.mean(1)-base.query_members(data, mean_latent, ids)[:, 0]).square()[torch.as_tensor(mask, device='cuda')].mean().sqrt())))
    return dict(records=result, information_condition='Posterior sees fitting B; reconstruction is a training-information diagnostic, not deployment performance')


def prepare(data):
    allowed = base.qualified_training_rows(data.a['mask'], data.rows('train'), data.a['split_fit'])
    rows = allowed[np.rint(np.linspace(0, len(allowed)-1, 8)).astype(int)]
    audits = dict(gan=gradient_audit(data, rows[:4], 'gan'), diffusion=gradient_audit(data, rows[:4], 'diffusion'), vae=vae_audit(data, rows))
    timings, equivalence = {}, {}
    for family in base.FACTORIES:
        model, optimizer, critic, opt_d, saved = load_start(family)
        base.update(model, optimizer, data, rows[:4], family, saved['step']+1, critic, opt_d)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        expected_d = {k: v.detach().clone() for k, v in critic.state_dict().items()} if critic else None
        del model, optimizer, critic, opt_d
        model, optimizer, critic, opt_d, saved = load_start(family)
        controlled_update(model, optimizer, data, rows[:4], family, saved['step']+1, critic, opt_d)
        difference = max(float((expected[k]-v).abs().max()) for k, v in model.state_dict().items())
        if critic:
            difference = max(difference, max(float((expected_d[k]-v).abs().max()) for k, v in critic.state_dict().items()))
        if difference != 0:
            raise ValueError(f'Control training path differs from original: {family}/{difference}')
        equivalence[family] = difference
        elapsed = []
        for intervention in (False, True):
            kwargs = dict(elbo_weight=0. if family == 'vae' and intervention else .1,
                          adversarial_weight=0. if family == 'gan' and intervention else .05,
                          mean_weight=audits['diffusion']['mean_weight'] if family == 'diffusion' and intervention else 0.)
            torch.cuda.synchronize()
            started = time.monotonic()
            controlled_update(model, optimizer, data, rows[:4], family, saved['step']+2, critic, opt_d, **kwargs)
            torch.cuda.synchronize()
            elapsed.append(time.monotonic()-started)
        timings[family] = dict(control_seconds=elapsed[0], intervention_seconds=elapsed[1], slower_seconds=max(elapsed))
        del model, optimizer, critic, opt_d, expected
        torch.cuda.empty_cache()
    updates = min(300, int(.8*2100/(2*sum(v['slower_seconds'] for v in timings.values()))))
    if updates < 100:
        raise ValueError('Benchmark cannot support a meaningful paired test inside the registered budget')
    configuration = dict(updates=updates, effective_batch=4, training_members=4, adaptation_seed=42,
                         lr_start=.00005, lr_end=.00001, weight_decay=.0001, clipping=1.,
                         mean_weight=audits['diffusion']['mean_weight'], budget_seconds=3600,
                         input_cache_sha256=read_json(base.DATA_ROOT/'data_complete.json')['sha256'],
                         source_contract=sha256(SOURCE/'generators.lock.json'),
                         source_checkpoints={f: sha256(SOURCE/f/'last.pt') for f in base.FACTORIES},
                         model_code=sha256(base.PROJECT/'src/gnss_field/models/conditioned_unet.py'),
                         training_code=sha256(__file__), objective_code=sha256(base.PROJECT/'src/gnss_field/objectives/ensemble_mean.py'),
                         original_training_code=sha256(SOURCE_CODE), evaluation_members=16,
                         roles='A input; fitting B training; inner B evaluation; C/D not scored or tuned',
                         original_path_parameter_max_difference=equivalence,
                         paired_interventions=dict(vae='remove late .1 ELBO only', gan='set generator adversarial coefficient to zero only; D updates retained', diffusion='add independent-member mean-MSE U only'))
    write_json(ROOT/'preflight.json', dict(audits=audits, timings=timings, equivalence=equivalence))
    immutable_json(ROOT/'experiment.lock.json', configuration)
    base.commit_budget('PREPARED')
    print('PAIRED_PREFLIGHT', json.dumps(dict(updates=updates, mean_weight=configuration['mean_weight'], timings=timings, equivalence=equivalence)), flush=True)
    return configuration


def check_budget():
    if (ROOT/'PAUSE').exists():
        raise base.EvaluationPaused('Paused at a completed update/hour boundary')
    if base.budget_seconds() >= base.LOCAL_CAP:
        raise base.LocalBudgetExhausted('Cumulative mechanism-validation budget reached')


def train_arm(data, family, arm, config, adaptation_seed=42):
    out = ROOT/f'seed{adaptation_seed}'/family/arm
    out.mkdir(parents=True, exist_ok=True)
    model, optimizer, critic, opt_d, saved = load_start(family)
    source_step = saved['step']
    allowed = base.qualified_training_rows(data.a['mask'], data.rows('train'), data.a['split_fit'])
    rng = np.random.default_rng(20261123+adaptation_seed)
    steps = config['updates']
    schedule = np.concatenate([rng.permutation(allowed) for _ in range(math.ceil(steps*4/len(allowed)))])[:steps*4].reshape(steps, 4)
    contract = sha256(ROOT/'experiment.lock.json')
    history, start = [], 0
    if (out/'last.pt').exists():
        current = torch.load(out/'last.pt', map_location='cuda', weights_only=False)
        if current['contract'] != contract or current['arm'] != arm or current['adaptation_seed'] != adaptation_seed:
            raise ValueError('Adaptation checkpoint contract mismatch')
        np.testing.assert_array_equal(current['schedule'], schedule)
        model.load_state_dict(current['model'])
        optimizer.load_state_dict(current['optimizer'])
        if critic:
            critic.load_state_dict(current['critic'])
            opt_d.load_state_dict(current['critic_optimizer'])
        base.restore_random(current['rng'])
        history, start = current['history'], current['step']
    for group in optimizer.param_groups:
        group['lr'] = config['lr_start']

    def save(step):
        base.atomic_checkpoint(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 critic=critic.state_dict() if critic else None, critic_optimizer=opt_d.state_dict() if critic else None,
                 rng=base.random_state(), schedule=schedule, step=step, source_step=source_step,
                 family=family, arm=arm, adaptation_seed=adaptation_seed, contract=contract, history=history))

    for step in range(start+1, steps+1):
        check_budget()
        lr = config['lr_end']+(config['lr_start']-config['lr_end'])*.5*(1+math.cos(math.pi*(step-1)/max(steps-1, 1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        intervention = arm == 'intervention'
        record = controlled_update(model, optimizer, data, schedule[step-1], family, source_step+step, critic, opt_d,
                   elbo_weight=0. if family == 'vae' and intervention else .1,
                   adversarial_weight=0. if family == 'gan' and intervention else .05,
                   mean_weight=config['mean_weight'] if family == 'diffusion' and intervention else 0.,
                   noise_offset=adaptation_seed*100003)
        history.append(dict(step=step, lr=lr, **record))
        if step % 50 == 0 or step == steps:
            save(step)
            write_json(out/'curve.json', history)
            write_json(ROOT/'status.json', dict(stage='TRAINING', family=family, arm=arm, seed=adaptation_seed, step=step, updated=time.time()))
            base.commit_budget('TRAINING')
            print(json.dumps(dict(family=family, arm=arm, seed=adaptation_seed, **history[-1])), flush=True)
        if (ROOT/'PAUSE').exists() or base.budget_seconds() >= base.LOCAL_CAP:
            save(step)
            check_budget()
    del model, optimizer, critic, opt_d
    torch.cuda.empty_cache()
    return out/'last.pt'


def diagnostic_metrics(samples, truth, mask, xy):
    result = base.distribution_metrics(samples, truth, mask, xy)
    count = mask.sum(0)
    active = count > 0
    clean = np.where(mask, truth, 0.)
    squared = np.where(mask, samples.mean(1)-truth, 0.)**2
    k = samples.shape[1]
    sorted_x = np.sort(samples.astype('f8'), axis=1)
    crps = abs(samples-clean[:, None]).mean(1)-(sorted_x*(2*np.arange(1, k+1)-k-1)[None, :, None]).sum(1)/(k*(k-1))
    result['macro_mse_mm2'] = float((squared.sum(0)[active]/count[active]).mean())
    result['macro_CRPS_mm'] = float(((crps*mask).sum(0)[active]/count[active]).mean())
    # One fixed station per 200km cell checks dependence beyond the local groups.
    _, first = np.unique(np.floor(xy/200000).astype(int), axis=0, return_index=True)
    fixed = np.sort(first)[:16]
    joint, eligible = [], 0
    for t in range(len(samples)):
        valid = fixed[mask[t, fixed]]
        if len(valid) >= 2:
            value, _ = base.fair_scores(torch.as_tensor(samples[t:t+1, :, valid], dtype=torch.float64),
                    torch.as_tensor(truth[t:t+1, valid], dtype=torch.float64), torch.ones(1, len(valid), dtype=torch.bool))
            joint.append(float(value))
            eligible += 1
    result.update(cross_cell_ES_mm=float(np.mean(joint)) if joint else None, cross_cell_hours=eligible,
                  cross_cell_station_columns=fixed.tolist())
    return result


def evaluate_checkpoint(data, family, checkpoint, output, members=16, sample_seed=20261123):
    receipt = output.with_suffix('.json')
    if receipt.exists():
        saved = read_json(receipt)
        if saved['checkpoint_sha256'] != sha256(checkpoint) or saved['members'] != members or saved['sample_seed'] != sample_seed:
            raise ValueError('Evaluation checkpoint changed')
        if saved['artifact_sha256'] != sha256(output.with_suffix('.npz')):
            raise ValueError('Prediction artifact changed')
        return saved
    model = base.make_model(family)
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    fit, held = data.a['split_fit'], data.a['split_inner']
    ids = np.concatenate([fit, held])
    allowed = base.qualified_training_rows(data.a['mask'], data.rows('train'), fit)
    panels = dict(training=allowed[np.rint(np.linspace(0, len(allowed)-1, 24)).astype(int)],
                  withheld_weather=data.rows('inner').reshape(12, -1)[:, [0, 6]].ravel())
    scores, arrays = {}, dict(station_ids=ids)
    for name, rows in panels.items():
        check_budget()
        samples = base.sample_population(model, data, rows, ids, members, sample_seed)
        target, mask = data.a['labels'][np.ix_(rows, ids)], data.a['mask'][np.ix_(rows, ids)]
        for role, columns in [('fitting', np.arange(len(fit))), ('withheld', np.arange(len(fit), len(ids)))]:
            scores[f'{name}_{role}'] = diagnostic_metrics(samples[:, :, columns], target[:, columns], mask[:, columns], data.a['xy'][ids[columns]])
        arrays.update({f'{name}_samples': samples, f'{name}_target': target, f'{name}_mask': mask, f'{name}_hours': data.hours[rows]})
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = output.with_suffix('.npz')
    with artifact.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **arrays)
    os.replace(artifact.with_suffix('.tmp'), artifact)
    result = dict(checkpoint_sha256=sha256(checkpoint), artifact_sha256=sha256(artifact), members=members,
                  sample_seed=sample_seed, scores=scores, source_cache=read_json(base.DATA_ROOT/'data_complete.json')['sha256'])
    write_json(receipt, result)
    del model
    torch.cuda.empty_cache()
    base.commit_budget('ENDPOINT_EVALUATION')
    print('ENDPOINT_EVALUATED', family, str(output.relative_to(ROOT)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base.PRIOR_SPENT = read_json(ROOT/'local_budget.json')['consumed_seconds'] if (ROOT/'local_budget.json').exists() else 0.
    base.CLOCK_START = time.monotonic()
    source_lock = read_json(SOURCE/'generators.lock.json')
    if sha256(SOURCE_CODE) != source_lock['training_code'] or sha256(base.DATA_ROOT/'data.npz') != source_lock['input_cache']:
        raise ValueError('Source training or data cache changed')
    data = base.PilotData()
    config = read_json(ROOT/'experiment.lock.json') if (ROOT/'experiment.lock.json').exists() else prepare(data)
    if config['training_code'] != sha256(__file__) or config['objective_code'] != sha256(base.PROJECT/'src/gnss_field/objectives/ensemble_mean.py'):
        raise ValueError('Immutable mechanism-validation release changed')
    if args.stage == 'prepare':
        return
    results = {}
    # Complete all training first, then compare only registered endpoints.
    for family in base.FACTORIES:
        for arm in ARMS:
            train_arm(data, family, arm, config)
    for family in base.FACTORIES:
        records = dict(source=evaluate_checkpoint(data, family, SOURCE/family/'last.pt', ROOT/'endpoints'/family/'source_k16'))
        for arm in ARMS:
            checkpoint = ROOT/'seed42'/family/arm/'last.pt'
            records[arm] = evaluate_checkpoint(data, family, checkpoint, ROOT/'endpoints'/family/f'{arm}_k16')
        results[family] = records
    write_json(ROOT/'paired_results.json', dict(status='COMPLETE', families=results, contract=sha256(ROOT/'experiment.lock.json')))
    write_json(ROOT/'status.json', dict(stage='COMPLETE', updated=time.time(), families=list(results)))
    base.commit_budget('COMPLETE')
    print('MECHANISM_VALIDATION_COMPLETE', flush=True)


if __name__ == '__main__':
    base.ROOT = ROOT
    with base.single_process():
        try:
            main()
        except (base.EvaluationPaused, base.LocalBudgetExhausted) as error:
            status = 'PAUSED' if isinstance(error, base.EvaluationPaused) else 'BUDGET_EXHAUSTED'
            write_json(ROOT/'status.json', dict(stage=status, reason=str(error), updated=time.time()))
            base.commit_budget(status)
        except Exception as error:
            write_json(ROOT/'status.json', dict(stage='FAILED', error=str(error), updated=time.time()))
            base.commit_budget('FAILED')
            raise
