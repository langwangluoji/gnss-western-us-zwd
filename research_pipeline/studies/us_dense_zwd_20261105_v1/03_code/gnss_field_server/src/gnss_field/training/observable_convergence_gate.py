"""B-only, independent-noise development gate for one locked EMA candidate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import io
from pathlib import Path
import numpy as np
import torch
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256, atomic_bytes
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.models.subspace import station_members
from gnss_field.models.diffusion import keyed_noise
from gnss_field.training.observable import load_bundle, acquire_study_lease
from gnss_field.training.release import verify_release


def hourly_energy(samples, truth, valid):
    x = samples*valid[:, None]
    y = torch.where(valid, truth, 0.)
    k = x.shape[1]
    distance = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    distance = distance.masked_fill(torch.eye(k, dtype=torch.bool, device=x.device), 0.)
    return (torch.linalg.vector_norm(x-y[:, None], dim=-1).mean(1)
        -distance.sum((1, 2))/(2*k*(k-1)))/valid.sum(1).sqrt()


def paired_week_gate(hours, candidate, reference):
    hours, candidate, reference = map(np.asarray, (hours, candidate, reference))
    if candidate.shape != hours.shape or reference.shape != hours.shape or not np.isfinite(candidate-reference).all():
        raise ValueError('Gate requires aligned finite hourly scores')
    weeks = hours//168
    blocks = np.unique(weeks)
    if len(blocks) < 2:
        raise ValueError('Gate needs multiple whole development weeks')
    differences = candidate-reference
    sums = np.array([differences[weeks == block].sum() for block in blocks])
    counts = np.array([(weeks == block).sum() for block in blocks])
    index = np.random.default_rng(20261122).integers(0, len(blocks), size=(2000, len(blocks)))
    draws = sums[index].sum(1)/counts[index].sum(1)
    interval = np.quantile(draws, [.025, .975])
    return dict(passed=bool(differences.mean() < 0 and interval[1] < 0),
        difference_mm=float(differences.mean()), interval95_mm=interval.tolist(),
        development_weeks=len(blocks), resamples=2000,
        scope='Selected B development interval; independent noise is not independent labels')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root, run = Path(args.bundle), Path(args.run)
    release = root.parent
    release_hash = verify_release(release)
    arrays, meta = load_bundle(root)
    if meta.get('convergence_protocol') != 'mean_convergence_v1':
        raise ValueError('Gate requires the registered mean convergence protocol')
    if read_json(run/'status.json')['status'] != 'COMPLETE':
        raise ValueError('Only a completed EMA selection may enter the B gate')
    selected = read_json(run/'selection.lock.json')
    checkpoint = Path(selected['checkpoint'])
    if sha256(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected candidate differs')
    reference = release/'reference/mean_selected.pt'
    if sha256(reference) != meta['reference_mean_sha256']:
        raise ValueError('Fixed old mean reference differs')
    current = torch.load(checkpoint, map_location='cpu', weights_only=False)
    old = torch.load(reference, map_location='cpu', weights_only=False)
    if (current['spec']['dependencies']['release_sha256'] != release_hash or
        current['spec']['dependencies']['bundle_manifest'] != sha256(root/'manifest.json') or
        current['spec']['stage'] != 'observable_mean_refine' or
        old['spec']['dependencies']['bundle_manifest'] != meta['reference_bundle_manifest_sha256']):
        raise ValueError('Gate foundation/release contract differs')
    immutable_json(run/'B_gate.lock.json', dict(release_sha256=release_hash,
        checkpoint_sha256=sha256(checkpoint), reference_sha256=sha256(reference),
        manifest_sha256=sha256(root/'manifest.json'), seed=20261122, members=64,
        selection='Single B-selected EMA only; raw diagnostic; no second-candidate retries', D2020=False))
    acquire_study_lease(run.parent)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    values = {k: torch.tensor(arrays[k], device=device) for k in
        ('condition', 'mean', 'truth', 'valid', 'projection', 'remainder', 'amplitude')}
    models = {}
    for name, state, version in [('candidate_EMA', current, 'ema'), ('candidate_raw', current, 'model'),
                                  ('old_EMA', old, 'ema'), ('old_raw', old, 'model')]:
        model = SubspaceMeanShift().to(device).eval().requires_grad_(False)
        model.load_state_dict(state[version])
        models[name] = model
    scores = {name: [] for name in [*models, 'zero']}
    indices = arrays['development_indices']
    if len(np.unique(arrays['hours'][indices]//168)) != 12 or len(indices) != 288:
        raise ValueError('Expected fixed 12-week, 288-hour B development population')
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            idx = indices[start:start+8]
            hours = arrays['hours'][idx]
            latent = keyed_noise(hours, 64, 20261122, (64,), device=device)
            remainder = keyed_noise(hours, 64, 20262122, (values['remainder'].shape[1],), device=device)
            error = keyed_noise(hours, 64, 20263122, (905,), device=device)[..., arrays['station_ids']]
            for name in scores:
                z = latent if name == 'zero' else models[name].sample_noise(values['condition'][idx], latent)
                samples = station_members(values['mean'][idx], z, remainder, error, values['projection'],
                    values['remainder'], values['amplitude'][idx], meta['tau_surface_mm'])
                if not torch.isfinite(samples).all():
                    raise ValueError('Nonfinite prediction cannot be removed')
                scores[name].append(hourly_energy(samples, values['truth'][idx], values['valid'][idx]).cpu().numpy())
    scores = {name: np.concatenate(value) for name, value in scores.items()}
    result = paired_week_gate(arrays['hours'][indices], scores['candidate_EMA'], scores['old_EMA'])
    result['passed'] &= bool(scores['candidate_EMA'].mean() < scores['zero'].mean())
    result.update(ES_mm={name: float(value.mean()) for name, value in scores.items()},
        raw_change_diagnostic=paired_week_gate(arrays['hours'][indices], scores['candidate_raw'], scores['old_raw']),
        selected_step=selected['step'], selection_unchanged=True, D2020=False,
        status='PASSED' if result['passed'] else 'SCIENTIFIC_GATE_NOT_PASSED')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, hours=arrays['hours'][indices], **scores)
    atomic_bytes(run/'B_gate_scores.npz', buffer.getvalue())
    write_json(run/'B_gate.json', result)
    print('B_CONVERGENCE_GATE', result, flush=True)


if __name__ == '__main__':
    main()
