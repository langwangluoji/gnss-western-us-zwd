"""C2019-only evaluation for a B-selected portable observable model."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import io
import signal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gnss_field.artifacts import read_json, sha256, immutable_json, write_json, atomic_bytes
from gnss_field.models.subspace import SubspaceEDM, station_members
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.models.diffusion import keyed_noise
from gnss_field.evaluation import point_metrics, crps_ensemble
from gnss_field.evaluation.metrics import joint_metrics
from gnss_field.training.observable import acquire_study_lease
from gnss_field.training.release import verify_release


def save_month(path, evaluation_sha256, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())
    write_json(Path(path).with_suffix('.manifest.json'),
        dict(evaluation_sha256=evaluation_sha256, data_sha256=sha256(path)))


def load_month(path, evaluation_sha256):
    path = Path(path)
    meta = path.with_suffix('.manifest.json')
    if not path.exists() or not meta.exists():
        return None  # An interrupted uncommitted month is recomputed.
    record = read_json(meta)
    if record['evaluation_sha256'] != evaluation_sha256 or record['data_sha256'] != sha256(path):
        raise ValueError('Cached month differs from its evaluation lock or recorded data hash')
    return dict(np.load(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=['mean', 'edm'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', help='Optional relocated file, verified against the original selection hash')
    args = parser.parse_args()
    root, run, output = Path(args.bundle), Path(args.run), Path(args.output)
    meta = read_json(root/'manifest.json')
    if meta['role'] != 'C' or meta['year'] != 2019:
        raise PermissionError('This entry only evaluates exposed C2019 development, never D2020')
    release_hash = verify_release(Path(__file__).resolve().parents[3])
    if sha256(root/'data.npz') != meta['data_sha256'] or sha256(root/'stations.csv') != meta['stations_sha256']:
        raise ValueError('Evaluation bundle differs')
    selection = read_json(run/'selection.lock.json')
    checkpoint = Path(args.checkpoint or selection['checkpoint'])
    if sha256(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (state['spec']['stage'] != f'observable_{args.arm}_refine' or
        state['spec']['dependencies']['bundle_manifest'] != meta['training_bundle_manifest_sha256']):
        raise ValueError('Engineering or mismatched foundation checkpoint cannot be evaluated as formal')
    if state['spec']['dependencies']['release_sha256'] != release_hash:
        raise ValueError('Selected checkpoint release differs from evaluation code and dependencies')
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output/'evaluation.lock.json', dict(source=sha256(__file__),
        release_sha256=release_hash,
        selection_sha256=sha256(run/'selection.lock.json'), evaluation_manifest_sha256=sha256(root/'manifest.json'),
        checkpoint_sha256=sha256(checkpoint), members=128, sampling_seed=20261119, D2020=False))
    evaluation_hash = sha256(output/'evaluation.lock.json')
    acquire_study_lease(run.resolve().parent)
    pause = [False]
    def request_pause(*_):
        pause[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT, *([signal.SIGUSR1] if hasattr(signal, 'SIGUSR1') else [])):
        signal.signal(sig, request_pause)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = dict(np.load(root/'data.npz'))
    if len(a['hours']) != 8760 or not (pd.to_datetime(a['hours'], unit='h').year == 2019).all():
        raise ValueError('Full C2019 manifest required')
    tensors = {k: torch.tensor(a[k], device=device) for k in ('condition', 'mean', 'amplitude', 'projection', 'remainder')}
    model = (SubspaceEDM() if args.arm == 'edm' else SubspaceMeanShift()).to(device).eval()
    model.load_state_dict(state['ema'])
    metadata = pd.read_csv(root/'stations.csv')
    mean_rows, crps_rows, group_sums = [], [], {}
    intervals = {v: [[], []] for v in (.8, .9, .95)}
    month = pd.to_datetime(a['hours'], unit='h').strftime('%Y-%m').to_numpy()
    member_buffer, hour_buffer, population_buffer = [], [], []
    loaded_month, loaded_name = None, None
    last_complete_month = None
    def safe_pause():
        if pause[0]:
            write_json(output/'status.json', dict(status='PAUSED', last_complete_month=last_complete_month))
            raise SystemExit(75)
    with torch.no_grad():
        for start in range(0, len(a['hours']), 8):
            safe_pause()
            stop = min(start+8, len(a['hours']))
            hours = a['hours'][start:stop]
            cached = output/f'forecast_{month[start]}.npz'
            if loaded_name != month[start]:
                loaded_month = load_month(cached, evaluation_hash)
                loaded_name = month[start]
                if loaded_month is not None:
                    np.testing.assert_array_equal(loaded_month['station_ids'], a['station_ids'])
            if loaded_month is not None:
                row = np.searchsorted(loaded_month['hours'], hours)
                np.testing.assert_array_equal(loaded_month['hours'][row], hours)
                samples = loaded_month['samples'][row]
            else:
                samples = None
            z = keyed_noise(hours, 128, 20261119, (64,), device=device)
            if samples is None:
                epsilon = keyed_noise(hours, 128, 20262119, (a['remainder'].shape[1],), device=device)
                eta = keyed_noise(hours, 128, 20263119, (905,), device=device)[..., a['station_ids']]
                latent = model.sample_noise(tensors['condition'][start:stop], z)
                samples = station_members(tensors['mean'][start:stop], latent, epsilon, eta,
                    tensors['projection'], tensors['remainder'], tensors['amplitude'][start:stop], meta['tau_surface_mm']).cpu().numpy()
            safe_pause()
            pop, truth = a['valid'][start:stop], a['truth'][start:stop]
            if not np.isfinite(samples.transpose(0, 2, 1)[a['population_F'][start:stop]]).all():
                raise ValueError('Nonfinite forecast cannot be removed from scoring')
            if args.arm == 'mean':
                point = tensors['mean'][start:stop]+(model.net(tensors['condition'][start:stop])@tensors['projection'].T)*tensors['amplitude'][start:stop, None]
                mean_rows.append(point.cpu().numpy())
            else:
                mean_rows.append(samples.mean(1))
            value = np.zeros(pop.shape)
            value[pop] = crps_ensemble(truth[pop], samples.transpose(0, 2, 1)[pop])
            crps_rows.append(value)
            group = joint_metrics(truth, samples, pop, metadata)
            for row in group['groups']:
                sums = group_sums.setdefault(row['group_id'], np.zeros(3))
                sums += [row['es_mm']*row['n_hours'], row['vs_mm']*row['n_hours'], row['n_hours']]
            for level in intervals:
                low, high = np.quantile(samples, [(1-level)/2, (1+level)/2], axis=1)
                intervals[level][0].append((truth >= low) & (truth <= high))
                intervals[level][1].append(high-low)
            member_buffer.append(samples)
            hour_buffer.append(hours)
            population_buffer.append(pop)
            if stop == len(a['hours']) or month[stop] != month[start]:
                safe_pause()
                if loaded_month is None:
                    save_month(output/f'forecast_{month[start]}.npz', evaluation_hash, hours=np.concatenate(hour_buffer),
                        station_ids=a['station_ids'], samples=np.concatenate(member_buffer), population=np.concatenate(population_buffer))
                last_complete_month = month[start]
                member_buffer, hour_buffer, population_buffer = [], [], []
                safe_pause()
    pop = a['valid']
    count = pop.sum(0)
    active = count > 0
    def macro(v):
        return float((np.where(pop, v, 0).sum(0)[active]/count[active]).mean())
    point = np.concatenate(mean_rows)
    result = dict(point=point_metrics(a['truth'], point, pop),
        mean_kind='analytic Gaussian mean' if args.arm == 'mean' else 'finite128 member mean',
        station_macro_CRPS_mm=macro(np.concatenate(crps_rows)),
        group_macro_ES_mm=float(np.mean([v[0]/v[2] for v in group_sums.values()])),
        group_macro_VS_mm=float(np.mean([v[1]/v[2] for v in group_sums.values()])),
        n_groups=len(group_sums), n_group_hours=int(sum(v[2] for v in group_sums.values())),
        intervals={str(round(k*100)): dict(coverage=macro(np.concatenate(v[0])), width_mm=macro(np.concatenate(v[1]))) for k, v in intervals.items()},
        scope='C2019 exposed development, not D2020 retrospective')
    write_json(output/'metrics.json', result)
    write_json(output/'status.json', dict(status='COMPLETE', months=12))


if __name__ == '__main__':
    main()
