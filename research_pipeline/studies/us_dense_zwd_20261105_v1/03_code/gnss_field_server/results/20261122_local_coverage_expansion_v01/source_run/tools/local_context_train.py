"""Three matched ES mean heads; local B development only, no server submission."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

from local_context_data import PROJECT, RUN, DESIGN
from local_context_features import LocalGridHead
from local_probe_diagnostics import save_npz
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256, fingerprint
from gnss_field.contracts import ScienceConfig
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.models.diffusion import keyed_noise
from gnss_field.objectives.scores import energy_score
from gnss_field.evaluation import point_metrics
from gnss_field.training.engine import TrainEngine, StageSpec, atomic_torch
from gnss_field.training.observable import _same
from gnss_field.runtime import configure

ARMS = ('PCA32_R64', 'local99_R64_projection', 'local99_grid')


def observed_loss(samples, truth, valid, fit):
    return energy_score(samples, truth, valid & fit[None])


class ThreeArmMean(nn.Module):
    def __init__(self, arm, arrays, scale):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown architecture arm')
        self.arm, self.scale = arm, float(scale)
        rank = arrays['projection'].shape[1]
        self.head = SubspaceMeanShift(rank=rank) if arm == ARMS[0] else LocalGridHead()
        for name in ('projection', 'operator', 'land', 'query_indices', 'query_weights'):
            self.register_buffer(name, torch.as_tensor(np.array(arrays[name])))

    def forward(self, features, condition, amplitude, gate):
        if self.arm == ARMS[0]:
            delta = self.head.net(condition) @ self.projection.T
        else:
            field = self.head(features)*self.scale
            if self.arm == ARMS[1]:
                coefficients = field[:, self.land] @ self.operator.T
                delta = coefficients @ self.projection.T
            else:
                delta = (field[:, self.query_indices]*self.query_weights).sum(-1)
        return delta*amplitude[:, None]*gate[:, None]


class StudyData:
    def __init__(self, device='cuda'):
        self.device = device
        self.root = RUN/'dataset'
        meta = read_json(self.root/'complete.json')
        for name, digest in meta['files'].items():
            if sha256(self.root/name) != digest:
                raise ValueError(f'Changed dataset: {name}')
        self.a = dict(np.load(self.root/'arrays.npz'))
        self.features = np.load(self.root/'features67.npy', mmap_mode='r')
        self.settings = read_json(self.root/'settings.json')
        self.t = {key: torch.as_tensor(self.a[key], device=device) for key in
            ('condition', 'mean', 'truth', 'valid', 'amplitude', 'gate', 'projection', 'remainder')}
        self.center = torch.tensor(self.a['center'], device=device)
        self.std = torch.tensor(self.a['scale'], device=device)
        self.n = len(self.a['station_ids'])
        self.noise_cache = {}

    def features_for(self, indices):
        local = torch.as_tensor(np.array(self.features[indices]), device=self.device)
        local = (local-self.center)/self.std
        pca = self.t['condition'][indices, None].expand(-1, local.shape[1], -1)
        return torch.cat([local, pca], -1)

    def delta(self, model, indices):
        features = None if model.arm == ARMS[0] else self.features_for(indices)
        return model(features, self.t['condition'][indices], self.t['amplitude'][indices], self.t['gate'][indices])

    def noise(self, indices, members, seed=None):
        b = len(indices)
        if seed is None:
            initial = torch.randn(b, members, self.t['projection'].shape[1], device=self.device)
            remainder = torch.randn(b, members, self.t['remainder'].shape[1], device=self.device)
            error = torch.randn(b, members, 905, device=self.device)[..., self.a['station_ids']]
        else:
            hours = self.a['hours'][indices]
            initial = keyed_noise(hours, members, seed, shape=(self.t['projection'].shape[1],), device=self.device)
            remainder = keyed_noise(hours, members, seed+1000, shape=(self.t['remainder'].shape[1],), device=self.device)
            error = keyed_noise(hours, members, seed+2000, shape=(905,), device=self.device)[..., self.a['station_ids']]
        return (initial @ self.t['projection'].T + remainder @ self.t['remainder'].T +
                self.settings['tau']*error)*self.t['amplitude'][indices, None, None]

    @torch.no_grad()
    def predict(self, model, indices):
        rows = []
        for start in range(0, len(indices), 8):
            ix = indices[start:start+8]
            rows.append((self.t['mean'][ix]+self.delta(model, ix)).cpu().numpy())
        return np.concatenate(rows)

    @torch.no_grad()
    def metrics(self, prediction, indices, fit, members=16, seed=20261117):
        valid = self.a['valid'][indices] & np.asarray(fit)[None]
        point = point_metrics(self.a['truth'][indices], prediction, valid)
        # Common fixed noise pools are independent of the optimization noise stream.
        key = (tuple(indices), members, seed)
        if key not in self.noise_cache:
            rows = [self.noise(indices[s:s+8], members, seed).cpu() for s in range(0, len(indices), 8)]
            self.noise_cache[key] = torch.cat(rows)
        noise = self.noise_cache[key]
        total = 0.
        for start in range(0, len(indices), 8):
            ix = indices[start:start+8]
            samples = torch.tensor(prediction[start:start+8], device=self.device)[:, None]+noise[start:start+8].to(self.device)
            total += float(energy_score(samples, self.t['truth'][ix],
                torch.tensor(valid[start:start+8], device=self.device)))*len(ix)
        return dict(**point, ES_mm=total/len(indices), members=members, seed=seed)


def make_engine(data, arm, fold, directory):
    design = read_json(DESIGN)
    torch.manual_seed(42)
    model = ThreeArmMean(arm, data.a, data.settings['grid_scale']).to(data.device)
    fit = torch.tensor(data.a['folds'] != fold, device=data.device)
    core, inner = data.a['core'], data.a['inner']
    if (data.a['valid'][core][:, fit.cpu().numpy()].sum(-1) < 4).any():
        raise ValueError('The locked core has an hour with fewer than four fitted targets')
    science = ScienceConfig(rank=64, condition_dim=32, learning_rate=design['learning_rate'],
        refinement_lr=design['learning_rate'], ema=design['EMA'], warmup=100,
        startup_evaluation=0, evaluate_every=100, log_every=100,
        minimum_refinement_steps=300, patience=3)
    dependencies = dict(design=sha256(DESIGN), data=sha256(data.root/'complete.json'),
        runner=sha256(__file__), features=sha256(Path(__file__).with_name('local_context_features.py')),
        engine=sha256(PROJECT/'src/gnss_field/training/engine.py'), arm=arm, fold=fold,
        training='B2018 fitted groups/core hours; fixed historical base has B exposure',
        objective='observation ES K4, shared independent frozen Gaussian noise',
        local_output_scale='training Z0 std times frozen weather amplitude', D2020=False)
    spec = StageSpec('local_context_refine', 1000, 1000, 8, 8, 42, dependencies)
    engine = TrainEngine(model, directory, data.a['hours'][core], spec, science,
        pause_paths=[RUN/'PAUSE'], max_seconds=1800)

    def objective(hours, current, step):
        ix = np.searchsorted(data.a['hours'], hours)
        prediction = data.t['mean'][ix]+data.delta(current, ix)
        samples = prediction[:, None]+data.noise(ix, 4)
        loss = observed_loss(samples, data.t['truth'][ix], data.t['valid'][ix], fit)
        return dict(loss=loss, observation_ES_mm=loss)

    def evaluator(current, step):
        prediction = data.predict(current, inner)
        metrics = data.metrics(prediction, inner, fit.cpu().numpy())
        return {'joint': dict(value=metrics['ES_mm'], rmse_mm=metrics['station_macro_rmse_mm'],
            mae_mm=metrics['station_macro_mae_mm'], bias_mm=metrics['station_macro_bias_mm'],
            population='B2018 inner hours and fitted spatial groups; development diagnostics')}

    return engine, objective, evaluator


def verify_resume(data):
    root = RUN/'engineering_resume'/sha256(__file__)[:12]
    if (root/'verification.json').exists():
        return read_json(root/'verification.json')
    start = time.monotonic()
    continuous, objective, _ = make_engine(data, ARMS[1], 0, root/'continuous')
    continuous.run(objective, stop_after=200)
    expected = continuous.state()
    split, objective, _ = make_engine(data, ARMS[1], 0, root/'split')
    split.run(objective, stop_after=100)
    del split
    resumed, objective, _ = make_engine(data, ARMS[1], 0, root/'split')
    resumed.run(objective, stop_after=200)
    actual = resumed.state()
    for key in ('model', 'ema', 'optimizers', 'cursor', 'rng'):
        _same(expected[key], actual[key])
    result = dict(passed=True, continuous=200, resumed=[100,100], bitwise=True,
        elapsed_seconds=time.monotonic()-start, separate_engineering_updates=400,
        scientific_update_cap=6000, scientific_checkpoints_untouched=True, runner=sha256(__file__))
    write_json(root/'verification.json', result)
    print('REAL_RESUME_PASS', result, flush=True)
    return result


def initialize_zero(engine, evaluator):
    """Recover an interrupted initial evaluation without relabeling trained EMA."""
    import copy
    directory = engine.directory
    initial, metrics = directory/'initial.pt', directory/'initial_metrics.json'
    if not initial.exists() and engine.step != 0:
        raise ValueError('Trained checkpoint is missing the original zero candidate')
    if not metrics.exists():
        frozen = copy.deepcopy(engine.ema)
        if initial.exists():
            state = torch.load(initial, map_location='cpu', weights_only=False)
            if state['signature'] != engine.signature or state['step'] != 0:
                raise ValueError('Initial candidate contract differs')
            frozen.load_state_dict(state['ema'])
        write_json(metrics, evaluator(frozen, 0))
    zero = read_json(metrics)['joint']
    if engine.step == 0:
        engine.products = {'joint': [dict(**zero, step=0, checkpoint=str(initial.resolve()))]}
    if not initial.exists():
        atomic_torch(initial, engine.state())


def train_one(data, arm, fold):
    directory = RUN/f'fold_{fold}'/arm
    engine, objective, evaluator = make_engine(data, arm, fold, directory)
    if engine.step == 0:
        assert np.array_equal(data.predict(engine.model, data.a['inner']), data.a['mean'][data.a['inner']])
    initialize_zero(engine, evaluator)
    result = engine.run(objective, evaluator=evaluator)
    if result['status'] != 'COMPLETE':
        raise RuntimeError(f'{arm} fold{fold}: {result["status"]}; resume retained')
    zero = read_json(directory/'initial_metrics.json')['joint']
    candidates = [dict(**zero, step=0, checkpoint=str(directory/'initial.pt'))]+engine.products.get('joint', [])
    selected = min(candidates, key=lambda row:(row['value'], row['step']))
    selection = dict(selected=selected, sha256=sha256(selected['checkpoint']),
        rule='B inner ES K16, unique EMA, zero retained; outer not used')
    immutable_json(directory/'selection.lock.json', selection)
    engine.ema.load_state_dict(torch.load(selected['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    # Lock precedes any use of outer labels for candidate evaluation.
    outer, inner = data.a['outer'], data.a['inner']
    prediction = data.predict(engine.ema, outer)
    held = data.a['folds'] == fold
    metrics = data.metrics(prediction, outer, held, members=64, seed=20261118)
    inner_check = data.metrics(data.predict(engine.ema, inner), inner, ~held, members=64, seed=20261118)
    save_npz(directory/'outer_predictions.npz', hours=data.a['hours'][outer], mean=prediction,
        truth=data.a['truth'][outer], population=data.a['valid'][outer] & held[None])
    result = dict(arm=arm, fold=fold, selection=selection, outer_B=metrics,
        inner_K64_independent_check=inner_check, stopped_step=engine.step)
    write_json(directory/'results.json', result)
    print('ARM_COMPLETE', arm, fold, metrics['station_macro_rmse_mm'], flush=True)
    del engine
    torch.cuda.empty_cache()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineering-only', action='store_true')
    args = parser.parse_args()
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    data = StudyData()
    verify_resume(data)
    if args.engineering_only:
        return
    immutable_json(RUN/'training.lock.json', dict(design=sha256(DESIGN), runner=sha256(__file__),
        dataset=sha256(data.root/'complete.json'), maximum_scientific_updates=6000,
        separate_engineering_updates=400, seed=42, server_jobs=False, C2019=False, D2020=False))
    started = time.monotonic()
    rows = []
    for fold in range(2):
        for arm in ARMS:
            path = RUN/f'fold_{fold}'/arm/'results.json'
            rows.append(read_json(path) if path.exists() else train_one(data, arm, fold))
    write_json(RUN/'training_complete.json', dict(status='COMPLETE', models=rows,
        elapsed_seconds=time.monotonic()-started, scientific_updates=sum(r['stopped_step'] for r in rows),
        C2019=False, D2020=False, server_jobs=False))
    print('THREE_ARM_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
