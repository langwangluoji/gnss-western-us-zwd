"""Executable stage graph for the registered observational field experiment.

All targets enter through role-restricted stores. Estimated coefficients initialize
the functional heads. Final refinement uses the published grid observation map.
"""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import io
import math

import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import (atomic_bytes, fingerprint, immutable_json, model_digest,
                                  read_json, package_digest, sha256, write_json)
from gnss_field.data import ContextBuilder, InputScenario, PackedStore
from gnss_field.data.manifests import build_training_manifest, spatial_folds, stratified_hours
from gnss_field.geometry import (basis_observation_matrix, coarsen_100km, curvature_loss,
                                 curvature_matrix, query_grid, surface_query)
from gnss_field.models.basis import SpatialBasis
from gnss_field.models.mean import MeanField
from gnss_field.models.pixel import ResidualField, gaussian_losses
from gnss_field.objectives import (CoefficientTransform, edm_loss, energy_score, fit_latent,
                                   gaussian_nll, gradient_penalty, masked_mse, observability)
from .engine import PauseRequested, StageSpec, TrainEngine, atomic_torch
from .fields import MeanCache, predict_patches
from .prediction import (evaluate_predictor, functional_predictor, mean_predictor,
                         pixel_predictor, safe_query_geometry)
from .system import FrozenDecoder, FunctionalModel


class ScientificGateFailed(RuntimeError):
    """A registered scientific prerequisite has failed."""


def _years(hours, years):
    hours = np.asarray(hours, dtype='i8')
    return hours[np.isin(pd.to_datetime(hours, unit='h').year, years)]


def _tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _arrays(directory):
    records = []
    for path in sorted(Path(directory).glob('forecast_*.npz')):
        with np.load(path) as z:
            records.append({k: z[k] for k in z.files})
    if not records:
        raise FileNotFoundError(f'No saved station forecasts in {directory}')
    return {k: (records[0][k] if k == 'station_ids' else np.concatenate([r[k] for r in records]))
            for k in records[0]}


class TrainingPipeline:
    ALIASES = ('mean_base', 'mean_all', 'mean_cache', 'basis_cv', 'representation_gate',
               'basis_final', 'condition', 'search', 'promote', 'full2019_rerank',
               'stability', 'direction_gate', 'ablation', 'replicate', 'product_lock',
               'evaluate2020', 'export')

    def __init__(self, science, runtime, directory, seed=42, *, legacy_source=None,
                 parent_directory=None):
        from gnss_field.runtime import configure
        configure(runtime, int(seed))
        self.science, self.runtime, self.seed = science, runtime, int(seed)
        self.directory = Path(directory).resolve()
        self.root = self.directory/f'seed_{self.seed}'
        self.root.mkdir(parents=True, exist_ok=True)
        self.parent_directory = Path(parent_directory).resolve() if parent_directory else None
        self.legacy_source = legacy_source
        self._stores = {}
        self._manifest = None
        self._cache = None
        self._decoder = None
        self._condition_months = {}
        self._source_digest = package_digest()
        self.scenario = InputScenario()
        self.cache_root = Path(self.runtime.cache_root).expanduser().resolve()
        immutable_json(self.root/'cache_binding.json', dict(cache_root=str(self.cache_root),
                       use_history=self.science.use_history, scenario=self.scenario.identity,
                       schema='runtime_hot_cache_v1'))

    def run_task(self, alias, *, arm=None, fold=None, variant=None, seed=None):
        if alias not in self.ALIASES:
            raise ValueError(f'Unregistered stage {alias}')
        self._pause()
        if alias == 'basis_cv':
            return self.basis_cv(fold)
        if alias == 'search':
            return self.search(arm)
        if alias == 'ablation':
            return self.ablation(variant)
        if alias == 'replicate':
            return self.replicate(seed)
        return getattr(self, alias)()

    def _pause(self):
        paths = (self.directory, self.root) + ((self.parent_directory,) if self.parent_directory else ())
        if any((p/'PAUSE').exists() for p in paths):
            raise PauseRequested('Campaign pause requested before stage update')

    def _store(self, purpose='train'):
        if purpose == 'retrospective':
            raise PermissionError('Construct retrospective store only after validating product lock')
        if purpose not in self._stores:
            self._stores[purpose] = PackedStore(self.runtime.data_root, purpose=purpose)
        return self._stores[purpose]

    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = build_training_manifest(self._store(), self.directory/'training_manifest.npz')
        return self._manifest

    @property
    def scale(self):
        return self._store().stats['z0_std']

    def _builder(self, store=None):
        return ContextBuilder(store or self._store(), use_history=self.science.use_history)

    def _hours(self, year, store=None):
        store = store or self._store('context')
        return _years(np.asarray(store.hours)[np.asarray(store.hours) >= store.manifest['start']+18], [year])

    def _model(self, arm, *, frozen=True):
        torch.manual_seed(self.seed)
        if arm in ('mean_base', 'mean_all'):
            model = MeanField()
        elif arm in ('m2_f', 'm2_j'):
            model = ResidualField()
        else:
            model = FunctionalModel(arm, self.science.rank, self.science.condition_dim,
                                    frozen_encoder=frozen)
        if not self.science.geometry_bias:
            from gnss_field.models.legacy_core import GeometryAttention
            for module in model.modules():
                if isinstance(module, GeometryAttention):
                    for parameter in module.geometry.parameters():
                        parameter.data.zero_()
                        parameter.requires_grad_(False)
        return model.to(self.runtime.device)

    def _engine(self, model, path, hours, stage, budget, horizon, effective, dependencies,
                science=None, **kwargs):
        self._pause()
        science = science or self.science
        store = self._store()
        dependencies = dict(dependencies, training_manifest=self.manifest.contract,
                            training_data=store.manifest['training_hash'], grid=store.grid_id,
                            release_sha256=self._source_digest,
                            scenario=self.scenario.identity, architecture=fingerprint(
                                {k: list(v.shape) for k, v in model.state_dict().items()}))
        spec = StageSpec(stage, budget, horizon, effective, self.runtime.micro_hours,
                         self.seed, dependencies)
        def batch_telemetry(batch):
            pretrain = stage.endswith('_pretrain')
            masks = None if pretrain else self.manifest.masks(batch)
            if stage == 'basis' and dependencies.get('fit_station_mask') is not None:
                masks = masks & dependencies['fit_station_mask']
            functional = stage.split('_')[0] in ('det', 'gaussian', 'edm', 'gan')
            nfe = 2*science.sampler_steps-1 if stage == 'edm_refine' else 1
            return dict(supervised_station_hours=0 if pretrain else int(masks.sum()),
                        direct_observation_supervision=not pretrain,
                        head_nfe_per_member=nfe if functional else None,
                        head_network_invocations=math.ceil(len(batch)/self.runtime.micro_hours)*nfe if functional else None,
                        network_count_scope='primary objective functional head only; adversarial updates/critic/encoder/patch networks excluded')
        return TrainEngine(model, path, hours, spec, science, batch_telemetry=batch_telemetry,
                           max_seconds=self.runtime.wall_hours*3600,
                           pause_paths=[self.directory/'PAUSE', self.root/'PAUSE',
                                        self.parent_directory/'PAUSE' if self.parent_directory else self.directory/'PAUSE'],
                           **kwargs)

    def _run(self, engine, objective, **kwargs):
        result = engine.run(objective, **kwargs)
        if result['status'] != 'COMPLETE':
            raise PauseRequested(f"Stage {engine.spec.stage} saved at step {engine.step}")
        return engine.directory/'last.pt'

    def _labels(self, hours, station_mask=None):
        store = self._store()
        ids = store.ids('B')
        valid = self.manifest.masks(hours).copy()
        if station_mask is not None:
            valid &= station_mask
        truth = _tensor(store.targets('B', hours), self.runtime.device)
        xy, heights = safe_query_geometry(store, ids)
        return truth, _tensor(valid, self.runtime.device, torch.bool), xy, heights

    def _mean_path(self, arm='mean_base'):
        meta = read_json(self.root/arm/'frozen.json')
        path = Path(meta['checkpoint'])
        if sha256(path) != meta['checkpoint_sha256']:
            raise ValueError('Frozen M0 checkpoint hash changed')
        return path

    def _mean_model(self, arm='mean_base'):
        model = self._model(arm)
        state = _load(self._mean_path(arm))
        model.load_state_dict(state['ema'])
        return model.eval().requires_grad_(False)

    def _mean_cache(self):
        if self._cache is None:
            self._cache = MeanCache(self._store('context'), self.cache_root/'mean',
                                    sha256(self._mean_path()), self.scenario.identity,
                                    context_policy='four_frame' if self.science.use_history else 'current_only')
        return self._cache

    def _context(self, hours):
        ctx = self._builder().build_context(hours, self.scenario, device=self.runtime.device)
        ctx.mean = self._mean_cache().get(hours, device=self.runtime.device)
        return ctx

    def _decoder_bundle(self):
        path = self.root/'basis_final'/'decoder.pt'
        bundle = _load(path)
        if bundle['grid'] != self._store().grid_id or bundle['mean_sha256'] != sha256(self._mean_path()):
            raise ValueError('Basis foundation hash differs')
        return bundle

    def _get_decoder(self):
        if self._decoder is None:
            bundle = self._decoder_bundle()
            self._decoder = FrozenDecoder(bundle['phi'].to(self.runtime.device), bundle['shape'],
                                           CoefficientTransform.from_state(bundle['transform']).to(self.runtime.device),
                                           self.scale)
        return self._decoder

    def _coefficients(self, hours):
        bundle = self._decoder_bundle()
        idx = np.searchsorted(bundle['hours'], hours)
        if (idx >= len(bundle['hours'])).any() or not np.array_equal(bundle['hours'][idx], hours):
            raise ValueError('Estimated training coefficient unavailable for requested hours')
        return self._get_decoder().transform.encode(bundle['coefficients'][idx].to(self.runtime.device))

    def _predictor(self, arm, checkpoint, store=None):
        store = store or self._store('development')
        model = self._model(arm)
        model.load_state_dict(_load(checkpoint)['ema'])
        model.eval().requires_grad_(False)
        if arm.startswith('mean_'):
            return mean_predictor(model, store, self.runtime)
        if arm.startswith('m2_'):
            return pixel_predictor(model, store, self.runtime, self._mean_cache(), self.science, self._builder(store))
        return functional_predictor(model, store, self.runtime, self._mean_cache(),
                                    self._get_decoder(), self.science,
                                    condition_provider=self._condition_provider(model))

    def _evaluate(self, arm, checkpoint, output, *, full=False, members=None, store=None, role='C'):
        store = store or self._store('development')
        hours = self._hours(2019 if role == 'C' else 2020, store)
        if not full:
            hours = stratified_hours(hours, 2019)
        return evaluate_predictor(store, hours, self._predictor(arm, checkpoint, store), output,
                                  members=members or self.science.screen_members, role=role,
                                  scenario=self.scenario, pause_check=self._pause, full_year=full,
                                  checkpoint_hash=sha256(checkpoint), use_history=self.science.use_history)

    def _screen(self, arm, directory):
        def evaluator(model, step):
            store = self._store('development')
            if arm.startswith('mean_'):
                predictor = mean_predictor(model, store, self.runtime)
            elif arm.startswith('m2_'):
                predictor = pixel_predictor(model, store, self.runtime, self._mean_cache(), self.science, self._builder(store))
            else:
                predictor = functional_predictor(model, store, self.runtime, self._mean_cache(),
                                                  self._get_decoder(), self.science,
                                                  condition_provider=self._condition_provider(model))
            result = evaluate_predictor(store, stratified_hours(self._hours(2019, store), 2019), predictor,
                    Path(directory)/f'eval_{step:06d}', members=self.science.screen_members,
                    scenario=self.scenario, pause_check=self._pause, checkpoint_hash=model_digest(model),
                    use_history=self.science.use_history)
            products = ('point',) if arm in ('mean_base', 'mean_all', 'det') else (
                       ('point', 'marginal') if arm.startswith('m2_') else ('point', 'joint'))
            return {p: result['values'][p] for p in products}
        return evaluator

    def _train_mean(self, arm):
        years = (2016, 2017) if arm == 'mean_base' else (2016, 2017, 2018)
        hours = self.manifest.select(years)
        weights = _tensor(self.manifest.weights(hours), self.runtime.device)
        model, directory = self._model(arm), self.root/arm
        builder, store = self._builder(), self._store()

        def objective(selected, current, step):
            context = builder.build_context(selected, self.scenario, device=self.runtime.device)
            field, _ = predict_patches(current, context, store, chunk=self.runtime.patch_chunk)
            truth, valid, xy, heights = self._labels(selected)
            point = surface_query(field, xy, heights).squeeze(1)
            mse = masked_mse(point, truth, valid, self.scale, weights)
            curvature = curvature_loss(field, context.maps[:, 3].bool(), self.scale)
            return dict(loss=mse+self.science.curvature_weight*curvature, mse=mse, curvature=curvature)

        budget = self.science.budgets['mean' if arm == 'mean_base' else 'mean_all']
        engine = self._engine(model, directory, hours, arm, budget, budget, 8, {'years': years})
        self._run(engine, objective, evaluator=self._screen(arm, directory))
        candidates = engine.products.get('point', [])
        if not candidates:
            raise ScientificGateFailed('Mean stage has no post-startup selection checkpoint')
        rows = []
        for entry in candidates:
            result = self._evaluate(arm, entry['checkpoint'], directory/'full2019'/str(entry['step']), full=True)
            rows.append(dict(entry, value=result['values']['point']['value']))
        chosen = min(rows, key=lambda r: (r['value'], r['step']))
        frozen = dict(chosen, checkpoint_sha256=sha256(chosen['checkpoint']), years=years,
                      statistics_years=[2016, 2017], full_year=2019)
        immutable_json(directory/'frozen.json', frozen)
        return {'checkpoint': Path(chosen['checkpoint']), 'frozen': directory/'frozen.json'}

    def mean_base(self):
        return self._train_mean('mean_base')

    def mean_all(self):
        return self._train_mean('mean_all')

    def mean_cache(self):
        hours = np.r_[self._hours(2018), self._hours(2019)]
        cache = self._mean_cache()
        cache.build(self._mean_model(), self._builder(), hours, device=self.runtime.device,
                    chunk=self.runtime.patch_chunk, pause_check=self._pause, scenario=self.scenario)
        return {'contract': cache.root/'contract.json'}

    def _basis_inputs(self, model):
        store = self._store()
        xyh = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel(), store.grid['height'].ravel()]
        return model(_tensor(xyh, self.runtime.device),
                     _tensor(store.grid['region'].ravel(), self.runtime.device, torch.bool))

    def _basis_fit(self, hours, phi, station_mask=None):
        truth, valid, xy, heights = self._labels(hours, station_mask)
        context = self._context(hours)
        base = surface_query(context.mean, xy, heights).squeeze(1)
        h = basis_observation_matrix(phi, self._store().shape, xy, heights)
        coefficients, losses = [], []
        for i in range(len(hours)):
            mask = valid[i]
            curvature = curvature_matrix(phi, self._store().shape, context.maps[i, 3].bool())
            residual = (truth[i, mask]-base[i, mask])/self.scale
            c, _ = fit_latent(h[mask], residual, curvature, penalty=self.science.coefficient_penalty,
                              curvature_weight=self.science.curvature_weight)
            c = c.float()
            loss = (h[mask]@c-residual).square().mean()+self.science.coefficient_penalty*c.square().mean()
            if len(curvature):
                loss = loss+self.science.curvature_weight*(curvature@c).square().mean()
            coefficients.append(c)
            losses.append(loss)
        return torch.stack(coefficients), torch.stack(losses).mean()

    def _train_basis(self, directory, station_mask, budget):
        hours = self.manifest.select([2018], self.science.rank+8, station_mask)
        torch.manual_seed(self.seed)
        model = SpatialBasis(self.science.rank, self.science.basis_width).to(self.runtime.device)
        engine = self._engine(model, directory, hours, 'basis', budget, budget, 16,
                              {'mean': sha256(self._mean_path()), 'fit_station_mask': station_mask})

        def objective(selected, current, step):
            _, loss = self._basis_fit(selected, self._basis_inputs(current), station_mask)
            return {'loss': loss}

        self._run(engine, objective)
        return engine

    def basis_cv(self, fold):
        if fold not in (0, 1, 2):
            raise ValueError('Basis CV fold must be 0, 1 or 2')
        store, directory = self._store(), self.root/f'basis_cv_{fold}'
        assignment = spatial_folds(store, self.science.study_seed)
        fit = assignment != fold
        engine = self._train_basis(directory, fit, self.science.budgets['basis_cv'])
        phi = self._basis_inputs(engine.ema).detach()
        hours = stratified_hours(self.manifest.select([2018], self.science.rank+8, fit), 2018)
        if not len(hours):
            raise ScientificGateFailed('No qualified stratified basis diagnostic hours')
        ids = store.ids('B')
        xy, heights = safe_query_geometry(store, ids)
        h = basis_observation_matrix(phi, store.shape, xy, heights)
        ranks, truth_rows, model_rows, base_rows, coarse_rows, valid_rows = [], [], [], [], [], []
        for hour in hours:
            self._pause()
            with torch.no_grad():
                c, _ = self._basis_fit([hour], phi, fit)
                ctx = self._context([hour])
                base = surface_query(ctx.mean, xy, heights).squeeze(1)
                field = ctx.mean+self.scale*(phi@c[0]).reshape(1, 1, *store.shape)
                truth, valid, _, _ = self._labels([hour])
                rank_mask = self.manifest.masks([hour])[0] & fit
                ranks.append(observability(h[_tensor(rank_mask, self.runtime.device, torch.bool)])['rank'])
                truth_rows.append(truth.cpu().numpy()[0])
                model_rows.append(surface_query(field, xy, heights).cpu().numpy()[0, 0])
                base_rows.append(base.cpu().numpy()[0])
                coarse_rows.append(surface_query(coarsen_100km(field), xy, heights).cpu().numpy()[0, 0])
                valid_rows.append(valid.cpu().numpy()[0] & ~fit)
        from gnss_field.evaluation import evaluate_arrays
        t, p, b, c, valid = map(np.asarray, (truth_rows, model_rows, base_rows, coarse_rows, valid_rows))
        from .prediction import safe_fine_sites
        kwargs = dict(metadata=store.stations.iloc[ids].reset_index(drop=True), safe_T=safe_fine_sites(store, ids))
        metrics = evaluate_arrays(hours, ids, t, p, valid, valid, coarsened_prediction=c, **kwargs)
        baseline = evaluate_arrays(hours, ids, t, b, valid, valid, coarsened_prediction=b, **kwargs)
        result = dict(fold=fold, hours=len(hours), ranks=ranks,
                      observable_fraction=float(np.mean(np.asarray(ranks) >= self.science.rank/2)),
                      metrics=metrics, baseline=baseline, fit_groups=np.unique(assignment[fit]),
                      interpretation='2018 representation-internal held B groups; historical B used by M0')
        write_json(directory/'diagnostics.json', result)
        return {'diagnostics': directory/'diagnostics.json', 'checkpoint': directory/'last.pt'}

    def representation_gate(self):
        diagnostics = [read_json(self.root/f'basis_cv_{i}'/'diagnostics.json') for i in range(3)]
        decisions = []
        for row in diagnostics:
            m, b = row['metrics'], row['baseline']
            fine, inc = m['fine'], m['fine']['model_increments']['0-50']
            b_inc = b['fine']['model_increments']['0-50']
            numbers = (fine['model']['station_macro_rmse_mm'], b['fine']['model']['station_macro_rmse_mm'],
                       fine['coarsened']['station_macro_rmse_mm'], inc['group_macro_rmse_mm'],
                       b_inc['group_macro_rmse_mm'], fine['coarsened_increments']['0-50']['group_macro_rmse_mm'])
            passed = all(x is not None and math.isfinite(x) for x in numbers)
            passed = passed and numbers[0] < min(numbers[1], numbers[2]) and numbers[3] < min(numbers[4], numbers[5])
            decisions.append(bool(passed and row['observable_fraction'] >= .95))
        path = self.root/'representation_gate.json'
        write_json(path, dict(passed=all(decisions), folds=decisions, rank=self.science.rank))
        if not all(decisions):
            raise ScientificGateFailed('Representation rank/held-group direction gate failed')
        return {'gate': path}

    def basis_final(self):
        gate = read_json(self.root/'representation_gate.json')
        if not gate['passed']:
            raise ScientificGateFailed('Final basis requires passed representation gate')
        directory, store = self.root/'basis_final', self._store()
        engine = self._train_basis(directory, np.ones(len(store.ids('B')), bool), self.science.budgets['basis'])
        phi = self._basis_inputs(engine.ema).detach()
        hours = self.manifest.select([2018], self.science.rank+8)
        coefficients = []
        for hour in hours:
            self._pause()
            with torch.no_grad():
                coefficients.append(self._basis_fit([hour], phi)[0][0].cpu())
        coefficients = torch.stack(coefficients)
        transform = CoefficientTransform.fit(coefficients)
        xy, heights = safe_query_geometry(store, store.ids('B'))
        atomic_torch(directory/'decoder.pt', dict(phi=phi.cpu(), shape=store.shape,
            transform=transform.state_dict(), coefficients=coefficients, hours=hours,
            observation_matrix=basis_observation_matrix(phi, store.shape, xy, heights).cpu(),
            neural_weights=engine.ema.state_dict(), grid=store.grid_id,
            mean_sha256=sha256(self._mean_path()), s0=self.scale, seed=self.seed,
            source='qualified_2018_B_regularized_training_inverse',
            checkpoint_sha256=sha256(directory/'last.pt')))
        return {'decoder': directory/'decoder.pt', 'checkpoint': directory/'last.pt'}

    def _observation_loss(self, selected, model, step):
        context = self._context(selected)
        h = model.condition(context, self.scale)
        decoder = self._get_decoder()
        truth, valid, xy, heights = self._labels(selected)
        base = surface_query(context.mean, xy, heights).squeeze(1)
        coefficients = model.coefficients(h, context.hours, self.science.train_members,
                                           self.science.sampling_seed+step, science=self.science)
        station = decoder.station(coefficients, base, xy, heights)
        if model.arm == 'det':
            return masked_mse(station[:, 0], truth, valid, self.scale)
        return energy_score(station, truth, valid, self._store().stats['surface_std'])

    def _pretrain_loss(self, selected, model, step):
        context = self._context(selected)
        h = model.condition(context, self.scale)
        target = self._coefficients(selected)
        if model.arm == 'det':
            return .5*(model.head(h)-target).square().mean()
        if model.arm == 'gaussian':
            mean, log_std = model.head(h)
            return gaussian_nll(mean, log_std, target)
        if model.arm == 'edm':
            return edm_loss(model.head, target, h, p_mean=self.science.p_mean,
                            p_std=self.science.p_std, sigma_data=self.science.sigma_data)
        noise = torch.randn_like(target)
        return -model.critic(model.head(noise, h), h).mean()

    def _foundation_dependencies(self):
        return {'mean': sha256(self._mean_path()), 'basis': sha256(self.root/'basis_final'/'decoder.pt')}

    def _fork(self, engine, source):
        """Extend a capped stage after checking every non-budget contract field."""
        source = Path(source)
        checkpoint = _load(source)
        origin, target = copy.deepcopy(checkpoint['spec']), asdict(engine.spec)
        origin_budget, target_budget = origin.pop('budget'), target.pop('budget')
        if origin != target or not 0 < origin_budget <= target_budget:
            raise ValueError('Promotion stage/data/seed/architecture/horizon contract differs')
        contract = read_json(source.parent/'contract.json')
        if fingerprint(contract['science']) != fingerprint(asdict(engine.science)):
            raise ValueError('Promotion hyperparameters changed')
        if checkpoint['signature'] != contract['signature'] or checkpoint['step'] > target_budget:
            raise ValueError('Promotion source checkpoint signature or step differs')
        if checkpoint['phase'] != 'COMPLETE':
            raise ValueError('Promotion source has an unfinished capped stage')
        receipt = engine.directory/'fork.json'
        immutable_json(receipt, dict(source=str(source.resolve()), sha256=sha256(source),
                                     source_signature=checkpoint['signature'], target_signature=engine.signature,
                                     source_budget=origin_budget, target_budget=target_budget))
        if not (engine.directory/'last.pt').exists():
            engine._load(source, promotion=True)
            engine.phase = 'RUNNING'
            engine.save()

    def _shared_encoder(self, model):
        bundle = _load(self.root/'condition'/'frozen.pt')
        model.encoder.load_state_dict(bundle['encoder'])
        if model_digest(model.encoder) != bundle['encoder_sha256']:
            raise ValueError('Shared encoder hash differs')
        model.encoder.eval().requires_grad_(False)
        return bundle['encoder_sha256']

    def _functional_run(self, arm, directory, params, *, fraction=1., source=None,
                        shared=True, joint_encoder=False):
        directory = Path(directory)
        science = replace(self.science, learning_rate=params['pretrain_lr'],
                          refinement_lr=params['refinement_lr'])
        model = self._model(arm, frozen=shared)
        dependencies = self._foundation_dependencies()
        if shared:
            dependencies['encoder'] = self._shared_encoder(model)
        hours = self._decoder_bundle()['hours']
        prekey = 'condition_pretrain' if not shared else f'{arm}_pretrain'
        refinekey = 'condition_refine' if not shared else f'{arm}_refine'
        prebudget, refinebudget = science.budgets[prekey], science.budgets[refinekey]
        precap = max(1, int(prebudget*fraction))
        critic_opt = None
        parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('critic.')]
        optimizer = torch.optim.AdamW(parameters, lr=science.learning_rate, weight_decay=science.weight_decay)
        if arm == 'gan':
            critic_opt = torch.optim.Adam(model.critic.parameters(), lr=science.learning_rate, betas=(0., .9))
            model.critic.requires_grad_(False)
        pre_engine = self._engine(model, directory/'pretrain', hours, f'{arm}_pretrain',
                       precap, prebudget, 32, dependencies, science=science, optimizers=[optimizer],
                       clip_groups=[parameters], auxiliary_optimizers=[critic_opt] if critic_opt else [])
        if source is not None:
            if not _load(source)['spec']['stage'].endswith('_pretrain'):
                raise ValueError('Functional promotion requires the saved pretraining source')
            self._fork(pre_engine, source)

        def objective(selected, current, step):
            return {'loss': self._pretrain_loss(selected, current, step)}

        def critic_update(selected, current, step):
            current.critic.requires_grad_(True)
            for _ in range(5):
                self._pause()
                critic_opt.zero_grad(set_to_none=True)
                for i in range(0, len(selected), self.runtime.micro_hours):
                    micro = selected[i:i+self.runtime.micro_hours]
                    with torch.no_grad():
                        h = current.condition(self._context(micro), self.scale)
                        real = self._coefficients(micro)
                        fake = current.head(torch.randn_like(real), h)
                    loss = current.critic(fake, h).mean()-current.critic(real, h).mean()
                    loss = loss+10*gradient_penalty(current.critic, real, fake, h)
                    (loss*len(micro)/len(selected)).backward()
                torch.nn.utils.clip_grad_norm_(current.critic.parameters(), science.gradient_clip, error_if_nonfinite=True)
                critic_opt.step()
            current.critic.requires_grad_(False)

        pretrain = self._run(pre_engine, objective, before_update=critic_update if arm == 'gan' else None)
        # This immutable reference always points to estimated-coefficient pretraining.
        # ES checkpoints live exclusively in the separate observation directory.
        immutable_json(directory/'pretraining_source.json', dict(checkpoint=str(pretrain.resolve()),
                                                                sha256=sha256(pretrain)))
        model.load_state_dict(pre_engine.ema.state_dict())
        if joint_encoder:
            model.encoder.requires_grad_(True)
        if arm == 'gan':
            model.critic.requires_grad_(False)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if joint_encoder:
            groups = [{'params': model.encoder.parameters(), 'lr': 1e-5, 'lr_scale': .1},
                      {'params': model.head.parameters(), 'lr': 1e-4}]
            optimizers = [torch.optim.AdamW(groups, lr=science.refinement_lr, weight_decay=science.weight_decay)]
        else:
            optimizers = [torch.optim.AdamW(parameters, lr=science.refinement_lr, weight_decay=science.weight_decay)]
        refinement_science = replace(science, refinement_lr=1e-4) if joint_encoder else science
        observation_hours = self.manifest.select([2018], 1 if arm == 'det' else 4)
        refine_engine = self._engine(model, directory/'refine', observation_hours, f'{arm}_refine',
                       max(1, int(refinebudget*fraction)), refinebudget, 32,
                       dict(dependencies, pretraining=sha256(pretrain)), science=refinement_science,
                       optimizers=optimizers, clip_groups=[parameters])

        def observation(selected, current, step):
            return {'loss': self._observation_loss(selected, current, step)}

        self._run(refine_engine, observation, evaluator=self._screen(arm, directory/'refine'))
        if shared and not joint_encoder and model_digest(model.encoder) != dependencies['encoder']:
            raise ValueError('Frozen shared encoder changed during head training')
        result = {'pretrain_checkpoint': str(pretrain.resolve()),
                  'contract': str((directory/'pretrain'/'contract.json').resolve())}
        for product, rows in refine_engine.products.items():
            if rows:
                result[product] = dict(value=rows[0]['value'], checkpoint=rows[0]['checkpoint'])
        return result

    def condition(self):
        directory = self.root/'condition'
        result = self._functional_run('det', directory,
                  {'pretrain_lr': self.science.learning_rate, 'refinement_lr': self.science.refinement_lr}, shared=False)
        state = _load(directory/'refine'/'last.pt')
        rows = state['products'].get('point', [])
        evaluated = []
        for row in rows:
            metrics = self._evaluate('det', row['checkpoint'], directory/'full2019'/str(row['step']), full=True)
            evaluated.append((metrics['values']['point']['value'], row['step'], row['checkpoint']))
        if not evaluated:
            raise ScientificGateFailed('Condition stage requires post-startup C selection checkpoint')
        _, _, selected = min(evaluated)
        model = self._model('det', frozen=False)
        model.load_state_dict(_load(selected)['ema'])
        atomic_torch(directory/'frozen.pt', dict(encoder=model.encoder.state_dict(),
                     encoder_sha256=model_digest(model.encoder), checkpoint=selected,
                     checkpoint_sha256=sha256(selected), foundations=self._foundation_dependencies()))
        self._build_condition_cache(model)
        return {'encoder': directory/'frozen.pt', 'checkpoint': Path(selected),
                'pretrain_checkpoint': Path(result['pretrain_checkpoint'])}

    def _condition_cache_root(self, model):
        store = self._store('context')
        source = {name: metadata for name, metadata in store.manifest['files'].items() if name.startswith('A/')}
        key = fingerprint(dict(encoder=model_digest(model.encoder), foundations=self._foundation_dependencies(),
                               inputs=source, statistics=store.stats, grid=store.grid_id,
                               scenario=self.scenario.identity, use_history=self.science.use_history))
        return self.cache_root/'condition'/key

    def _condition_provider(self, model):
        if any(p.requires_grad for p in model.encoder.parameters()):
            return None
        path = self._condition_cache_root(model)/'contract.json'
        if not path.exists():
            return None
        contract = read_json(path)
        if contract['encoder_sha256'] != model_digest(model.encoder):
            return None
        if contract['scenario'] != self.scenario.identity or contract['use_history'] != self.science.use_history:
            raise ValueError('Frozen condition cache input policy changed')
        def provide(hours, context):
            if context.scenario_hash != contract['scenario']:
                raise ValueError('Frozen condition cache scenario differs')
            result = []
            for hour in hours:
                month = str(pd.to_datetime(hour, unit='h').strftime('%Y-%m'))
                cache_path = path.parent/f'{month}.pt'
                if not cache_path.exists() or not cache_path.with_suffix('.json').exists():
                    # D2020 is encoded once per inference batch from its deployed A inputs.
                    return model.condition(context, self.scale)
                key = (contract['contract'], month)
                if key not in self._condition_months:
                    meta = read_json(cache_path.with_suffix('.json'))
                    if meta['contract'] != contract['contract'] or meta['sha256'] != sha256(cache_path):
                        raise ValueError('Frozen condition cache checksum differs')
                    self._condition_months[key] = _load(cache_path)
                cached = self._condition_months[key]
                index = np.searchsorted(cached['hours'], hour)
                if cached['contract'] != contract['contract'] or index == len(cached['hours']) or cached['hours'][index] != hour:
                    return model.condition(context, self.scale)
                result.append(cached['h'][index])
            return torch.stack(result)
        return provide

    def _build_condition_cache(self, model):
        model.encoder.eval().requires_grad_(False)
        directory = self._condition_cache_root(model)
        directory.mkdir(parents=True, exist_ok=True)
        hours = np.r_[self._decoder_bundle()['hours'], self._hours(2019)]
        months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
        contract = fingerprint(dict(encoder=model_digest(model.encoder), foundations=self._foundation_dependencies(),
                                    scenario=self.scenario.identity, hours=hours))
        immutable_json(directory/'contract.json', dict(contract=contract, encoder_sha256=model_digest(model.encoder),
                       scenario=self.scenario.identity, use_history=self.science.use_history))
        immutable_json(self.root/'condition'/'cache.json', dict(directory=str(directory), contract=contract,
                       runtime_cache_root=str(self.cache_root)))
        for month in sorted(set(months)):
            path = directory/f'{month}.pt'
            if path.exists() and path.with_suffix('.json').exists():
                meta = read_json(path.with_suffix('.json'))
                if meta['contract'] != contract or meta['sha256'] != sha256(path):
                    raise ValueError('Frozen condition cache contract differs')
                continue
            selected = hours[months == month]
            features = []
            for hour in selected:
                self._pause()
                with torch.no_grad():
                    features.append(model.condition(self._context([hour]), self.scale).cpu()[0])
            atomic_torch(path, dict(contract=contract, hours=selected, h=torch.stack(features)))
            write_json(path.with_suffix('.json'), dict(contract=contract, sha256=sha256(path)))

    def _pixel_run(self, arm, directory, params, *, fraction=1., source=None):
        science = replace(self.science, learning_rate=params['pretrain_lr'], refinement_lr=params['refinement_lr'])
        model, directory = self._model(arm), Path(directory)
        hours = self.manifest.select([2018])
        variance = list(model.variance_head.parameters())
        mean_parameters = [p for n, p in model.named_parameters() if not n.startswith('variance_head.')]
        optimizers = [torch.optim.AdamW(mean_parameters, lr=science.learning_rate, weight_decay=science.weight_decay),
                      torch.optim.AdamW(variance, lr=science.learning_rate, weight_decay=science.weight_decay)]
        horizon = science.budgets[arm]
        # Both pixel arms start with identical seed weights and the same 2018 cursor.
        engine = self._engine(model, directory/'observations', hours, f'{arm}_observations',
                              max(1, int(horizon*fraction)), horizon, 8,
                              {'mean': sha256(self._mean_path()), 'initialization_seed': self.seed},
                              science=science, optimizers=optimizers, clip_groups=[mean_parameters, variance])
        if source is not None:
            self._fork(engine, source)

        def objective(selected, current, step):
            context = self._context(selected)
            field, features = predict_patches(current, context, self._store(), chunk=self.runtime.patch_chunk,
                                              residual=True, scale=self.scale)
            truth, valid, xy, heights = self._labels(selected)
            point = surface_query(field, xy, heights).squeeze(1)
            station_features = query_grid(features, xy).transpose(1, 2)
            geometry = []
            store, builder = self._store(), self._builder()
            for hour in selected:
                available = np.isfinite(store.context_values([hour])[0]).all(-1)
                _, geo = builder._geometry(available.tobytes(), 'stations')
                ids = store.ids('B')
                geometry.append(np.c_[heights/3000, np.minimum(geo['nearest'][ids]/400000, 2.5), geo['count'][ids]/16])
            std = current.standard_deviation(station_features, _tensor(geometry, self.runtime.device),
                                             store.stats['surface_std'], faithful=arm == 'm2_f')
            mse, nll = gaussian_losses(point[valid], truth[valid], std[valid].square(),
                                       store.stats['surface_std'], arm == 'm2_f')
            loss = mse+nll if arm == 'm2_f' else nll
            return dict(loss=loss, mse=mse, nll=nll)

        checkpoint = self._run(engine, objective, evaluator=self._screen(arm, directory/'observations'))
        result = dict(pretrain_checkpoint=str(checkpoint.resolve()),
                      contract=str((directory/'observations'/'contract.json').resolve()))
        for product, rows in engine.products.items():
            if rows:
                result[product] = dict(value=rows[0]['value'], checkpoint=rows[0]['checkpoint'])
        return result

    @staticmethod
    def _reference_validator(path, role):
        if role == 'contract':
            obj = read_json(path)
            if not obj.get('signature') or not obj.get('spec'):
                raise ValueError('Missing scientific training contract')
            return
        obj = _load(path)
        if not obj.get('signature') or not obj.get('spec') or 'model' not in obj:
            raise ValueError('Invalid training checkpoint')
        if role == 'pretrain_checkpoint':
            stage = obj['spec']['stage']
            if not (stage.endswith('_pretrain') or stage == 'm2_f_observations'):
                raise ValueError('Promotion source must be estimated pretraining or M2 observation warm-start')

    def search(self, arm):
        from gnss_field.orchestration.study import run_study
        if arm not in ('det', 'm2_f', 'gaussian', 'edm', 'gan'):
            raise ValueError('Unknown registered search arm')
        directory = self.root/'search'/arm

        def objective(params, trial_directory, number):
            runner = self._pixel_run if arm == 'm2_f' else self._functional_run
            return runner(arm, trial_directory, params, fraction=.25)

        report = run_study(directory, arm, objective, trials=self.science.study_trials,
                           seed=self.science.study_seed, startup=self.science.study_startup,
                           validate_references=True, reference_validator=self._reference_validator)
        if report['state'] != 'COMPLETE' or not report['promotions']:
            raise ScientificGateFailed('Search has no complete promotable configuration')
        return {'report': directory/'report.json'}

    def promote(self):
        entries = []
        for arm in ('det', 'm2_f', 'gaussian', 'edm', 'gan'):
            report = read_json(self.root/'search'/arm/'report.json')
            if len(report['promotions']) > 2:
                raise ValueError('Product promotion union exceeds two complete configurations')
            for choice in report['promotions']:
                self._pause()
                directory = self.root/'promoted'/arm/f"trial_{choice['trial_number']:04d}"
                runner = self._pixel_run if arm == 'm2_f' else self._functional_run
                result = runner(arm, directory, choice['params'], source=choice['pretrain_checkpoint'])
                entries.append(dict(arm=arm, directory=str(directory), params=choice['params'], result=result))
                if arm == 'm2_f':
                    j_directory = self.root/'promoted'/'m2_j'/f"trial_{choice['trial_number']:04d}"
                    j_result = self._pixel_run('m2_j', j_directory, choice['params'])
                    entries.append(dict(arm='m2_j', directory=str(j_directory), params=choice['params'], result=j_result))
        write_json(self.root/'promotions.json', {'entries': entries})
        return {'promotions': self.root/'promotions.json'}

    def _record(self, arm, checkpoint, directory, result, step, *, members=64):
        return dict(model=arm, checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_sha256=sha256(checkpoint), evaluation_directory=str(Path(directory).resolve()),
                    stage=f'{arm}_refine', step=step, year=2019, full_year=True, members=members,
                    selection_values=result['values'], metrics=result['metrics'])

    def full2019_rerank(self):
        from gnss_field.evaluation import select_products
        from .baselines import fit_geostat, get_baseline_predictor
        rows = []
        for entry in read_json(self.root/'promotions.json')['entries']:
            arm, directory = entry['arm'], Path(entry['directory'])
            final = directory/('observations' if arm.startswith('m2_') else 'refine')/'last.pt'
            state = _load(final)
            paths = {r['checkpoint']: r['step'] for candidates in state['products'].values() for r in candidates}
            for checkpoint, step in paths.items():
                output = directory/'full2019'/Path(checkpoint).stem/'K64'
                result = self._evaluate(arm, checkpoint, output, full=True, members=64)
                rows.append(self._record(arm, checkpoint, output, result, step))
        # Every baseline is evaluated on exactly the same C-hour population.
        store = self._store('development')
        parameters_path = self.root/'baselines'/'geostat.json'
        parameters = fit_geostat(self._store(), parameters_path)
        for name in ('coarse_bilinear', 'A_IDW', 'A_kriging', 'M0_innovation_IDW'):
            output = self.root/'baselines'/name/'full2019'
            predictor = get_baseline_predictor(name, store, self.runtime,
                                               mean_cache=self._mean_cache(), parameters=parameters)
            result = evaluate_predictor(store, self._hours(2019, store), predictor, output, members=64,
                      scenario=self.scenario, pause_check=self._pause, full_year=True,
                      checkpoint_hash=fingerprint(dict(name=name, parameters=parameters)),
                      use_history=self.science.use_history)
            rows.append(self._record(name, parameters_path, output, result, 30001))
        for arm in ('mean_base', 'mean_all'):
            checkpoint = self._mean_path(arm)
            output = self.root/arm/'baseline_full2019'
            result = self._evaluate(arm, checkpoint, output, full=True, members=64)
            rows.append(self._record(arm, checkpoint, output, result, _load(checkpoint)['step']))
        selections = {arm: select_products([r for r in rows if r['model'] == arm], require_full_year=True)
                      for arm in sorted({r['model'] for r in rows})}
        write_json(self.root/'full2019.json', dict(records=rows, selections=selections))
        return {'rerank': self.root/'full2019.json'}

    def _direction_decisions(self, records):
        from gnss_field.evaluation import direction_gate, probability_gate, select_bstar, select_products
        controls = ('coarse_bilinear', 'A_IDW', 'A_kriging', 'mean_base', 'mean_all',
                    'M0_innovation_IDW', 'm2_f', 'm2_j', 'det', 'gaussian')
        bstar = select_bstar(records, controls)
        gaussian = select_products([r for r in records if r['model'] == 'gaussian'],
                                   require_full_year=True)['joint']['selected']
        families = {}
        for arm in ('det', 'gaussian', 'edm', 'gan'):
            products = select_products([r for r in records if r['model'] == arm], require_full_year=True)
            point = products['point']['selected']
            if point is None:
                continue
            fine, baseline = point['metrics']['fine'], bstar['metrics']['fine']
            endpoints = {
                'point_vs_bstar': (fine['model']['station_macro_rmse_mm'], baseline['model']['station_macro_rmse_mm']),
                'point_vs_self100': (fine['model']['station_macro_rmse_mm'], fine['coarsened']['station_macro_rmse_mm']),
                'increment_vs_bstar': (fine['model_increments']['0-50']['group_macro_rmse_mm'],
                                       baseline['model_increments']['0-50']['group_macro_rmse_mm']),
                'increment_vs_self100': (fine['model_increments']['0-50']['group_macro_rmse_mm'],
                                         fine['coarsened_increments']['0-50']['group_macro_rmse_mm'])}
            if any(x is None or y is None for x, y in endpoints.values()):
                fine_gate = dict(fine_pass=False, reason='missing registered fine endpoint')
            else:
                fine_gate = direction_gate({k: {'difference_mm': x-y} for k, (x, y) in endpoints.items()}, formal=False)
            joint = products['joint']['selected']
            probability = dict(probability_pass=False)
            if joint is not None and gaussian is not None:
                m, g = joint['metrics'], gaussian['metrics']
                probability = probability_gate(
                    es_difference=m['joint']['group_macro_es_mm']-g['joint']['group_macro_es_mm'],
                    vs_difference=m['joint']['group_macro_vs_mm']-g['joint']['group_macro_vs_mm'],
                    crps_difference=m['primary']['station_macro_crps_mm']-g['primary']['station_macro_crps_mm'],
                    intervals=m['primary']['intervals'], formal=False)
            families[arm] = dict(fine=fine_gate, probability=probability, products=products,
                                 advances=bool(fine_gate['fine_pass'] or probability.get('calibrated_joint_pass', False)))
        return dict(bstar=bstar, gaussian=gaussian, families=families,
                    any_gain=any(r['advances'] for r in families.values()), year=2019)

    def stability(self):
        from gnss_field.evaluation import sampling_stability
        old = read_json(self.root/'full2019.json')['records']
        new, checks = [], []
        old_gates = self._direction_decisions(old)
        for record in old:
            if record['model'] not in ('gaussian', 'edm', 'gan'):
                new.append(record)
                continue
            output = Path(record['evaluation_directory']).parent/'K128'
            result = self._evaluate(record['model'], record['checkpoint'], output, full=True, members=128)
            updated = self._record(record['model'], record['checkpoint'], output, result, record['step'], members=128)
            a, b = _arrays(record['evaluation_directory']), _arrays(output)
            check = sampling_stability(a['samples'], b['samples'],
                point64_mm=record['selection_values']['point']['value'], point128_mm=result['values']['point']['value'],
                order64=[], order128=[], gates64={}, gates128={}, tolerance_mm=self.science.stability_tolerance_mm,
                member_ids64=np.arange(64), member_ids128=np.arange(128), prefix_tolerance_mm=.001,
                noise_provenance64=self._sampling_provenance(record, a),
                noise_provenance128=self._sampling_provenance(updated, b))
            checks.append(dict(check, checkpoint=record['checkpoint']))
            new.append(updated)
        def order(records, product):
            eligible = [r for r in records if product in r['selection_values']]
            return [r['checkpoint'] for r in sorted(eligible, key=lambda r: (r['selection_values'][product]['value'], r['checkpoint']))]
        order_changed = any(order(old, p) != order(new, p) for p in ('point', 'joint'))
        new_gates = self._direction_decisions(new)
        def flags(gates):
            return {arm: (r['fine']['fine_pass'], r['probability'].get('calibrated_joint_pass', False))
                    for arm, r in gates['families'].items()}
        gates_changed = flags(old_gates) != flags(new_gates)
        stable = all(r['stable'] for r in checks) and not order_changed and not gates_changed
        selected_records = old if stable else new
        result = dict(stable=stable, full_128_rerank_complete=True, members=64 if stable else 128,
                      order_changed=order_changed, gates_changed=gates_changed, checks=checks,
                      point_instability=not stable, selection_year=2019)
        write_json(self.root/'stability.json', result)
        write_json(self.root/'stable2019.json', {'records': selected_records})
        return {'stability': self.root/'stability.json', 'records': self.root/'stable2019.json'}

    def _sampling_provenance(self, record, arrays):
        store = self._store('context')
        inputs = {name: meta for name, meta in store.manifest['files'].items() if name.startswith('A/')}
        sampler = dict(arm=record['model'], implementation='keyed_noise_sha256_hour_member_v1',
                       steps=self.science.sampler_steps, sigma_min=self.science.sigma_min,
                       sigma_max=self.science.sigma_max, rho=self.science.rho,
                       sigma_data=self.science.sigma_data, rank=self.science.rank)
        return dict(sampler=sampler, sampler_sha256=fingerprint(sampler), seed=self.science.sampling_seed,
                    release_sha256=self._source_digest,
                    checkpoint_sha256=record['checkpoint_sha256'], foundations=self._foundation_dependencies(),
                    input_sha256=fingerprint(dict(files=inputs, statistics=store.stats, grid=store.grid_id,
                        use_history=self.science.use_history, scenario=self.scenario.identity)),
                    hours_sha256=fingerprint(arrays['hours']), station_ids_sha256=fingerprint(arrays['station_ids']))

    def direction_gate(self):
        result = self._direction_decisions(read_json(self.root/'stable2019.json')['records'])
        result['status'] = 'ADVANCE' if result['any_gain'] else 'STOPPED_NO_GAIN'
        path = self.root/'direction_gate.json'
        write_json(path, result)
        if not result['any_gain']:
            raise ScientificGateFailed('STOPPED_NO_GAIN')
        return {'gate': path}

    def _advanced(self):
        gate = read_json(self.root/'direction_gate.json')
        return {arm: entry for arm, entry in gate['families'].items() if entry['advances']}

    def _locked_configurations(self, arm):
        family = self._advanced()[arm]
        result = {}
        for product in ('point', 'joint'):
            passed = family['fine']['fine_pass'] if product == 'point' else family['probability'].get('calibrated_joint_pass', False)
            selected = family['products'][product]['selected']
            if not passed or selected is None:
                continue
            checkpoint = Path(selected['checkpoint'])
            contract = read_json(checkpoint.parent/'contract.json')
            params = {'pretrain_lr': contract['science']['learning_rate'],
                      'refinement_lr': contract['science']['refinement_lr']}
            key = fingerprint(params)[:12]
            item = result.setdefault(key, dict(key=key, params=params, products=[],
                                    pretrain_checkpoint=str(checkpoint.parent.parent/'pretrain'/'last.pt'),
                                    selected_checkpoints={}))
            item['products'].append(product)
            item['selected_checkpoints'][product] = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint))
            item['pretrain_checkpoint_sha256'] = sha256(item['pretrain_checkpoint'])
        return list(result.values())

    def _copy_foundations(self, child, *, mean=True, basis=True):
        import shutil
        if mean:
            for arm in ('mean_base', 'mean_all'):
                destination = child.root/arm/'frozen.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root/arm/'frozen.json', destination)
            child._cache = self._mean_cache()
        if basis:
            destination = child.root/'basis_final'/'decoder.pt'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root/'basis_final'/'decoder.pt', destination)

    def _rebuild_foundations(self, child, *, mean=True, basis=True):
        if mean:
            child.mean_base()
            child.mean_all()
            child.mean_cache()
        if basis:
            for fold in range(3):
                child.basis_cv(fold)
            child.representation_gate()
            child.basis_final()
        child.condition()

    def _select_child_products(self, child, trained):
        from gnss_field.evaluation import select_products
        records, eligible, bindings = [], {}, {}
        members = read_json(self.root/'stability.json')['members']
        for entry in trained:
            arm, config = entry['arm'], entry['config']
            parent_configs = {c['key']: c for c in self._locked_configurations(arm)}
            expected = parent_configs.get(config['key'])
            if (expected is None or fingerprint(expected['params']) != fingerprint(config['params'])
                    or set(expected['products']) != set(config['products'])):
                raise ValueError('Auxiliary training changed the parent locked hyperconfiguration')
            for product in config['products']:
                identity = (arm, product)
                if identity in bindings and bindings[identity] != config['key']:
                    raise ValueError('A family product has multiple parent hyperconfigurations')
                bindings[identity] = config['key']
            state = _load(Path(entry['directory'])/'refine'/'last.pt')
            candidates = {r['checkpoint']: r['step'] for rows in state['products'].values() for r in rows}
            for checkpoint, step in candidates.items():
                output = Path(entry['directory'])/'full2019'/Path(checkpoint).stem/f'K{members}'
                result = child._evaluate(entry['arm'], checkpoint, output, full=True, members=members)
                row = child._record(entry['arm'], checkpoint, output, result, step, members=members)
                row.update(configuration_key=config['key'], parent_hyperparameters=config['params'])
                records.append(row)
                for product in entry['config']['products']:
                    eligible.setdefault((arm, product), []).append(row)
        selected = {}
        for (arm, product), candidates in eligible.items():
            choice = select_products(candidates, products=(product,), require_full_year=True)[product]['selected']
            if choice is None:
                raise ScientificGateFailed(f'No eligible auxiliary checkpoint for {arm}/{product}')
            selected.setdefault(arm, {})[product] = choice
        return dict(records=records, selected_products=selected, seed=child.seed,
                    science=asdict(child.science), directory=str(child.directory),
                    foundations=child._foundation_dependencies(),
                    condition_sha256=sha256(child.root/'condition'/'frozen.pt'))

    def ablation(self, variant):
        allowed = ('pretrain_only', 'no_geometry_bias', 'current_only', 'rank64', 'rank256', 'joint_encoder')
        if variant not in allowed:
            raise ValueError('Unknown one-factor ablation')
        advanced = self._advanced()
        if not advanced:
            raise ScientificGateFailed('No passed family for ablation')
        directory = self.root/'ablations'/variant
        directory.mkdir(parents=True, exist_ok=True)
        if variant == 'pretrain_only':
            results, selected = {}, {}
            for arm in advanced:
                for config in self._locked_configurations(arm):
                    source = Path(config['pretrain_checkpoint'])
                    if not _load(source)['spec']['stage'].endswith('_pretrain'):
                        raise ValueError('Pretraining ablation source phase differs')
                    output = directory/arm/config['key']
                    result = self._evaluate(arm, source, output, full=True, members=read_json(self.root/'stability.json')['members'])
                    results[f"{arm}_{config['key']}"] = dict(metrics=result['metrics'], checkpoint=str(source), sha256=sha256(source))
                    row = self._record(arm, source, output, result, _load(source)['step'],
                                       members=read_json(self.root/'stability.json')['members'])
                    row['stage'] = f'{arm}_pretrain'
                    row.update(configuration_key=config['key'], parent_hyperparameters=config['params'])
                    for product in config['products']:
                        products = selected.setdefault(arm, {})
                        if product in products:
                            raise ValueError('Pretraining ablation requires one parent configuration per family product')
                        products[product] = row
            write_json(directory/'results.json', dict(results=results, selected_products=selected,
                directory=str(self.directory), science=asdict(self.science), seed=self.seed,
                foundations=self._foundation_dependencies(), condition_sha256=sha256(self.root/'condition'/'frozen.pt')))
            return {'results': directory/'results.json'}
        science = self.science
        if variant == 'no_geometry_bias':
            science = replace(science, geometry_bias=False)
        elif variant == 'current_only':
            science = replace(science, use_history=False)
        elif variant.startswith('rank'):
            science = replace(science, rank=int(variant[4:]))
        child = TrainingPipeline(science, self.runtime, directory, self.seed, parent_directory=self.directory)
        rebuild_mean = variant == 'current_only'
        rebuild_basis = variant in ('current_only', 'rank64', 'rank256')
        self._copy_foundations(child, mean=not rebuild_mean, basis=not rebuild_basis)
        if variant == 'joint_encoder':
            # The same selected pretraining trajectory is replayed exactly before encoder refinement.
            import shutil
            destination = child.root/'condition'/'frozen.pt'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root/'condition'/'frozen.pt', destination)
        else:
            self._rebuild_foundations(child, mean=rebuild_mean, basis=rebuild_basis)
        results, trained = {}, []
        for arm in advanced:
            if variant == 'joint_encoder' and arm != 'edm':
                continue
            for config in self._locked_configurations(arm):
                source = config['pretrain_checkpoint'] if variant == 'joint_encoder' else None
                head_directory = child.root/'heads'/arm/config['key']
                results[f"{arm}_{config['key']}"] = child._functional_run(arm, head_directory,
                               config['params'], source=source, joint_encoder=variant == 'joint_encoder')
                trained.append(dict(arm=arm, directory=str(head_directory), config=config))
        summary = self._select_child_products(child, trained)
        write_json(directory/'results.json', dict(summary, results=results))
        return {'results': directory/'results.json'}

    def replicate(self, seed):
        if seed not in (17, 101):
            raise ValueError('Registered repeat seeds are 17 and 101')
        child = TrainingPipeline(self.science, self.runtime, self.directory, seed,
                                 parent_directory=self.directory)
        self._rebuild_foundations(child)
        results, trained = {}, []
        for arm in self._advanced():
            for config in self._locked_configurations(arm):
                directory = child.root/'heads'/arm/config['key']
                result = child._functional_run(arm, directory, config['params'])
                results[f"{arm}_{config['key']}"] = dict(arm=arm, config=config, result=result)
                trained.append(dict(arm=arm, directory=str(directory), config=config))
        summary = self._select_child_products(child, trained)
        write_json(child.root/'replication.json', dict(summary, results=results))
        return {'replication': child.root/'replication.json'}

    def _config_hash(self):
        # Include the instantiated configuration so a caller cannot change science
        # while leaving an old resolved.json on disk.
        resolved = self.directory/'resolved.json'
        return fingerprint(dict(science=asdict(self.science), runtime=asdict(self.runtime),
                                resolved_sha256=sha256(resolved) if resolved.exists() else None))

    def product_lock(self):
        from gnss_field.evaluation import build_product_lock
        gate = read_json(self.root/'direction_gate.json')
        families = gate['families']
        products = {}
        for product in ('point', 'joint'):
            candidates = []
            for arm, family in families.items():
                passed = family['fine']['fine_pass'] if product == 'point' else family['probability'].get('calibrated_joint_pass', False)
                selected = family['products'][product]['selected']
                if passed and selected is not None:
                    candidates.append(selected)
            if candidates:
                selected = min(candidates, key=lambda r: (r['selection_values'][product]['value'], r['checkpoint']))
                products[product] = {'selected': selected}
        stability = read_json(self.root/'stability.json')
        lock = build_product_lock(products, config_sha256=self._config_hash(),
                    manifest_sha256=self._store('context').manifest['fingerprint'], stability=stability)
        lock['foundations'] = self._foundation_dependencies()
        lock['condition_sha256'] = sha256(self.root/'condition'/'frozen.pt')
        lock['controls'] = {'bstar': gate['bstar'], 'gaussian': gate['gaussian']}
        from gnss_field.evaluation import select_products
        records = read_json(self.root/'stable2019.json')['records']
        for arm in ('m2_f', 'm2_j'):
            selected = select_products([r for r in records if r['model'] == arm],
                       products=('marginal',), require_full_year=True)['marginal']['selected']
            if selected is not None:
                if sha256(selected['checkpoint']) != selected['checkpoint_sha256']:
                    raise ValueError('M2 marginal checkpoint changed before lock')
                lock['controls'][arm+'_marginal'] = dict(selected, product='marginal',
                    claim_scope='station marginal distribution only; no joint field claim')
        lock['export_events'] = {'source': 'A_only', 'rule': 'twelve largest daily mean absolute A Z0 innovations',
                                 'hours_per_event': 24}
        lock['seed'] = self.seed
        lock['release_sha256'] = self._source_digest
        lock['family_configurations'] = {arm: self._locked_configurations(arm) for arm in self._advanced()}
        lock['replicates'] = {str(seed): read_json(self.directory/f'seed_{seed}'/'replication.json') for seed in (17, 101)}
        lock['ablations'] = {variant: read_json(self.root/'ablations'/variant/'results.json')
            for variant in ('pretrain_only', 'no_geometry_bias', 'current_only', 'rank64', 'rank256', 'joint_encoder')}
        lock['lock_sha256'] = fingerprint({k: v for k, v in lock.items() if k != 'lock_sha256'})
        immutable_json(self.root/'product_lock.json', lock)
        return {'lock': self.root/'product_lock.json'}

    def _validated_lock(self):
        from gnss_field.evaluation import validate_retrospective_access
        path = self.root/'product_lock.json'
        if not path.exists():
            raise PermissionError('Retrospective evaluation requires product lock')
        lock = read_json(path)
        if lock.get('release_sha256') != self._source_digest:
            raise ValueError('Software release changed after product lock')
        manifest_hash = read_json(Path(self.runtime.data_root)/'manifest.json')['fingerprint']
        for product, entry in lock['products'].items():
            current_hash = sha256(entry['checkpoint'])
            validate_retrospective_access(lock, year=2020, role='D', config_sha256=self._config_hash(),
                  manifest_sha256=manifest_hash, checkpoint_sha256=current_hash, product=product)
        if lock['foundations'] != self._foundation_dependencies():
            raise ValueError('Frozen foundation hashes changed after product lock')
        if lock['condition_sha256'] != sha256(self.root/'condition'/'frozen.pt'):
            raise ValueError('Frozen condition hash changed after product lock')
        for entry in lock['controls'].values():
            if entry is not None and sha256(entry['checkpoint']) != entry['checkpoint_sha256']:
                raise ValueError('Locked control checkpoint hash changed')
        self._validate_auxiliary(lock)
        return lock

    def _validate_auxiliary(self, lock):
        from gnss_field.contracts import ScienceConfig
        configurations = lock['family_configurations']
        for configs in configurations.values():
            for config in configs:
                if sha256(config['pretrain_checkpoint']) != config['pretrain_checkpoint_sha256']:
                    raise ValueError('Parent family pretraining checkpoint changed after lock')
                for reference in config['selected_checkpoints'].values():
                    if sha256(reference['checkpoint']) != reference['sha256']:
                        raise ValueError('Parent family selection checkpoint changed after lock')
        for group in ('replicates', 'ablations'):
            for name, result in lock[group].items():
                child = TrainingPipeline(ScienceConfig(**result['science']), self.runtime, result['directory'],
                                          result['seed'], parent_directory=self.directory)
                if child._foundation_dependencies() != result['foundations']:
                    raise ValueError('Locked repeat/ablation foundations changed')
                if sha256(child.root/'condition'/'frozen.pt') != result['condition_sha256']:
                    raise ValueError('Locked repeat/ablation encoder changed')
                expected = {arm: {p: config for config in configs for p in config['products']}
                            for arm, configs in configurations.items()
                            if group != 'ablations' or name != 'joint_encoder' or arm == 'edm'}
                if set(result['selected_products']) != set(expected):
                    raise ValueError('Locked repeat/ablation family coverage differs')
                for arm, products in result['selected_products'].items():
                    if set(products) != set(expected[arm]):
                        raise ValueError('Locked repeat/ablation family products differ')
                    for product, selected in products.items():
                        config = expected[arm][product]
                        if (selected['model'] != arm or selected['configuration_key'] != config['key']
                                or fingerprint(selected['parent_hyperparameters']) != fingerprint(config['params'])):
                            raise ValueError('Locked repeat/ablation parent configuration differs')
                        if selected['year'] != 2019 or not selected['full_year'] or sha256(selected['checkpoint']) != selected['checkpoint_sha256']:
                            raise ValueError('Locked repeat/ablation checkpoint differs')

    def _control_predictor(self, entry, store):
        arm = entry['model']
        if arm in ('coarse_bilinear', 'A_IDW', 'A_kriging', 'M0_innovation_IDW'):
            from .baselines import get_baseline_predictor
            return get_baseline_predictor(arm, store, self.runtime, mean_cache=self._mean_cache(),
                                          parameters=read_json(entry['checkpoint']))
        return self._predictor(arm, entry['checkpoint'], store)

    def evaluate2020(self):
        # All current hashes are verified before opening the D role view or cache.
        lock = self._validated_lock()
        store = PackedStore(self.runtime.data_root, purpose='retrospective', retrospective_lock=lock)
        hours, directory = self._hours(2020, store), self.root/'evaluate2020'
        self._mean_cache().build(self._mean_model(), self._builder(store), hours, device=self.runtime.device,
                    chunk=self.runtime.patch_chunk, pause_check=self._pause, scenario=self.scenario)
        results, records = {}, {}
        for name, entry in lock['controls'].items():
            if entry is None:
                continue
            output = directory/'controls'/name
            result = evaluate_predictor(store, hours, self._control_predictor(entry, store), output,
                        members=lock['stability']['members'], role='D', scenario=self.scenario,
                        full_year=True, pause_check=self._pause, checkpoint_hash=sha256(entry['checkpoint']),
                        use_history=self.science.use_history)
            results[name], records[name] = result, _arrays(output)
        for product, entry in lock['products'].items():
            output = directory/product
            result = self._evaluate(entry['model'], entry['checkpoint'], output, full=True,
                                    members=lock['stability']['members'], store=store, role='D')
            results[product], records[product] = result, _arrays(output)
        formal = self._formal_evaluation(store, records, results)
        auxiliary = self._evaluate_auxiliary(lock, store, directory, records, results)
        write_json(directory/'summary.json', dict(lock_sha256=lock['lock_sha256'], results=results, formal=formal,
                                                 auxiliary=auxiliary, selection='locked C2019 products',
                                                 evaluation='D2020 retrospective'))
        return {'summary': directory/'summary.json'}

    def _evaluate_auxiliary(self, lock, store, directory, controls, control_results):
        from gnss_field.contracts import ScienceConfig
        output = {}
        for group in ('replicates', 'ablations'):
            for name, result in lock[group].items():
                self._pause()
                child = TrainingPipeline(ScienceConfig(**result['science']), self.runtime, result['directory'],
                                          result['seed'], parent_directory=self.directory)
                if not result['selected_products']:
                    output[f'{group}_{name}'] = {'status': 'SKIPPED_NO_PASSED_FAMILY'}
                    continue
                child._mean_cache().build(child._mean_model(), child._builder(store), self._hours(2020, store),
                            device=self.runtime.device, chunk=self.runtime.patch_chunk,
                            pause_check=self._pause, scenario=self.scenario)
                families = {}
                for arm, products in result['selected_products'].items():
                    records = {key: value for key, value in controls.items() if key in ('bstar', 'gaussian')}
                    metrics = {key: value for key, value in control_results.items() if key in ('bstar', 'gaussian')}
                    for product, selected in products.items():
                        if selected['model'] != arm:
                            raise ValueError('Auxiliary retrospective product changed family identity')
                        path = directory/group/name/arm/product
                        metrics[product] = child._evaluate(arm, selected['checkpoint'], path, full=True,
                                       members=lock['stability']['members'], store=store, role='D')
                        records[product] = _arrays(path)
                    # Each family's fixed parent configurations receive their own
                    # seed confirmation and mechanistic ablation metrics.
                    families[arm] = dict(results={p: metrics[p] for p in products}, sources=products,
                        formal=child._formal_evaluation(store, records, metrics) if group == 'replicates' else {})
                output[f'{group}_{name}'] = {'families': families}
        return output

    def _fine_bootstrap(self, store, model, baseline, *, block_days=7):
        from gnss_field.evaluation import paired_block_bootstrap, pair_increment_arrays, crossed_increment_bootstrap, direction_gate
        from .prediction import safe_fine_sites
        ids, hours, truth = model['station_ids'], model['hours'], model['truth']
        if not np.array_equal(hours, baseline['hours']) or not np.array_equal(ids, baseline['station_ids']):
            raise ValueError('Formal comparisons require aligned station/hour forecast artifacts')
        population = model['population_F'] & model['normal_support_S0'] & safe_fine_sites(store, ids)[None]
        metadata = store.stations.iloc[ids].reset_index(drop=True)
        kwargs = dict(replicates=self.science.bootstrap_replicates, block_days=block_days,
                      seed=self.science.sampling_seed)
        results = {}
        for name, comparison in (('bstar', baseline['prediction']), ('self100', model['coarsened_prediction'])):
            results[f'point_vs_{name}'] = paired_block_bootstrap(hours, truth, model['prediction'], comparison, population, **kwargs)
            p = pair_increment_arrays(truth, model['prediction'], population, metadata)
            b = pair_increment_arrays(truth, comparison, population, metadata)
            valid = p['population'] & (p['distance_km'][None] <= 50)
            results[f'increment_vs_{name}'] = crossed_increment_bootstrap(hours, p['truth'], p['prediction'],
                                      b['prediction'], valid, p['group_ids'], **kwargs)
        return dict(comparisons=results, gate=direction_gate(results, formal=True), block_days=block_days,
                    population='F intersect S0 intersect fine sites')

    def _probability_cells(self, store, arrays):
        from scipy.spatial.distance import cdist
        from gnss_field.evaluation import crps_ensemble, energy_score_ensemble, variogram_score
        truth, members = arrays['truth'], arrays['samples']
        valid = arrays['population_F'] & arrays['normal_support_S0']
        if not np.isfinite(truth[valid]).all() or not np.isfinite(members.transpose(0, 2, 1)[valid]).all():
            raise ValueError('Probability forecasts and truth must be finite on declared scored cells')
        crps = np.full(truth.shape, np.nan)
        crps[valid] = crps_ensemble(truth[valid], members.transpose(0, 2, 1)[valid])
        result = {'crps': crps}
        for level in (.8, .9, .95):
            low, high = np.quantile(members, [(1-level)/2, (1+level)/2], axis=1, method='linear')
            result[str(round(level*100))] = np.where(valid, (truth >= low) & (truth <= high), np.nan)
        metadata = store.stations.iloc[arrays['station_ids']]
        groups = metadata.dense_group.to_numpy()
        unique = np.unique(groups[~pd.isna(groups)])
        es, vs = np.full((len(truth), len(unique)), np.nan), np.full((len(truth), len(unique)), np.nan)
        for j, group in enumerate(unique):
            for t in range(len(truth)):
                self._pause()
                selected = np.flatnonzero((groups == group) & valid[t])
                if len(selected) >= 4:
                    observed, samples = truth[t, selected], members[t][:, selected]
                    distance = cdist(metadata[['x', 'y']].to_numpy()[selected], metadata[['x', 'y']].to_numpy()[selected])/1000
                    es[t, j] = energy_score_ensemble(observed, samples)
                    vs[t, j] = variogram_score(observed, samples, distance)
        result.update(es=es, vs=vs)
        return result

    def _cell_bootstrap(self, hours, model, baseline, *, crossed=False, block_days=7,
                        max_attempts=None):
        from gnss_field.evaluation.bootstrap import (_blocks, _draw_blocks, _attempt_limit,
                                                     BootstrapExhaustedError)
        model, baseline = np.asarray(model), np.asarray(baseline)
        if model.ndim != 2 or model.shape != baseline.shape or len(hours) != len(model):
            raise ValueError('Probability comparison cell dimensions differ')
        if np.isinf(model).any() or np.isinf(baseline).any():
            raise ValueError('Probability cells must be finite or declared missing')
        valid = np.isfinite(model)
        if not np.array_equal(valid, np.isfinite(baseline)):
            raise ValueError('Probability comparison validity differs')
        keep = valid.any(0)
        a, b = model[:, keep], baseline[:, keep]
        if not a.shape[1] or self.science.bootstrap_replicates < 1:
            raise ValueError('No scored probability cells or positive replicate count')
        def statistic(x):
            # Accepted draws retain every selected column occurrence.
            return float((np.nansum(x, axis=0)/np.isfinite(x).sum(0)).mean())
        rng = np.random.default_rng(self.science.sampling_seed)
        blocks = _blocks(hours, block_days)
        values = []
        attempts = rejected = 0
        limit = _attempt_limit(self.science.bootstrap_replicates, max_attempts)
        while len(values) < self.science.bootstrap_replicates:
            if attempts == limit:
                raise BootstrapExhaustedError(attempts, rejected, len(values), self.science.bootstrap_replicates)
            attempts += 1
            index = _draw_blocks(blocks, rng)
            columns = rng.integers(0, a.shape[1], a.shape[1]) if crossed else np.arange(a.shape[1])
            aa, bb = a[index][:, columns], b[index][:, columns]
            if not np.isfinite(aa).any(0).all():
                rejected += 1
                continue
            values.append(statistic(aa)-statistic(bb))
        return dict(difference=statistic(a)-statistic(b), ci95=np.quantile(values, [.025, .975]).tolist(),
                    replicate_differences=values, block_days=block_days, replicates=len(values),
                    attempts=attempts, rejected_draws=rejected, rejection_rate=rejected/attempts,
                    max_attempts=limit, active_column_indices=np.flatnonzero(keep).tolist(),
                    n_columns=int(keep.sum()), n_blocks=len(blocks), seed=self.science.sampling_seed,
                    population='fixed eligible dense groups' if crossed else 'fixed eligible stations',
                    resampling='calendar blocks and dense groups' if crossed else 'calendar blocks, fixed stations')

    def _formal_evaluation(self, store, records, results):
        from gnss_field.evaluation import holm_adjust, probability_gate
        probability_cells = None
        if 'joint' in records and 'gaussian' in records:
            a, b = records['joint'], records['gaussian']
            for key in ('hours', 'station_ids', 'population_F', 'normal_support_S0'):
                if key in a or key in b:
                    if key not in a or key not in b or not np.array_equal(a[key], b[key]):
                        raise ValueError('Formal probability comparisons require aligned artifacts and populations')
            probability_cells = (self._probability_cells(store, a), self._probability_cells(store, b))
        primary = self._formal_evaluation_block(store, records, results, 7, probability_cells)
        sensitivity = self._formal_evaluation_block(store, records, results, 14, probability_cells)
        sensitivity['changed_conclusions'] = [name for name, claim in primary['selected_product_claims'].items()
            if claim['confirmed'] != sensitivity['selected_product_claims'][name]['confirmed']]
        for endpoint, flag in (('fine', 'fine_pass'), ('probability', 'calibrated_joint_pass')):
            if endpoint in primary and primary[endpoint]['gate'][flag] != sensitivity[endpoint]['gate'][flag]:
                sensitivity['changed_conclusions'].append(endpoint+'_gate')
        if ('probability' in primary and primary['probability']['gate']['formal_coverage']
                != sensitivity['probability']['gate']['formal_coverage']):
            sensitivity['changed_conclusions'].append('nominal_coverage_confirmation')
        primary['sensitivity'] = {'14': sensitivity}
        return primary

    def _formal_evaluation_block(self, store, records, results, block_days, probability_cells):
        from gnss_field.evaluation import holm_adjust, probability_gate
        formal, claims = {'block_days': block_days}, {}
        if 'point' in records:
            fine = self._fine_bootstrap(store, records['point'], records['bstar'], block_days=block_days)
            formal['fine'] = fine
            pvalues = []
            for comparison in fine['comparisons'].values():
                draws = np.asarray(comparison['replicate_differences_mm'])
                pvalues.append(float((1+(draws >= 0).sum())/(len(draws)+1)))
            claims['fine'] = max(pvalues)
        if 'joint' in records and 'gaussian' in records:
            a, b = probability_cells
            scores = {key: self._cell_bootstrap(records['joint']['hours'], a[key], b[key],
                      crossed=key in ('es', 'vs'), block_days=block_days) for key in a}
            nominal = {key: self._cell_bootstrap(records['joint']['hours'], a[key],
                       np.where(np.isfinite(a[key]), int(key)/100, np.nan), block_days=block_days)
                       for key in ('80', '90', '95')}
            for value in nominal.values():
                value['direction'] = 'model_coverage_minus_nominal'
            gate = probability_gate(es_difference=scores['es']['difference'], vs_difference=scores['vs']['difference'],
                     crps_difference=scores['crps']['difference'], intervals=results['joint']['metrics']['primary']['intervals'],
                     coverage_difference_ci={k: nominal[k]['ci95'] for k in ('80', '90', '95')}, formal=True)
            formal['probability'] = dict(comparisons=scores, coverage_minus_nominal=nominal, gate=gate,
                                         comparison_direction='model_minus_gaussian', population='F intersect S0',
                                         intervals=results['joint']['metrics']['primary']['intervals'])
            claims['joint'] = max((1+sum(x >= 0 for x in scores[key]['replicate_differences']))/
                                   (len(scores[key]['replicate_differences'])+1) for key in ('es', 'vs'))
        names = list(claims)
        adjusted = holm_adjust([claims[n] for n in names]) if names else []
        formal['selected_product_claims'] = {
            n: dict(p_value=claims[n], holm_p_value=float(p),
                    confirmed=bool(p < .05 and (formal['fine']['gate']['fine_pass'] if n == 'fine'
                        else formal['probability']['gate']['calibrated_joint_pass'])))
            for n, p in zip(names, adjusted)}
        return formal

    def export(self):
        lock = self._validated_lock()
        if not (self.root/'evaluate2020'/'summary.json').exists():
            raise FileNotFoundError('Export requires completed retrospective station evaluation')
        store, directory = self._store('context'), self.root/'export'
        hours = self._hours(2020, store)
        # Event hours are selected entirely from deployed A inputs under the locked rule.
        days = np.unique(hours//24)
        event_scores = []
        for day in days:
            selected = hours[hours//24 == day]
            values = store.context_values(selected)[..., 0]
            score = float(np.nanmean(np.abs(np.diff(values, axis=0)))) if len(values) > 1 else 0.
            event_scores.append((score, int(day)))
        event_days = {d for _, d in sorted(event_scores, reverse=True)[:12]}
        paths = {}
        for product, entry in lock['products'].items():
            model = self._model(entry['model'])
            model.load_state_dict(_load(entry['checkpoint'])['ema'])
            model.eval().requires_grad_(False)
            months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
            for month in sorted(set(months)):
                self._pause()
                path = directory/product/f'{month}.npz'
                if path.exists():
                    meta = read_json(path.with_suffix('.json'))
                    if meta['lock'] != lock['lock_sha256'] or sha256(path) != meta['sha256']:
                        raise ValueError('Export checkpoint/lock/content mismatch')
                    paths[f'{product}_{month}'] = path
                    continue
                selected = hours[months == month]
                mean, std, support, events, event_hours = [], [], [], [], []
                for hour in selected:
                    self._pause()
                    with torch.no_grad():
                        context = self._context([hour])
                        forecast = model.predict_field(context, self._get_decoder(), members=lock['stability']['members'],
                                                        seed=self.science.sampling_seed, science=self.science)
                        mean.append(forecast.mean.cpu().numpy()[0])
                        support.append(context.maps[0, 3].bool().cpu().numpy())
                        std.append(forecast.samples.std(1, unbiased=False).cpu().numpy()[0] if forecast.samples is not None
                                   else np.zeros_like(mean[-1]))
                        if forecast.samples is not None and hour//24 in event_days:
                            event_hours.append(hour)
                            events.append(forecast.samples.cpu().numpy()[0])
                buffer = io.BytesIO()
                np.savez_compressed(buffer, hours=selected, mean=np.asarray(mean), std=np.asarray(std),
                                    event_hours=np.asarray(event_hours), event_members=np.asarray(events),
                                    support=np.asarray(support), land_mask=store.grid['region'], grid_id=store.grid_id)
                atomic_bytes(path, buffer.getvalue())
                write_json(path.with_suffix('.json'), dict(lock=lock['lock_sha256'], sha256=sha256(path),
                           units='Z0_mm', checkpoint_sha256=entry['checkpoint_sha256']))
                paths[f'{product}_{month}'] = path
        write_json(directory/'manifest.json', dict(lock=lock['lock_sha256'], files={k: str(v) for k, v in paths.items()},
                                                  event_days=sorted(event_days)))
        return {'manifest': directory/'manifest.json'}
