"""Frozen local pilot inputs; no residual observation is an inference feature."""
import numpy as np
import torch
from local_probe_diagnostics import PROJECT, ProbeData, save_npz
from local_shared_field import balanced_blocks
from local_joint_field import local_bases, observation_factors
from local_conditional_gaussian import wet_feature
from gnss_field.artifacts import read_json, sha256
from gnss_field.models.diffusion import keyed_noise
from gnss_field.models.subspace import station_members

ROOT = PROJECT/'results/20261119_local_edm_subspace_v01'


def fit_condition_pca(features, training, rank=32):
    x = np.asarray(features, 'f8')[training]
    center, scale = x.mean(0), x.std(0)
    active = scale > 1e-5
    scale = scale.clip(1e-5)
    normalized = np.where(active, (x-center)/scale, 0.)
    _, singular, right = np.linalg.svd(normalized, full_matrices=False)
    projection = right[:rank].T
    common_scale = float(np.sqrt(np.mean((normalized@projection)**2)))
    return dict(center=center, scale=scale, active=active, projection=projection,
                common_scale=np.array(common_scale), singular_values=singular)


def transform_condition(features, state):
    x = np.asarray(features, 'f8')
    x = np.where(state['active'], (x-state['center'])/state['scale'], 0.)
    return ((x@state['projection'])/state['common_scale']).astype('f4')


class PilotData:
    def __init__(self, device='cuda'):
        self.device = device
        self.data = ProbeData(device=device)
        self.b = self.data.arrays(2018)
        self.train, self.development = balanced_blocks(self.b['hours'])
        assert (self.train.sum(), self.development.sum()) == (576, 288)
        self.covariance_path = PROJECT/'results/20261119_local_fused_probability_v02/B_only_parameters.lock.json'
        self.scale_path = PROJECT/'results/20261119_local_conditional_probability_v01/B_only_parameters.lock.json'
        base = read_json(self.covariance_path)
        self.sigma = np.array(base['model']['std'])
        self.weight = base['weight']
        self.amplitude = read_json(self.scale_path)
        self.v_path = PROJECT/'results/20261119_local_method_research_v02/observable_subspace.npz'
        self.v = np.load(self.v_path)['right_vectors'][:64]
        np.testing.assert_allclose(self.v@self.v.T, np.eye(64), atol=1e-12)
        self.bases = local_bases(self.data.store)
        self.pca = fit_condition_pca(self.b['features'][:, :-16], self.train)
        ROOT.mkdir(parents=True, exist_ok=True)
        save_npz(ROOT/'condition_pca.npz', **self.pca)
        self.contract = dict(foundation=self.data.contract, covariance=sha256(self.covariance_path),
            amplitude=sha256(self.scale_path), projection=sha256(self.v_path), pca=sha256(ROOT/'condition_pca.npz'),
            schema='1740_Avalues_flags+960_coarse_pool+4_cycles; no M0 feature, PCA32 scale-only common normalization',
            source=sha256(__file__))
        self.years = {}

    def arrays(self, year):
        if year in self.years:
            return self.years[year]
        a = self.data.arrays(year)
        store = self.data.store if year == 2018 else self.data.development
        ids = a['station_ids']
        factors = observation_factors(store, ids, self.bases)
        h = np.concatenate([s*m for s, m in zip(self.sigma, factors)], 1)
        projection = h@self.v.T
        remainder = h-projection@self.v
        wet, present = wet_feature(store.context_values(a['hours']))
        x = np.where(present, (wet-self.amplitude['A_wetness_center'])/self.amplitude['A_wetness_std'], 0.)
        amplitude = np.exp(self.amplitude['a']+self.amplitude['b']*x)
        arrays = dict(hours=a['hours'], station_ids=ids, store=store,
            truth=torch.tensor(a['truth'], device=self.device), valid=torch.tensor(a['valid'], device=self.device),
            condition=torch.tensor(transform_condition(a['features'][:, :-16], self.pca), device=self.device),
            mean=torch.tensor(a['baseline']+self.weight*(a['idw']-a['baseline']), device=self.device),
            amplitude=torch.tensor(amplitude, dtype=torch.float32, device=self.device),
            projection=torch.tensor(projection, dtype=torch.float32, device=self.device),
            remainder=torch.tensor(remainder, dtype=torch.float32, device=self.device))
        self.years[year] = arrays
        return arrays

    def samples(self, model, arrays, indices, members, *, seed=None, gaussian=False):
        b = len(indices)
        condition = arrays['condition'][indices]
        if seed is None:
            initial = torch.randn(b, members, 64, device=self.device)
            remainder = torch.randn(b, members, self.v.shape[1], device=self.device)
            error = torch.randn(b, members, len(arrays['station_ids']), device=self.device)
        else:
            hours = arrays['hours'][indices]
            initial = keyed_noise(hours, members, seed, shape=(64,), device=self.device)
            remainder = keyed_noise(hours, members, seed+1000, shape=(self.v.shape[1],), device=self.device)
            all_error = keyed_noise(hours, members, seed+2000, shape=(905,), device=self.device)
            error = all_error[..., arrays['station_ids']]
        latent = initial if gaussian else model.sample_noise(condition, initial)
        return station_members(arrays['mean'][indices], latent, remainder, error,
            arrays['projection'], arrays['remainder'], arrays['amplitude'][indices], self.sigma[-1])
