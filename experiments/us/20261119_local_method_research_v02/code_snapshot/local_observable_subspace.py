"""Fixed-prior observation-sensitive sizing, preserving the Gaussian remainder."""
import numpy as np
import torch
from local_probe_diagnostics import ProbeData, PROJECT, emit
from local_joint_field import local_bases, observation_factors
from gnss_field.artifacts import read_json, write_json, sha256
from scipy.ndimage import binary_erosion


def main():
    torch.set_num_threads(2)
    root = PROJECT/'results/20261119_local_method_research_v02'
    data = ProbeData(device='cpu')
    b = data.arrays(2018)
    covariance_path = PROJECT/'results/20261119_local_fused_probability_v02/B_only_parameters.lock.json'
    parameters = read_json(covariance_path)['model']['std']
    bases = local_bases(data.store)
    matrices = observation_factors(data.store, b['station_ids'], bases)
    # Availability weights use B2018 qualification only, not observation magnitudes.
    available = b['valid'].mean(0)
    design = np.concatenate([s*h for s, h in zip(parameters, matrices)], axis=1)
    whitened = np.sqrt(available)[:, None]*design/parameters[-1]
    _, singular, right = np.linalg.svd(whitened, full_matrices=False)
    energy = np.cumsum(singular**2)/(singular**2).sum()
    rows = []
    for rank in (32, 64, 128):
        v = right[:rank]
        projected = design@v.T
        remaining = design-projected@v
        # Gaussian generated modes + orthogonal Gaussian remainder exactly
        # reproduce the original station covariance, including modes not learned.
        reconstructed = projected@projected.T+remaining@remaining.T
        rows.append(dict(rank=rank, B_weighted_operator_energy_fraction=float(energy[rank-1]),
                         covariance_reconstruction_max_mm2=float(abs(reconstructed-design@design.T).max())))
    write_json(root/'observable_subspace.json', dict(source=sha256(__file__),
        inputs='B2018-fitted covariance prior, B geometry and qualification; SVD does not refit observation magnitudes',
        covariance_parameters_sha256=sha256(covariance_path),
        latent_prior_dimension=design.shape[1], nonzero_B_stations=int((available>0).sum()),
        singular_values=singular, modes_singular_value_above1=int((singular>1).sum()), candidates=rows,
        interpretation='Operator energy is not evidence of recoverable field detail; no generator trained',
        proposal='Z0=mu_mix+Psi[V_R.T z_theta+(I-V_R.T V_R)epsilon]; retain Gaussian remainder; at z~N(0,I) prior covariance is unchanged'))
    np.savez_compressed(root/'observable_subspace.npz', right_vectors=right[:128], singular_values=singular,
                        B_station_ids=b['station_ids'], availability=available)
    # C geometry is evaluated only after fixing V; it never rotates the basis.
    v = right[:64]
    c_ids = data.development.ids('C')
    c_matrices = observation_factors(data.development, c_ids, bases)
    c_design = np.concatenate([s*h for s, h in zip(parameters, c_matrices)], 1)
    grid_design = np.concatenate([s*basis for s, basis in zip(parameters, bases)], 1)
    def fraction(a):
        return np.sum((a@v.T)**2, 1)/np.maximum(np.sum(a*a, 1), 1e-30)
    def summary(value):
        return dict(n=len(value), quantiles=np.quantile(value, [0., .25, .5, .75, 1.]), mean=float(value.mean()))
    b_fraction, c_fraction, grid_fraction = fraction(design), fraction(c_design), fraction(grid_design)
    land = data.store.grid['region'].astype(bool)
    boundary = land & ~binary_erosion(land, iterations=4)
    c_cache = data.arrays(2019)
    support = c_cache['support'].mean(0) >= .5
    write_json(root/'subspace_geometry_coverage.json', dict(source=sha256(__file__), rank=64,
        interpretation='Learnable fraction of latent Gaussian field variance, excluding tau; no C labels fitted or scored',
        summaries=dict(B_active=summary(b_fraction[available > 0]), C_query_valid=summary(c_fraction[data.development.query_valid[c_ids]]),
            land=summary(grid_fraction[land.ravel()]), boundary_100km=summary(grid_fraction[boundary.ravel()]),
            supported_at_least_half_C_subset_hours=summary(grid_fraction[(support & land).ravel()])),
        prior_remainder='Remaining latent-field variance stays Gaussian; no difficult station removed'))
    emit(str(rows))


if __name__ == '__main__':
    main()
