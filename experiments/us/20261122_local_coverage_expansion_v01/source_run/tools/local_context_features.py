"""Input-only local aggregation and vertex-first field heads."""
import numpy as np
import torch
from torch import nn


def kernel_aggregate(values, kernel, valid=None):
    values = np.asarray(values, dtype='f8')
    ok = np.isfinite(values).all(-1)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    weights = np.asarray(kernel, dtype='f8')*ok[None]
    density = weights.sum(-1)
    clean = np.where(ok[:, None], values, 0.)
    result = (weights @ clean)/np.maximum(density[:, None], 1e-30)
    return result.astype('f4'), density.astype('f4')


def fit_normalizer(features, training):
    # Accumulate one hour at a time; no copy of the whole feature cube.
    count, total, squares = 0, None, None
    for index in training:
        x = np.asarray(features[index], dtype='f8')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training features')
        total = x.sum(0) if total is None else total+x.sum(0)
        squares = (x*x).sum(0) if squares is None else squares+(x*x).sum(0)
        count += len(x)
    center = total/count
    std = np.sqrt(np.maximum(squares/count-center*center, 0))
    std[std < 1e-5] = 1.
    return center.astype('f4'), std.astype('f4')


class VertexQuery(nn.Module):
    """Query a compact vertex set; outside stations need explicit safe geometry."""
    def __init__(self, shape, xy, heights, nodes):
        super().__init__()
        xy = np.asarray(xy)
        x, y = np.floor(xy).astype(int).T
        if ((x < 0)|(y < 0)|(x >= shape[1]-1)|(y >= shape[0]-1)).any():
            raise ValueError('Incomplete query geometry')
        corners = np.stack([y*shape[1]+x, y*shape[1]+x+1,
                            (y+1)*shape[1]+x, (y+1)*shape[1]+x+1], -1)
        lookup = {int(n): i for i, n in enumerate(nodes)}
        indices = np.array([[lookup[int(n)] for n in row] for row in corners])
        dx, dy = (xy-np.floor(xy)).T
        weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], -1)
        weights *= np.exp(-np.asarray(heights)[:, None]/3000)
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, field):
        return (field[..., self.indices]*self.weights).sum(-1)


def project_operator(basis, land):
    q, r = np.linalg.qr(np.asarray(basis, dtype='f8')[land], mode='reduced')
    if np.linalg.matrix_rank(r) != r.shape[0]:
        raise ValueError('Spatial basis is rank deficient on land')
    return np.linalg.solve(r, q.T)


class LocalGridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(99, 64), nn.SiLU(), nn.Linear(64, 32),
                                 nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class LocalFeatureBuilder:
    """Pure A-only constructor; no target-label reader is accepted by this API."""
    def __init__(self, grid_xy, a_xy, a_height, terrain, land, nodes, stats, a_qxy, a_query_valid):
        from scipy.spatial.distance import cdist
        from scipy.ndimage import uniform_filter
        self.nodes = np.asarray(nodes)
        self.grid_xy = np.asarray(grid_xy)[nodes]
        self.a_xy = np.asarray(a_xy)
        self.a_height = np.asarray(a_height)
        self.height = np.asarray(terrain).ravel()[nodes]
        self.land = np.asarray(land).ravel()[nodes]
        self.mean, self.std = np.asarray(stats['input_mean']), np.asarray(stats['input_std'])
        self.scale = float(stats['z0_std'])
        distance = cdist(np.asarray(grid_xy)[nodes], a_xy)
        self.distance = distance
        self.kernels = [np.where(distance <= 3*s, np.exp(-.5*(distance/s)**2), 0.)
                        for s in (75000., 150000.)]
        terrain = np.asarray(terrain, dtype='f8')
        relief = np.sqrt(np.maximum(uniform_filter(terrain**2, 3)-uniform_filter(terrain, 3)**2, 0))
        dy, dx = np.gradient(terrain)
        self.terrain = np.stack([terrain/3000, relief/3000, dx/3000, dy/3000], -1).reshape(-1, 4)[nodes]
        self.a_qxy, self.a_query_valid = a_qxy, np.asarray(a_query_valid, bool)

    def build(self, frames, m0_grid, pca, cycle):
        from gnss_field.geometry import query_grid
        frames = np.asarray(frames)
        if frames.shape[0] != 4 or frames.shape[-1] != 5:
            raise ValueError('Four frames and five jointly-valid variables are required')
        columns = []
        for raw in frames:
            valid = np.isfinite(raw).all(-1)
            normalized = (raw-self.mean)/self.std
            for kernel in self.kernels:
                value, density = kernel_aggregate(normalized, kernel, valid)
                columns.extend([value, density[:, None]])
        current_valid = np.isfinite(frames[0]).all(-1)
        m0_at_a = query_grid(torch.tensor(np.asarray(m0_grid), dtype=torch.float32), self.a_qxy).numpy()
        innovation = ((frames[0, :, 0]-m0_at_a)/self.scale)[:, None]
        for kernel in self.kernels:
            value, density = kernel_aggregate(innovation, kernel, current_valid & self.a_query_valid)
            columns.extend([value, density[:, None]])
        columns.append(self.terrain)
        for kernel in self.kernels:
            ah, _ = kernel_aggregate(self.a_height[:, None], kernel, current_valid)
            # Empty-neighborhood height difference remains zero.
            present = (kernel*current_valid[None]).sum(-1)>0
            columns.append(np.where(present, (ah[:, 0]-self.height)/3000, 0.)[:, None])
        # These amplitudes use the same grid as the eventual publication operator.
        from gnss_field.training.baselines import _idw
        idw = _idw(self.a_xy[current_valid], frames[0, current_valid, 0], self.grid_xy, self.mean[0])
        columns.extend([np.asarray(m0_grid).ravel()[self.nodes, None]/self.scale,
                        idw[:, None]/self.scale])
        columns.append(np.broadcast_to(cycle, (len(self.nodes), 4)))
        nearest = np.min(self.distance[:, current_valid], axis=1) if current_valid.any() else np.full(len(self.nodes), 400000.)
        count = ((self.distance <= 150000)*current_valid[None]).sum(-1)
        columns.append(np.c_[self.land, np.minimum(nearest/400000, 2.5), count/16])
        columns.append(np.broadcast_to(pca, (len(self.nodes), 32)))
        result = np.concatenate(columns, -1).astype('f4')
        if result.shape[1] != 99 or not np.isfinite(result).all():
            raise ValueError('Invalid local feature schema or values')
        return result
