"""Differentiable mosaic/query operators and input-only support geometry."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree, Delaunay, QhullError


def mosaic(patches, origins, shape):
    single = patches.ndim == 4
    if single:
        patches = patches.unsqueeze(0)
    b, p, c, h, w = patches.shape
    if h != w or len(origins) != p:
        raise ValueError('Patch geometry mismatch')
    axis = torch.arange(h, device=patches.device, dtype=patches.dtype)
    win = torch.sin(torch.pi * (axis + .5) / h).square()
    weight = win[:, None] * win[None, :]
    origins = np.asarray(origins, dtype=int)
    stride = h // 2
    expected = np.array([(y, x) for y in range(0, shape[0] - h + 1, stride)
                         for x in range(0, shape[1] - w + 1, stride)])
    if np.array_equal(origins, expected):
        cols = (patches * weight).permute(0, 2, 3, 4, 1).reshape(b, c * h * w, p)
        result = F.fold(cols, tuple(shape), h, stride=stride)
        denom = F.fold(weight.reshape(1, h * w, 1).expand(1, -1, p), tuple(shape), h, stride=stride)
    else:
        result = patches.new_zeros(b, c, *shape)
        denom = patches.new_zeros(1, 1, *shape)
        for j, (y, x) in enumerate(origins):
            result[:, :, y:y+h, x:x+w] = result[:, :, y:y+h, x:x+w] + patches[:, j] * weight
            denom[:, :, y:y+h, x:x+w] += weight
    if (denom <= 0).any():
        raise ValueError('Mosaic has uncovered cells')
    out = result / denom
    return out[0] if single else out


def query_grid(field, xy):
    """Return [...,N] from [...,H,W]; xy is (column,row) in grid centres."""
    h, w = field.shape[-2:]
    xy = torch.as_tensor(xy, device=field.device, dtype=field.dtype)
    if not torch.isfinite(xy).all() or (xy < 0).any() or (xy[:, 0] > w-1).any() or (xy[:, 1] > h-1).any():
        raise ValueError('Nonfinite or outside-grid query')
    x = xy[:, 0].floor().long().clamp(max=w-2)
    y = xy[:, 1].floor().long().clamp(max=h-2)
    dx, dy = xy[:, 0] - x, xy[:, 1] - y
    return (field[..., y, x]*(1-dx)*(1-dy) + field[..., y, x+1]*dx*(1-dy)
            + field[..., y+1, x]*(1-dx)*dy + field[..., y+1, x+1]*dx*dy)


def surface_query(field, xy, heights):
    heights = torch.as_tensor(heights, dtype=field.dtype, device=field.device)
    return query_grid(field, xy) * torch.exp(-heights / 3000.)


def basis_observation_matrix(phi, shape, xy, heights):
    fields = phi.T.reshape(phi.shape[1], *shape)
    return surface_query(fields, xy, heights).T


def coarsen_100km(field):
    if any(n % 4 for n in field.shape[-2:]):
        raise ValueError('Grid must be aligned to the global 4x4 origin')
    shape = field.shape
    x = field.reshape(-1, 1, *shape[-2:])
    x = F.interpolate(F.avg_pool2d(x, 4), size=shape[-2:], mode='bilinear', align_corners=False)
    return x.reshape(shape)


def safe_coarsening_mask(land):
    x = torch.as_tensor(land, dtype=torch.float64)
    # A positive interpolation weight on an incomplete coarse cell is unsafe.
    pooled = F.avg_pool2d(x[None, None], 4)
    complete = (pooled == 1).double()
    restored = F.interpolate(complete, size=x.shape, mode='bilinear', align_corners=False)[0, 0]
    return (restored == 1) & (x == 1)


def curvature_matrix(phi, shape, support):
    m = phi.reshape(*shape, phi.shape[-1])
    mask = torch.as_tensor(support, device=phi.device, dtype=torch.bool)
    good = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    lap = m[:-2, 1:-1] + m[2:, 1:-1] + m[1:-1, :-2] + m[1:-1, 2:] - 4*m[1:-1, 1:-1]
    return lap[good]


def curvature_loss(field, support, scale):
    if field.ndim == 4:
        field = field[:, 0]
    if support.ndim == 2:
        support = support.unsqueeze(0).expand(len(field), -1, -1)
    lap = field[:, :-2, 1:-1] + field[:, 2:, 1:-1] + field[:, 1:-1, :-2] + field[:, 1:-1, 2:] - 4*field[:, 1:-1, 1:-1]
    good = support[:, :-2, 1:-1] & support[:, 2:, 1:-1] & support[:, 1:-1, :-2] & support[:, 1:-1, 2:] & support[:, 1:-1, 1:-1]
    return (lap[good] / scale).square().mean() if good.any() else field.sum()*0


def input_geometry(input_xy, queries):
    input_xy, queries = np.asarray(input_xy), np.asarray(queries)
    n = len(queries)
    if len(input_xy) == 0:
        return dict(indices=np.zeros((n, 1), int), weights=np.zeros((n, 1)),
                    support=np.zeros(n, bool), nearest=np.full(n, 1e6), count=np.zeros(n, int))
    tree = cKDTree(input_xy)
    distance, indices = tree.query(queries, k=min(4, len(input_xy)))
    if distance.ndim == 1:
        distance, indices = distance[:, None], indices[:, None]
    count = np.asarray(tree.query_ball_point(queries, 150000, return_length=True))
    hull = np.zeros(n, bool)
    if len(input_xy) >= 3:
        try:
            hull = Delaunay(input_xy).find_simplex(queries) >= 0
        except QhullError:
            pass
    weight = np.where(distance <= 150000, 1 / np.maximum(distance, 1.)**2, 0.)
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-30)
    return dict(indices=indices, weights=weight,
                support=(distance[:, 0] <= 100000) & (count >= 3) & hull,
                nearest=distance[:, 0], count=count)


def patch_dictionary(context, origins, grid_origin, mu_scale=1.):
    """Flatten hour x patch with all A tokens padded outside each 150km halo."""
    origins = np.asarray(origins, int)
    out = {key: [] for key in ('stations', 'padding', 'coarse', 'maps', 'baseline', 'mu_condition', 'baseline_condition', 'domain')}
    b = len(context.hours)
    for y, x in origins:
        centre = context.stations.new_tensor(np.asarray(grid_origin) + np.array([x+12, y+12])*25000) / 400000
        stations = context.stations.clone()
        stations[..., 10:12] -= centre
        inside = (stations[..., 10:12].abs() <= 450000/400000).all(-1)
        padding = context.padding | ~inside
        stations = stations.masked_fill(padding[..., None], 0)
        maps = context.maps[:, :, y:y+24, x:x+24].clone()
        maps[:, 1] -= centre[0]
        maps[:, 2] -= centre[1]
        coarse = context.coarse[:, :, :, y//4:y//4+6, x//4:x//4+6].clone()
        coarse[:, :, 7] -= centre[0]
        coarse[:, :, 8] -= centre[1]
        baseline = F.avg_pool2d(context.baseline[:, :, y:y+24, x:x+24], 4)
        # Context baseline is upsampled from the common coarse grid; slice coarse raw exactly.
        coarse_raw = context.coarse_z0
        if coarse_raw is not None:
            baseline = coarse_raw[:, :, y//4:y//4+6, x//4:x//4+6]
        basefine = F.interpolate(baseline, (24, 24), mode='bilinear', align_corners=False)
        mu = basefine if context.mean is None else context.mean[:, :, y:y+24, x:x+24]
        for key, value in dict(stations=stations, padding=padding, coarse=coarse, maps=maps,
                               baseline=baseline, baseline_condition=basefine/mu_scale,
                               mu_condition=mu/mu_scale, domain=torch.ones_like(basefine)).items():
            out[key].append(value)
    # patch-major ordering is explicit; the caller reshapes P,B before mosaicking.
    return {key: torch.cat(value, 0) for key, value in out.items()}
