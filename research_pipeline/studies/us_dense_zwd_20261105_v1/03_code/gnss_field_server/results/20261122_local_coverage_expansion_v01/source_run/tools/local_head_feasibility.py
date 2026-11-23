"""Numerical spike for the proposed local head; no fitting or accuracy claim."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from gnss_field.geometry import surface_query, query_grid
from gnss_field.artifacts import write_json, sha256
from architecture_bottleneck_diagnostics import PROJECT, ROOT, PILOT


def main():
    torch.set_num_threads(2); torch.manual_seed(42)
    spatial = dict(np.load(PILOT/'server_bundle/spatial.npz'))
    c = dict(np.load(PILOT/'server_evaluation/data.npz'))
    metadata = pd.read_csv(PILOT/'server_evaluation/stations.csv')
    active = c['valid'].any(0)
    xy = metadata[['x','y']].to_numpy()[active]
    xy = (xy-np.array([spatial['x'][0,0],spatial['y'][0,0]]))/25000
    heights = metadata.height.to_numpy()[active]
    h,w = spatial['shape']
    features = torch.randn(h,w,99)
    head = nn.Sequential(nn.Linear(99,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    values = head(features).squeeze(-1)*50
    reference = surface_query(values,xy,heights)
    q = torch.tensor(xy,dtype=torch.float32); x=q[:,0].floor().long(); y=q[:,1].floor().long()
    dx,dy=q[:,0]-x,q[:,1]-y
    corners = torch.stack([features[y,x],features[y,x+1],features[y+1,x],features[y+1,x+1]],1)
    weight = torch.stack([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy],1)
    shortcut = (head(corners).squeeze(-1)*weight).sum(1)*50*torch.exp(-torch.tensor(heights)/3000)
    maximum = float((shortcut-reference).abs().max().detach())
    wrong_features = query_grid(features.permute(2,0,1),xy).T
    wrong = head(wrong_features).squeeze(-1)*50*torch.exp(-torch.tensor(heights)/3000)
    wrong_error = float((wrong-reference).abs().max().detach())
    assert maximum < .001 and wrong_error > .001
    with torch.no_grad():
        nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        assert head(features).abs().max()==0
    s = spatial['phi'].astype('f8')@spatial['v'].astype('f8').T
    land = spatial['land'].ravel().astype(bool)
    qbasis, rbasis = np.linalg.qr(s[land], mode='reduced')
    singular = np.linalg.svd(rbasis,compute_uv=False)
    assert np.linalg.matrix_rank(rbasis)==64
    projector = np.linalg.solve(rbasis,qbasis.T)
    random_field = torch.randn(h*w,dtype=torch.float64,requires_grad=True)*50
    coefficients = torch.tensor(projector)@random_field[torch.tensor(land)]
    field = (torch.tensor(s)@coefficients).reshape(h,w)
    query_basis = surface_query(torch.tensor(s.T.reshape(64,h,w)),xy,heights).T
    projection_reference = surface_query(field,xy,heights)
    projection_shortcut = query_basis@coefficients
    projection_error = float((projection_shortcut-projection_reference).abs().max().detach())
    assert projection_error < .001
    gradient = torch.autograd.grad(projection_shortcut.sum(),random_field)[0].numpy()
    expected = (query_basis.numpy()@projector).sum(0)
    np.testing.assert_allclose(gradient[land],expected,atol=1e-9,rtol=1e-8)
    result = dict(source_sha256=sha256(__file__),parameters=sum(p.numel() for p in head.parameters()),
        grid_corner_head_query_max_mm=maximum,wrong_interpolate_features_then_head_max_mm=wrong_error,
        zero_residual_initialization=True,R64_projection_query_max_mm=projection_error,
        projection_gradient_full_land_integral=True,land_points=int(land.sum()),
        R64_grid_condition_number=float(singular.max()/singular.min()),
        interpretation='Numerical feasibility on real query geometry and fixed basis, random features; no neural training or accuracy evidence')
    write_json(ROOT/'head_feasibility.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
