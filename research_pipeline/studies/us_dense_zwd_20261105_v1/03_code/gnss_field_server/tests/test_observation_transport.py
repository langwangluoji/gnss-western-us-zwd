import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.terrain_diffusion.transport import ObservationSummary,TransportModel


def sample():
    c=torch.zeros(1,12,3,38);c[...,3]=1
    # Surface ZWD corresponding to the same normalised-at-sea-level field.
    heights=torch.tensor([0.,1.,2.]);c[...,0]=100*torch.exp(-heights/3)
    return dict(context=c,cgeo=torch.tensor([[[0.,0.,0.,0.],[10.,0.,1.,0.],[20.,0.,2.,0.]]]),
                query=torch.zeros(1,2,41),qgeo=torch.tensor([[[5.,0.,.5,0.],[15.,0.,1.5,0.]]]),
                history=torch.zeros(1,6,2,3),target=torch.zeros(1,2),mask=torch.ones(1,2,dtype=torch.bool))


def test_height_transport_and_missing():
    d=sample();m=ObservationSummary('transport',0.,1.)
    y=m(d).reshape(1,2,9,3)
    expected=100*torch.exp(-d['qgeo'][...,2]/3)
    torch.testing.assert_close(y[...,0],expected[...,None].expand(1,2,9),rtol=1e-5,atol=1e-5)
    d['context'][:,:,1,3]=0;d['context'][:,:,1,0]=99999
    torch.testing.assert_close(m(d).reshape(1,2,9,3)[...,0],expected[...,None].expand(1,2,9),rtol=1e-5,atol=1e-5)


def test_summary_is_query_independent_and_all_missing_finite():
    d=sample();m=ObservationSummary('transport',0.,1.);y=m(d)
    e=dict(d);e['qgeo']=d['qgeo'][:,:1];e['query']=d['query'][:,:1]
    torch.testing.assert_close(y[:,:1],m(e))
    d['context'][...,3]=0
    assert torch.isfinite(m(d)).all() and torch.count_nonzero(m(d))==0


def test_transport_model_sampling_and_label_isolation():
    d=sample();m=TransportModel('transport',0.,1.,width=16,blocks=1).eval()
    noise=torch.randn(1,2)
    with torch.no_grad():
        p=m.sample(d,noise,2);d['target']+=50000
        torch.testing.assert_close(p,m.sample(d,noise,2))
    p=m.sample(d,noise,2,checkpoint=True);p.square().mean().backward()
    assert m.summary.scale_logit.grad is not None and torch.isfinite(m.summary.scale_logit.grad)


def test_calendar_names_match_scope():
    from tools.terrain_diffusion.transport_evaluate import calendar_name
    from tools.terrain_diffusion.transport_report import require_full_calendar
    import pytest
    assert calendar_name(False,24)=='quarterly24'
    assert calendar_name(True,4)=='all_months4'
    require_full_calendar(dict(rows=list(range(12)),hours=24))
    with pytest.raises(ValueError):require_full_calendar(dict(rows=[0,3,6,9],hours=24))


def test_conditioner_transfer_complete_and_decoder_fresh():
    from tools.terrain_diffusion.transport import transfer_conditioner
    source=TransportModel('transport',0,1,width=16,blocks=1)
    target=TransportModel('transport',0,1,width=16,blocks=1)
    before=target.head[1].weight.detach().clone()
    transfer_conditioner(target,source.state_dict())
    for key,value in source.state_dict().items():
        if key.startswith(('context.','spatial.','temporal.','summary.','query.')):
            torch.testing.assert_close(value,target.state_dict()[key])
    torch.testing.assert_close(before,target.head[1].weight)


def test_exact_gaussian_crps():
    import numpy as np
    from tools.terrain_diffusion.transport_gate import gaussian_crps
    score=gaussian_crps(np.array([0.]),np.array([1.]),np.array([0.]))
    np.testing.assert_allclose(score,[(np.sqrt(2)-1)/np.sqrt(np.pi)],atol=1e-10)
