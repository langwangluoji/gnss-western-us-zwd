import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from tools.terrain_diffusion.data import parse_solution, require_role
from tools.terrain_diffusion.model import DiffusionModel, schedule, v_target, recover


def test_allowed_columns_independent():
    a=' ABCD 16:001:00000 2300 1 99999 100 2200 2300 -999 280 950 nan\n'
    b=a.replace('99999','nan').replace('-999','12345')
    h,x,meta=parse_solution(a,'ABCD')
    h2,y,_=parse_solution(b,'ABCD')
    np.testing.assert_equal(h,h2);np.testing.assert_equal(x,y)
    np.testing.assert_equal(x[0],[100,280,950])
    _,bad,_=parse_solution(a.replace('280','999'),'ABCD')
    assert np.isnan(bad[0,1]) and bad[0,0]==100


def test_roles_are_closed():
    require_role('training');require_role('validation')
    for role in ['confirmation','legacy_C','legacy_D']:
        with pytest.raises(PermissionError):require_role(role)


def test_v_roundtrip():
    ab=schedule();x=torch.randn(5,7);noise=torch.randn_like(x);a=ab[torch.tensor([1,10,50,90,100])][:,None]
    xt=a.sqrt()*x+(1-a).sqrt()*noise
    x0,eps=recover(xt,v_target(x,noise,a),a)
    torch.testing.assert_close(x0,x);torch.testing.assert_close(eps,noise)
    assert ab[0]==1 and torch.all(ab[:-1]>ab[1:])


def inputs():
    torch.manual_seed(3)
    return dict(context=torch.randn(1,12,5,38),cgeo=torch.randn(1,5,4),
                query=torch.randn(1,4,41),qgeo=torch.randn(1,4,4),history=torch.zeros(1,6,4,3))


def test_permutation_and_causality():
    m=DiffusionModel(width=16,blocks=1).eval();d=inputs()
    with torch.no_grad():
        h=m.encode(d['context'],d['cgeo'])
        changed=d['context'].clone();changed[:,6:]+=17
        h2=m.encode(changed,d['cgeo'])
        torch.testing.assert_close(h[:,:6],h2[:,:6])
        x=torch.randn(1,4);step=torch.tensor([50]);p=m(x,step,d,h)
        perm=torch.tensor([2,0,3,1]);e=dict(d)
        e['query']=d['query'][:,perm];e['qgeo']=d['qgeo'][:,perm];e['history']=d['history'][:,:,perm]
        torch.testing.assert_close(m(x[:,perm],step,e,h),p[:,perm],atol=2e-5,rtol=2e-5)


def test_sampler_gradients():
    m=DiffusionModel(width=16,blocks=1).eval();d=inputs()
    out=m.sample(d,torch.randn(1,4),steps=3,checkpoint=True)
    out.square().mean().backward()
    g=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    assert np.isfinite(g) and g>0


def test_metrics_known_population():
    from tools.terrain_diffusion.metrics import evaluate
    truth=np.array([[1.,2.],[3.,np.nan]])
    samples=np.stack([truth+1,truth+1]);samples[:,1,1]=999
    result=evaluate(samples,truth)
    assert result['station_macro_rmse_mm']==1
    assert result['fair_crps_mm']==1 and result['valid_count']==3


def test_temporal_chain_and_pretraining():
    from tools.terrain_diffusion.train import trajectory_loss,transfer_encoder
    m=DiffusionModel(width=16,blocks=1);target=DiffusionModel(width=16,blocks=1)
    transfer_encoder(target,m.state_dict())
    torch.testing.assert_close(m.context[0].weight,target.context[0].weight)
    d=inputs();d['target']=torch.randn(1,4);d['mask']=torch.ones(1,4,dtype=torch.bool)
    loss,seq=trajectory_loss(m,[d,dict(d)],members=2,steps=2)
    loss.backward()
    assert len(seq)==2 and torch.isfinite(loss)


def test_predictions_cannot_change_scoring_population():
    from tools.terrain_diffusion.metrics import evaluate
    with pytest.raises(ValueError,match='Nonfinite'):
        evaluate(np.array([[[1.,np.nan]],[[1.,2.]]]),np.array([[1.,2.]]))


def test_temporal_missing_hour_keeps_generation():
    from tools.terrain_diffusion.train import trajectory_loss
    m=DiffusionModel(width=16,blocks=1);d=inputs()
    d['target']=torch.randn(1,4);d['mask']=torch.ones(1,4,dtype=torch.bool)
    empty=dict(d);empty['mask']=torch.zeros_like(d['mask'])
    loss,seq=trajectory_loss(m,[d,empty],members=2,steps=2)
    assert torch.isfinite(loss) and len(seq)==2
    loss.backward()


def test_checkpoint_provenance(tmp_path):
    from tools.terrain_diffusion.train import checkpoint_receipt
    p=tmp_path/'source.pt';torch.save(dict(model={},step=123,config={'seed':42}),p)
    receipt=checkpoint_receipt(str(p))
    assert receipt['selected_step']==123 and len(receipt['sha256'])==64


def test_report_uses_declared_data(monkeypatch):
    from tools.terrain_diffusion import report
    seen=[]
    class FakeData:
        def __init__(self,path,device):seen.append(path);self.contract={'data_sha256':'abc'}
    monkeypatch.setattr(report,'Dataset',FakeData)
    report.run_data({'data':'custom/path','data_sha256':'abc'})
    assert seen==['custom/path']
    with pytest.raises(ValueError):report.run_data({'data':'wrong/path','data_sha256':'def'})


def test_hidden_labels_do_not_condition_generation():
    m=DiffusionModel(width=16,blocks=1).eval();d=inputs()
    d['target']=torch.zeros(1,4);d['mask']=torch.ones(1,4,dtype=torch.bool)
    noise=torch.randn(1,4)
    with torch.no_grad():
        a=m.sample(d,noise,steps=2);mean,_=m.baseline(d)
        d['target']=torch.full((1,4),99999.)
        torch.testing.assert_close(a,m.sample(d,noise,steps=2))
        torch.testing.assert_close(mean,m.baseline(d)[0])
