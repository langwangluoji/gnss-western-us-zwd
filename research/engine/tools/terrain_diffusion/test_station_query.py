import pytest
import torch
import numpy as np


def test_hole_layout_is_balanced_over_day_and_night_without_shuffling_time():
    from .station_query import hole_schedule
    schedule=hole_schedule(3000,42)
    assert schedule.sum()==750
    assert np.all(schedule.reshape(-1,4).sum(1)==1)
    # First epoch starts23,11 alternately. Random block positions avoid locking
    # every hole to the same half of a day as fixed step%4 scheduling does.
    start=np.where(np.arange(3000)%2==0,23,11)
    assert ((start[schedule]==11).mean()>.4) and ((start[schedule]==23).mean()>.4)
    np.testing.assert_array_equal(schedule,hole_schedule(3000,42))


def test_native_station_windows_preserve_masks_and_exclude_later_hours():
    from .station_query import station_tokens
    torch.set_num_threads(2)
    raw=torch.ones(14,6,3)*torch.tensor([100.,280.,900.]);raw[:,:,0]+=torch.arange(14)[:,None]
    raw[2,0,1]=torch.nan;raw[11:13,0,0]=torch.nan
    kwargs=dict(hours=torch.arange(14),ages=torch.zeros(14,6),mean=torch.tensor([100.,280.,900.]),std=torch.tensor([45.,20.,100.]),length=12)
    a=station_tokens(raw,**kwargs)
    assert a['history'].shape==(3,12,6,11)
    assert a['history'][0,2,0,1]==0 and a['history'][0,2,0,4]==0
    assert a['latest'][0,0]==110 and a['available'].all()
    changed=raw.clone();changed[12:]+=1000
    b=station_tokens(changed,**kwargs)
    torch.testing.assert_close(a['history'][0],b['history'][0],atol=0,rtol=0)


@pytest.mark.parametrize('kind',['gru','lstm','transformer'])
def test_native_station_temporal_encoding_is_causal(kind):
    from .station_query import StationTemporal
    torch.manual_seed(4);net=StationTemporal(kind,width=32).eval()
    x=torch.randn(2,12,6,11);static=torch.randn(6,8)
    a=net(x,static,all_times=True);x[:,6:]+=50
    torch.testing.assert_close(net(x,static,all_times=True)[:,:6],a[:,:6],atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('geometry_dtype',[torch.float32,torch.float64])
def test_station_query_zero_init_reordering_and_empty_support(geometry_dtype):
    from .p14_fusion import P14Fusion
    from .station_query import StationResidual
    torch.manual_seed(15);old=P14Fusion('gru',12)
    with torch.no_grad():old.base.point[-1].weight.normal_(0,.02)
    net=StationResidual(old.state_dict(),length=12).eval();B,N,C=2,5,6
    uv=torch.rand(B,1,N,2)*2-1
    b=dict(point=torch.randn(B,N,113),support=torch.randn(B,30,8,8),uv=uv,shape=(8,8),
        static=torch.randn(B,N,9),grid_static=torch.randn(64,8),edges=torch.randn(B,64,3,10),
        edge_mask=torch.ones(B,64,3),legacy_history=torch.randn(B,12,64,25),time_uv=uv[:,0],
        station_history=torch.randn(B,12,C,11),condition_static=torch.randn(C,8),
        qxy=torch.randn(N,2)*100,cxy=torch.randn(C,2)*100,qheight=torch.rand(N)*2000,
        cheight=torch.rand(C)*2000,qcoast=torch.rand(N)*300,ccoast=torch.rand(C)*300,
        latest=torch.rand(B,C)*100,available=torch.ones(B,C,dtype=torch.bool),age=torch.ones(B,C),
        mean=torch.tensor([100.,280.,900.]),std=torch.tensor([45.,20.,100.]),time_scale=100/45)
    for key in ('qxy','cxy','qheight','cheight','qcoast','ccoast'):b[key]=b[key].to(geometry_dtype)
    with torch.no_grad():
        ref=net.reference(b);torch.testing.assert_close(net(b),ref,atol=0,rtol=0)
        net.local[-1].weight.normal_(0,.1);a=net(b)
        small=dict(b,point=b['point'][:,:2],static=b['static'][:,:2],uv=uv[:,:,:2],time_uv=b['time_uv'][:,:2],
            qxy=b['qxy'][:2],qheight=b['qheight'][:2],qcoast=b['qcoast'][:2])
        torch.testing.assert_close(net(small),a[:,:2],atol=1e-6,rtol=1e-5)
        order=torch.randperm(C);changed=dict(b,station_history=b['station_history'][:,:,order])
        for key in ('condition_static','cxy','cheight','ccoast'):changed[key]=b[key][order]
        for key in ('latest','available','age'):changed[key]=b[key][:,order]
        torch.testing.assert_close(net(changed),a,atol=1e-6,rtol=1e-5)
        empty=dict(b,available=torch.zeros_like(b['available']))
        torch.testing.assert_close(net(empty),ref,atol=0,rtol=0)
        assert ((a-ref).abs()*45<=5.0001).all()
    net(b).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in net.time.parameters() if p.grad is not None)>0
    assert not any(p.grad is not None for p in net.reference.parameters())
