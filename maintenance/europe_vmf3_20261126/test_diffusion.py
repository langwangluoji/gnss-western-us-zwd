import numpy as np
import torch
from . import diffusion_data as data
from . import prob_models


def test_forecast_anchor_preserves_joint_increments_and_new_current_mean():
    joint=np.array([[[10.,12.,15.,19.,23.,28.,35.]]]);current=np.array([[8.]])
    mean=data.anchored_mean(joint,current)
    np.testing.assert_array_equal(mean,[[[8,10,13,17,21,26,33]]])


def test_probability_condition_masks_only_explicit_vmf_channels():
    c=torch.arange(218).float().reshape(1,1,218)
    base=data.condition_view(c,'base');vmf=data.condition_view(c,'vmf')
    torch.testing.assert_close(base[...,:182],c[...,:182],rtol=0,atol=0)
    assert base[...,182:].count_nonzero()==0
    torch.testing.assert_close(vmf,c,rtol=0,atol=0)


def test_europe_weather_summary_uses_actual16_records_and_masks_innovations():
    x=np.ones((2,16,4,15),dtype='f4');x[...,3:6]=0;x[...,:3]=999;x[...,9:11]=888
    summary=data.weather_summary(x)
    assert summary.shape==(2,90)
    assert summary[0,0]==0 and summary[0,9]==0


def test_vmf_gp_remains_psd_unit_diagonal_and_query_consistent():
    corr=.95*(.8**np.abs(np.arange(7)[:,None]-np.arange(7)[None]))+.05*np.eye(7)
    model=prob_models.EuropeWeatherGP(dict(spatial_rho=.5,time_correlation=corr.tolist()))
    c=torch.randn(2,4,218);summary=torch.randn(2,90);static=torch.randn(4,9);ids=torch.tensor([0,1,2,3])
    R=model.correlation(c,summary,static,ids)
    torch.testing.assert_close(R.diagonal(dim1=-2,dim2=-1),torch.ones(2,28,dtype=torch.float64),atol=1e-8,rtol=0)
    assert torch.linalg.eigvalsh(R).min()>0
    q=torch.tensor([1,3]);sub=model.correlation(c[:,q],summary,static[q],ids[q]);ix=(q[:,None]*7+torch.arange(7)).flatten()
    torch.testing.assert_close(sub,R[:,ix][:,:,ix],atol=1e-8,rtol=0)


def test_reused_local_zero_velocity_preserves_gaussian_path():
    from tools.terrain_diffusion.native_process_ops import heun0
    model=prob_models.process('local');c=torch.zeros(2,3,308);noise=torch.randn(2,3,7)
    out=heun0(model,c,noise,steps=4)
    torch.testing.assert_close(out,noise,atol=0,rtol=0)


def test_gaussian_supervision_counts_actual_cells_and_skips_empty_block():
    from .prob_train import gaussian_loss
    model=prob_models.gaussian();c=torch.zeros(1,1,218)
    y=torch.tensor([[[1.,float('nan'),0.,2.,float('nan'),-1.,0.]]])
    loss,count=gaussian_loss(model,y,c)
    assert count==5 and abs(float(loss.detach())-.6)<1e-6
    empty,count=gaussian_loss(model,torch.full_like(y,float('nan')),c)
    assert count==0 and float(empty.detach())==0
