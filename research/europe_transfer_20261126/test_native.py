import numpy as np
import torch
from . import native


def test_native_condition_has_no_future_or_query_observation():
    rng=np.random.default_rng(3);T=40;C=3;Q=2
    cache=dict(base=rng.normal(size=(T,5)).astype('f4'),innovation=rng.normal(size=(T,5,3)).astype('f4'),weather=rng.normal(size=(T,5,18)).astype('f4'),context=np.array([0,1,2]),start=10,self_error=np.zeros((T,C),dtype='f4'))
    a=dict(values=rng.normal(size=(60,5,3)).astype('f4'),hours=np.arange(60)*6,static=np.zeros((5,9),dtype='f4'),xy=np.zeros((5,2),dtype='f4'))
    x=native.inputs(cache,a,np.array([25,26]),np.array([3,4]),np.zeros(9))
    a['values'][:,3:]=999;a['values'][37:,:3]=999
    y=native.inputs(cache,a,np.array([25,26]),np.array([3,4]),np.zeros(9))
    for key in ('history','condition','background'):
        np.testing.assert_array_equal(x[key],y[key])
    assert x['condition'].shape==(2,2,111)
    assert x['history'].shape==(2,16,3,15)


def test_transferred_joint_is_exact_zero_initialized_and_supports_411_ids():
    model=native.make_model(411)
    condition=torch.zeros(2,3,111);history=torch.zeros(2,16,4,15);history[...,3:6]=1
    pred,_=model(condition,history,torch.zeros(4,9),torch.tensor([0,1,409,410]),torch.zeros(3,9))
    torch.testing.assert_close(pred,torch.zeros(2,3,7),rtol=0,atol=0)
    assert model.lags.tolist()==[0,1,3,6,12]


def test_layout_boundaries_keep_training_clock_consecutive():
    chunks=[dict(start=16,fit_start=32,base=np.zeros((160,2))),dict(start=144,fit_start=32,base=np.zeros((160,2)))]
    blocks=native.training_blocks(chunks)
    absolute=[int(chunks[i]['start'])+j for i,j in blocks]
    assert absolute[0]==48
    assert np.all(np.diff(absolute)==4)
    assert absolute[-1]==300


def test_missing_gnss_masks_both_innovation_channels():
    cache=dict(base=np.zeros((40,4),dtype='f4'),innovation=np.ones((40,4,3),dtype='f4'),weather=np.zeros((40,4,18),dtype='f4'),context=np.array([0,1]),start=10,self_error=np.ones((40,2),dtype='f4'))
    values=np.ones((60,4,3),dtype='f4');values[20,0,0]=np.nan
    a=dict(values=values,hours=np.arange(60)*6,static=np.zeros((4,9),dtype='f4'),xy=np.zeros((4,2),dtype='f4'))
    result=native.inputs(cache,a,np.array([25]),np.array([2,3]),np.ones(9))
    assert result['history'][0,0,0,3]==0
    np.testing.assert_array_equal(result['history'][0,0,0,9:11],[0,0])


def test_inference_age_includes_observations_before_history_window():
    cache=dict(base=np.zeros((50,4),dtype='f4'),innovation=np.zeros((50,4,3),dtype='f4'),weather=np.zeros((50,4,18),dtype='f4'),context=np.array([0,1]),start=0,self_error=np.zeros((50,2),dtype='f4'))
    values=np.ones((50,4,3),dtype='f4');values[10:25,0,0]=np.nan
    a=dict(values=values,hours=np.arange(50)*6,static=np.zeros((4,9),dtype='f4'),xy=np.zeros((4,2),dtype='f4'))
    result=native.inputs(cache,a,np.array([30]),np.array([2,3]),np.zeros(9))
    # History begins at15; last observed record9 is36physical hours earlier.
    np.testing.assert_allclose(result['history'][0,0,0,6],np.log1p(36)/5,rtol=1e-6)
