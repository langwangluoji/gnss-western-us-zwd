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
