import numpy as np
import torch
from . import field


def fixture():
    rng=np.random.default_rng(4)
    static=np.zeros((8,9),dtype='f4');static[:,:2]=rng.uniform(-.1,.1,(8,2));static[:,2]=.1
    grid=np.zeros((64,9),dtype='f4');x,y=np.meshgrid(np.linspace(-.2,.2,8),np.linspace(-.2,.2,8));grid[:,:2]=np.c_[x.ravel(),y.ravel()]
    a=dict(values=rng.uniform(10,150,(40,8,3)).astype('f4'),hours=np.arange(40)*6,
           static=static,grid_static=grid,xy=static[:,:2]*1000,grid_xy=grid[:,:2]*1000,
           lo=np.array([-200,-200]),hi=np.array([200,200]),grid_shape=[8,8],height=static[:,2]*3000,
           grid_height=grid[:,2]*3000,role=np.array(['train']*6+['dev']*2))
    return a


def test_query_labels_and_future_do_not_enter_condition():
    a=fixture();d=field.FieldData(a,device='cpu')
    b=d.batch(np.array([20,21]),np.array([0,1,2,3]),np.array([4,5]))
    a['values'][:,4:]=99999;a['values'][22:,:4]=88888
    c=field.FieldData(a,device='cpu').batch(np.array([20,21]),np.array([0,1,2,3]),np.array([4,5]))
    for key in ('point','support','history','edges'):
        torch.testing.assert_close(b[key],c[key],rtol=0,atol=0)


def test_actual_p14_gru_has_finite_backward_on_masked_observations():
    a=fixture();a['values'][10:19,0,0]=np.nan
    d=field.FieldData(a,device='cpu');b=d.batch(np.array([20,21]),np.array([0,1,2,3]),np.array([4,5]))
    model=field.EuropeBase();p=model(b)
    assert p.shape==(2,2)
    loss=(p-b['target']).square().mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_nonconsecutive_training_block_rejected():
    import pytest
    d=field.FieldData(fixture(),device='cpu')
    with pytest.raises(ValueError,match='consecutive'):
        d.batch(np.array([20,22]),np.array([0,1,2]),np.array([4]))
