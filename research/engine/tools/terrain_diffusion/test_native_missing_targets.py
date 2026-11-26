import numpy as np
import torch
from .native_weather_cache import select_targets
from .native_weather_experiment import complete_trajectory_loss


def test_station_selection_handles_a_network_outage_without_filling_labels():
    truth=np.ones((12,10,7));truth[5:,:,:]=np.nan
    truth[:,8:]=np.nan;hidden=np.ones(10,dtype=bool);hidden[:2]=False
    before=truth.copy()
    ids,fallback=select_targets(truth,hidden,5,np.random.default_rng(7))
    assert fallback and len(np.unique(ids))==5 and hidden[ids].all()
    assert not np.isin(ids,[8,9]).any()
    np.testing.assert_array_equal(truth,before)


class KnownLoss(torch.nn.Module):
    def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
    def loss(self,x,c,U,eigen):return self.scale*x.square().mean()


def test_masked_joint_training_uses_only_observed_marginal_trajectories():
    target=torch.arange(3*4*7,dtype=torch.float32).reshape(3,4,7)/100
    target[1,2,3]=float('nan');target[2]=float('nan')
    model=KnownLoss();condition=torch.zeros(3,4,64);static=torch.zeros(4,9)
    static[:,0]=torch.arange(4)/10
    loss,count,origins=complete_trajectory_loss(model,target,condition,static)
    expected=target[torch.isfinite(target).all(-1)].square().mean()
    torch.testing.assert_close(loss,expected)
    assert count==7 and origins==2
    loss.backward();assert torch.isfinite(model.scale.grad)


def test_no_observed_trajectory_has_zero_supervision_not_fabricated_targets():
    model=KnownLoss();target=torch.full((2,3,7),float('nan'))
    loss,count,origins=complete_trajectory_loss(model,target,torch.zeros(2,3,64),torch.zeros(3,9))
    assert count==origins==0 and loss==0 and torch.isfinite(loss)
