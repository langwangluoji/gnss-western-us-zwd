import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).with_name('native_spectral_fast.py')
spec=importlib.util.spec_from_file_location('native_spectral_fast',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_spectral_fast as module


@pytest.mark.parametrize('condition_dim',[64,175])
def test_commuted_complete_graph_messages_preserve_nonzero_denoiser(condition_dim):
    assert hasattr(module,'CommutedFrequencyTokens'), 'equivalent full-spectrum backend is missing'
    from .native_frequency_tokens import NativeFrequencyTokens
    torch.manual_seed(31);old=NativeFrequencyTokens(condition_dim,16).eval()
    torch.nn.init.normal_(old.output.weight,std=.1)
    new=module.CommutedFrequencyTokens(condition_dim,16).eval();new.load_state_dict(old.state_dict())
    x=torch.randn(2,5,7);c=torch.randn(2,5,condition_dim);k=torch.tensor([.2,.7])
    U=torch.linalg.qr(torch.randn(5,5)).Q;eigen=torch.linspace(0,2,5)
    torch.testing.assert_close(new.denoise(x,c,k,U,eigen),old.denoise(x,c,k,U,eigen),rtol=2e-5,atol=2e-6)
    for solver in ('ddim','heun'):
        torch.testing.assert_close(new.sample(c,x,U,eigen,steps=20,solver=solver),old.sample(c,x,U,eigen,steps=20,solver=solver),rtol=2e-5,atol=2e-6)
