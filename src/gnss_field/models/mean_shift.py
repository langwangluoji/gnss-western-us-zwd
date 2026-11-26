"""Non-diffusion mean control on the same fixed Gaussian field representation."""
from torch import nn


class SubspaceMeanShift(nn.Module):
    def __init__(self, rank=64, condition_dim=32, width=128):
        super().__init__()
        self.rank, self.condition_dim = rank, condition_dim
        self.net = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, rank))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def sample_noise(self, condition, noise):
        return noise+self.net(condition)[:, None]
