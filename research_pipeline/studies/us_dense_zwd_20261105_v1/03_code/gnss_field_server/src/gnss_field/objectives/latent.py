from dataclasses import dataclass
import torch


def observability(h):
    singular = torch.linalg.svdvals(h.detach().double())
    threshold = singular[0]*1e-6 if len(singular) else 0.
    return dict(rank=int((singular > threshold).sum()), columns=h.shape[-1], observations=h.shape[0],
                largest=float(singular[0]) if len(singular) else 0.,
                smallest=float(singular[-1]) if len(singular) else 0.)


def fit_latent(h, residual, curvature=None, *, penalty=.01, curvature_weight=.001, diagnostics=False):
    """FP64 differentiable inverse of training observations, NOT a clean-field label."""
    h, residual = h.double(), residual.double()
    if not len(h) or not torch.isfinite(h).all() or not torch.isfinite(residual).all():
        raise ValueError('Finite training observations are required')
    r = h.shape[1]
    normal = h.T@h/len(h) + (penalty/r)*torch.eye(r, device=h.device, dtype=h.dtype)
    if curvature is not None and len(curvature):
        curvature = curvature.double()
        normal = normal + curvature_weight*(curvature.T@curvature)/len(curvature)
    c = torch.linalg.solve(normal, h.T@residual/len(h))
    return c, observability(h) if diagnostics else {'source': 'regularized_training_observation_inverse'}


@dataclass
class CoefficientTransform:
    mean: torch.Tensor
    scale: torch.Tensor
    active: torch.Tensor

    @classmethod
    def fit(cls, coefficients):
        x = coefficients.double()
        mean = x.mean(0)
        rms = (x-mean).square().mean(0).sqrt()
        return cls(mean.float(), (2*rms).clamp_min(1e-6).float(), rms >= 1e-6)

    def to(self, device):
        return CoefficientTransform(self.mean.to(device), self.scale.to(device), self.active.to(device))

    def encode(self, c):
        mean, scale = self.mean.to(c), self.scale.to(c)
        return ((c-mean)/scale)*self.active.to(c)

    def decode(self, z):
        return self.mean.to(z) + z*self.scale.to(z)*self.active.to(z)

    def state_dict(self):
        return dict(mean=self.mean.cpu(), scale=self.scale.cpu(), active=self.active.cpu())

    @classmethod
    def from_state(cls, state):
        return cls(state['mean'], state['scale'], state['active'])
