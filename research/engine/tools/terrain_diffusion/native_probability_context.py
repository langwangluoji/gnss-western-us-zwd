"""Explicit known-query information views, without changing frozen mean models."""
from .native_weather_models import NativeGaussian,NativeDiffusion
from .native_frequency_tokens import NativeFrequencyTokens,probability_model


def condition_dimension(view):return {'latent':64,'raw':111,'skip':175}[view]


def condition_view(embedding,raw,view):
    if view=='latent':return embedding
    if view=='raw':return raw
    if view=='skip':
        import torch
        return torch.cat([embedding,raw],-1)
    raise ValueError(f'unknown condition view: {view}')


def make_probability(domain,condition_dim=64,width=64):
    if condition_dim==64 and width==64:return probability_model(domain)
    if domain=='gaussian':return NativeGaussian(condition_dim,width)
    if domain=='frequency_tokens':return NativeFrequencyTokens(condition_dim,width)
    return NativeDiffusion(condition_dim,width,domain)


def require_condition_view(protocol,view):
    if protocol['config'].get('condition_view','latent')!=view:
        raise ValueError('preconditioner condition view differs')


def gaussian_from_protocol(protocol):
    config=protocol['config']
    return NativeGaussian(condition_dimension(config.get('condition_view','latent')),config.get('width',64))
