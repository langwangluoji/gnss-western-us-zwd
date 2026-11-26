"""Identity, normalization and independent-member contracts for new joint fusion."""
import numpy as np


def validate_embedding_identity(manifest,training,cache_sha,weights_sha,protocol_sha):
    fixed=dict(model_kind='transformer',mean_variant='directed_v2_joint',weights='last',
        mean_weights_sha256=weights_sha,mean_protocol_sha256=protocol_sha,complete=True)
    if any(training.get(k)!=v or manifest.get(k)!=v for k,v in fixed.items()):
        raise ValueError('fusion requires exactly the registered joint v2/last source')
    if manifest.get('cache_manifest_sha256')!=cache_sha:raise ValueError('embedding cache differs')


def require_statistics(stats):
    if stats.get('role')!='training_only':raise ValueError('training-only normalization required')
    for key in ('bias_mm','scale_mm','count'):
        x=np.asarray(stats.get(key,[]),dtype='float64')
        if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('invalid seven-lead statistic: '+key)
        if key in ('scale_mm','count') and (x<=0).any():raise ValueError('nonpositive training statistic')


def centered_pairs(samples,mean):
    samples=np.asarray(samples);mean=np.asarray(mean)
    if samples.ndim<2 or len(samples)<4 or len(samples)%2 or mean.shape!=samples.shape[1:]:
        raise ValueError('even IID members and exactly matching point-field shape required')
    if not np.isfinite(samples).all() or not np.isfinite(mean).all():raise ValueError('nonfinite generated field')
    return mean+(samples[0::2].astype('float64')-samples[1::2].astype('float64'))/np.sqrt(2)


def require_equal_receipt(actual,expected):
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('frozen fusion artifact source differs')


def require_recipe(actual,expected):
    require_equal_receipt(actual,expected)


def trajectory_clock(truth):
    truth=np.asarray(truth)
    if truth.ndim!=4 or truth.shape[-1]!=7:raise ValueError('expected block,origin,query,seven leads')
    valid=np.isfinite(truth).all(-1)
    return dict(supervised_updates=int(valid.any(axis=(1,2)).sum()),complete_trajectories=int(valid.sum()))
