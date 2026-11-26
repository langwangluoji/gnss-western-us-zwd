"""Geometry-only query-field mixture; never consume a target observation.

Distances are to the nearest currently retained physical reference group.
Expert fields share the W0 height convention; blending commutes with the common
height factor. Weights apply at a query point after each expert's field sampling.
This differs from blending grid nodes and then sampling that mixed grid.
"""
import numpy as np


def smoothstep(x):
    z=np.clip(np.asarray(x,dtype=np.float64),0.,1.)
    return z*z*(3.-2.*z)


def support_weights(nearest_km, topology='three'):
    d=np.asarray(nearest_km,dtype=np.float64)
    if not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Finite nonnegative retained-reference distances required')
    far=smoothstep((d-100.)/100.)
    if topology=='three':
        middle=smoothstep((d-25.)/25.)
        weights=np.stack([1.-middle,middle*(1.-far),middle*far],axis=-1)
    elif topology=='two':
        weights=np.stack([np.zeros_like(d),1.-far,far],axis=-1)
    else:
        raise ValueError('Expected two or three expert topology')
    assert np.all(weights>=0.) and np.allclose(weights.sum(-1),1.,rtol=0,atol=1e-14)
    return weights


def blend_query(nearest_km, legacy, random, compact, *, topology='three'):
    fields=np.stack([np.asarray(v,dtype=np.float64) for v in [legacy,random,compact]],axis=-1)
    weights=support_weights(nearest_km,topology)
    if fields.shape!=weights.shape or not np.isfinite(fields).all():
        raise ValueError('Same-shape finite expert predictions required')
    return np.sum(weights*fields,axis=-1)
