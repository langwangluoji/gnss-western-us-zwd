import importlib
import importlib.util
import pytest


def test_completion_uses_mask_derived_supervision_and_optimizer_clock():
    assert importlib.util.find_spec('inducing_flow_completion') is not None,'mask-aware completion guard missing'
    m=importlib.import_module('inducing_flow_completion')
    support=dict(blocks=3000,supervised_blocks=2999,total_finite_labels=120)
    status=dict(steps=3000,supervised_updates=2999)
    final=dict(step=3000,scheduler=dict(last_epoch=2999),optimizer=dict(state={0:dict(step=2999)}))
    curve=dict(step=3000,supervised_updates=2999,total_labels=120)
    m.require_counts(status,final,curve,support)
    with pytest.raises(ValueError):m.require_counts({**status,'supervised_updates':3000},final,curve,support)
    with pytest.raises(ValueError):m.require_counts(status,{**final,'scheduler':dict(last_epoch=3000)},curve,support)
    with pytest.raises(ValueError):m.require_counts(status,final,{**curve,'total_labels':121},support)
