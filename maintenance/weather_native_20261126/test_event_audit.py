import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).with_name('event_audit.py')
spec=importlib.util.spec_from_file_location('event_audit',path);module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_fair_brier_removes_member_noise_with_missing_truth():
    assert hasattr(module,'fair_brier'),'IID-corrected Brier score missing'
    events=np.array([[0.,1.,0.],[1.,1.,1.]])
    actual=module.fair_brier(events,np.array([1.,0.,np.nan]))
    np.testing.assert_array_equal(actual[:2],[0.,1.]);assert np.isnan(actual[2])


def test_tail_weighted_crps_scores_censored_increment_distributions():
    assert hasattr(module,'tail_crps'),'tail-weighted CRPS missing'
    x=np.array([[0.,0.],[2.,-2.]])
    up=module.tail_crps(x,np.array([3.,-3.]),1,'increase')
    down=module.tail_crps(x,np.array([3.,-3.]),1,'decrease')
    np.testing.assert_array_equal(up,[1.,0.]);np.testing.assert_array_equal(down,[0.,1.])


def test_event_source_guard_rejects_calendar_swap_and_validates_locked_supplement():
    assert hasattr(module,'validate_event_lineage'),'calendar-aware frozen event guard missing'
    lock=dict(sources={'calendar8/s42/full':dict(prediction_sha256='main',weights_sha256='w'),
        'calendar15/s42/full':dict(prediction_sha256='extra',weights_sha256='w'),
        'calendar8/s17/full':dict(prediction_sha256='main17',weights_sha256='w17')})
    line=dict(prediction_sha256='extra',probability_weights_sha256='w',members=32,solver='heun0',sampling_steps=20,
        seed=261126,independent_iid_member_noise=True,cache_manifest_sha256='cache15',embedding_manifest_sha256='emb15',source={'mean':'frozen'})
    config=dict(calendar_day=15,seed=42,months=12)
    module.validate_event_lineage(line,lock,15,42,'full',config,'cache15','emb15',{'mean':'frozen'})
    with pytest.raises(ValueError):module.validate_event_lineage({**line,'prediction_sha256':'main'},lock,15,42,'full',config,'cache15','emb15',{'mean':'frozen'})
    with pytest.raises(ValueError):module.validate_event_lineage(line,lock,15,42,'full',{**config,'calendar_day':8},'cache15','emb15',{'mean':'frozen'})
    line={**line,'prediction_sha256':'new17','probability_weights_sha256':'w17'};config['seed']=17
    module.validate_event_lineage(line,lock,15,17,'full',config,'cache15','emb15',{'mean':'frozen'})
    for change in ({'cache_manifest_sha256':'cache8'},{'solver':'ddim'},{'members':2},{'source':{'mean':'other'}}):
        with pytest.raises(ValueError):module.validate_event_lineage({**line,**change},lock,15,17,'full',config,'cache15','emb15',{'mean':'frozen'})


def test_event_calendar_checks_actual_continuous_hours_and_layout_order():
    assert hasattr(module,'validate_calendar_origins'),'actual event calendar hours guard missing'
    rows=[]
    for layout in range(3):
        for month in range(1,13):
            start=np.datetime64(f'2020-{month:02d}-15T00','h').astype('int64')
            rows.extend(range(start,start+72))
    origins=np.array(rows).reshape(216,12);module.validate_calendar_origins(origins,15)
    wrong=origins.copy();wrong[0,0]=np.datetime64('2020-01-08T00','h').astype('int64')
    with pytest.raises(ValueError):module.validate_calendar_origins(wrong,15)
