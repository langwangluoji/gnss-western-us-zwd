import numpy as np
import pytest


def test_benchmark_panel_spans_every_month_with_fixed_reproducible_order():
    from gnss_field.generative.benchmark import benchmark_panel
    hours = np.arange(np.datetime64('2016-01-01T18').astype('datetime64[h]').astype('i8'),
                      np.datetime64('2019-01-01').astype('datetime64[h]').astype('i8'))
    panel = benchmark_panel(hours, per_month=8)
    assert len(panel) == 36*8
    assert len(np.unique(panel.astype('datetime64[h]').astype('datetime64[M]'))) == 36
    assert np.array_equal(panel, benchmark_panel(hours[::-1], per_month=8))
    assert not np.array_equal(panel, np.sort(panel))


def test_measurement_extrapolation_accounts_full_population_and_both_products():
    from gnss_field.generative.benchmark import extrapolate_costs
    result = extrapolate_costs(training_hours=26262, development_hours=8760,
        preparation_seconds_per_hour=.1, evaluation_seconds_per_hour={'vae':.2,'gan':.3,'diffusion':.4},
        pilot_seconds_per_update=.5, pilot_validation_seconds_per_hour=.25,
        checkpoint_seconds=.8, representation_check_seconds=10.)
    assert result['preparation_seconds'] == pytest.approx((26262+8760)*.1)
    assert result['evaluation_seconds_per_call']['vae'] == pytest.approx(576*.2)
    assert result['diagnostics_seconds'] >= 12*5*216*.5
    # Shared + complete, 3 families, two products; K64/128 costs are separately measured.
    assert result['final_evaluation_seconds'] >= 4*8760*(.2+.3+.4)
    with pytest.raises(ValueError):
        extrapolate_costs(training_hours=8, development_hours=8, preparation_seconds_per_hour=float('nan'),
            evaluation_seconds_per_hour={'vae':1,'gan':1,'diffusion':1}, pilot_seconds_per_update=1,
            pilot_validation_seconds_per_hour=1, checkpoint_seconds=1, representation_check_seconds=1)

def test_sustained_timing_separates_startup_data_and_writes():
    from gnss_field.generative.benchmark import sustained_timing
    cold = dict(step=8, telemetry=dict(data_seconds=2., model_seconds=18., checkpoint_seconds=1.))
    hot = dict(step=32, telemetry=dict(data_seconds=8., model_seconds=24., checkpoint_seconds=3.))
    result = sustained_timing(cold, hot, cold_wall_seconds=23.)
    assert result['seconds_per_update'] == pytest.approx(.5)
    assert result['data_seconds_per_update'] == pytest.approx(.25)
    assert result['model_seconds_per_update'] == pytest.approx(.25)
    assert result['startup_seconds'] == pytest.approx(18.)
    assert result['checkpoint_seconds'] == 3.
    with pytest.raises(ValueError):
        sustained_timing(cold, cold, cold_wall_seconds=23.)

def test_missing_preparation_panel_excludes_hot_rows_and_covers_four_months(tmp_path):
    from gnss_field.generative.benchmark import missing_preparation_panel
    calendar = np.arange(np.datetime64('2016-01-01','h').astype('i8'),
                         np.datetime64('2016-05-01','h').astype('i8'))
    class Data:
        train_hours=calendar; cache_root=tmp_path
        def _has_shard(self,path): return True
        def _load_shard(self,path): return {'hour_ids':calendar[::2]}
    panel=missing_preparation_panel(Data())
    assert len(panel)==128
    assert not np.isin(panel,calendar[::2]).any()
    assert len(np.unique(panel.astype('datetime64[h]').astype('datetime64[M]')))==4

def test_benchmark_uses_configured_evaluation_tiling_for_all_paths():
    from gnss_field.generative.benchmark import benchmark_evaluation_runtime
    assert benchmark_evaluation_runtime({'runtime':{'evaluation_hours_batch':16,'evaluation_chunk':16}})=={
        'evaluation_hours_batch':16,'evaluation_chunk':16}
    assert benchmark_evaluation_runtime({'runtime':{'evaluation_hours_batch':16},
        'evaluation':{'evaluation_hours_batch':8,'evaluation_chunk':32}})=={
        'evaluation_hours_batch':8,'evaluation_chunk':32}
    with pytest.raises(ValueError):
        benchmark_evaluation_runtime({'runtime':{'evaluation_hours_batch':17}})
