import directed_summary as summary


def test_supplement_uses_preregistered_raw_joint_f6_not_best_ema():
    assert hasattr(summary,'should_supplement'), 'raw-only supplement trigger is missing'
    rows=[dict(arm='H_delta',weights='last',horizon_macro_rmse_mm=[10]*6+[14]),
        dict(arm='joint',weights='last',horizon_macro_rmse_mm=[10]*6+[14.1]),
        dict(arm='H_delta',weights='ema',horizon_macro_rmse_mm=[10]*6+[15]),
        dict(arm='joint',weights='ema',horizon_macro_rmse_mm=[10]*6+[13])]
    assert summary.should_supplement(rows) is False
    rows[1]['horizon_macro_rmse_mm'][6]=13.9
    assert summary.should_supplement(rows) is True
