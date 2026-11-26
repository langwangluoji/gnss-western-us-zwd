import pytest
from weather_calibration_lock import require_lineage


def test_reused_evaluation_must_match_arm_cache_parent_and_noise():
    expected=dict(name='T_weather',seed=42,members=32,gamma_seed=261128,source_evaluation_sha256='parent',
        training_statistic_manifest_sha256='train',development_statistic_manifest_sha256='dev',code_sha256={'model':'source'})
    require_lineage(expected,expected)
    for key,value in [('name','G_weather'),('seed',17),('gamma_seed',1),('members',16),('source_evaluation_sha256','other'),
        ('training_statistic_manifest_sha256','old'),('development_statistic_manifest_sha256','other'),('code_sha256',{})]:
        with pytest.raises(ValueError):require_lineage({**expected,key:value},expected)
