from pathlib import Path
from research.layout import paths


def test_layout_separates_code_evidence_and_local_artifacts(tmp_path):
    result = paths(tmp_path)
    assert result['engine'] == tmp_path / 'research/engine'
    assert result['study'] == tmp_path / 'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1'
    assert result['results_public'] == tmp_path / 'experiments/us'
    assert result['package'] == tmp_path / 'src/gnss_field'
