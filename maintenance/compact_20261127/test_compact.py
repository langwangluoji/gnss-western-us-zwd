"""Protection boundaries for the file retirement ledger (no real data)."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('compact', Path(__file__).with_name('compact.py'))


def module():
    assert SPEC.origin and Path(SPEC.origin).exists(), 'compaction implementation not yet written'
    mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(mod)
    return mod


def test_core_raw_and_metadata_survive():
    m = module()
    assert m.classify('project', m.R + '/20261126_native_weather_cache_v2/train_s17/truth.npy')[0] == 'keep_core'
    assert m.classify('project', 'artifacts/legacy_layout/research_pipeline/data/pwv_data/a.pwv.gz')[0] == 'keep_raw'
    assert m.classify('project', m.R + '/old_run/protocol.json')[0] == 'keep_evidence'
    assert m.classify('project', m.R + '/old_run/checkpoint_500.pt')[0] == 'archive_model'
    assert m.classify('project', m.R + '/old_run/evaluation.npz')[0] == 'retire_derived'
    assert m.classify('project', m.R + '/old_run/normalization.npz')[0] == 'archive_model'
    assert m.classify('project', m.R + '/old_run/feature_normalization.npz')[0] == 'archive_model'
    assert m.classify('project', m.R + '/old_run/latent_basis.npy')[0] == 'archive_model'
    assert m.classify('project', 'research/engine/build/a.npy')[0] == 'keep_other'


def test_uncertain_frozen_data_stays():
    m = module()
    assert m.classify('cache_d', 'europe_ref_gan_20261103_v1/frozen_2023/train_reference_utc_rebuild.h5')[0] == 'keep_raw'
    assert m.classify('cache_d', 'europe_ref_gan_20261103_v1/frozen_2023/train_reference_utc.h5.partial')[0] == 'retire_derived'
    assert m.classify('cache_d', 'europe_dense_zwd_20261104_v1/physics/202001.weather.h5')[0] == 'retire_derived'
    assert m.classify('cache_d', 'europe_dense_zwd_20261104_v1/physics/202001.receipt.json')[0] == 'keep_evidence'


def test_manifest_paths_cannot_escape(tmp_path):
    m = module()
    for bad in ('../outside.npy', '/outside.npy', 'E:/outside.npy', 'a/../../x.npy'):
        with pytest.raises(ValueError):
            m.safe_path(tmp_path, bad)
    assert str(m.safe_path(tmp_path, 'a/b.npy')).endswith('a\\b.npy')


def test_no_model_or_raw_in_delete_plan_without_archive():
    m = module()
    item = dict(scope='project', path=m.R + '/old/last.pt', action='archive_model', size=1, sha256='a'*64)
    with pytest.raises(ValueError, match='archive'):
        m.deletion_rows([item], {})
    assert m.deletion_rows([dict(item, action='keep_raw')], {}) == []
    receipt = {'a'*64: {'asset': 'models-01.tar.gz', 'verified_remote': True}}
    assert len(m.deletion_rows([item], receipt)) == 1


def test_trusted_numpy_checkpoint_still_uses_restricted_loader(tmp_path, monkeypatch):
    import importlib
    import numpy as np
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    check = importlib.import_module('verify_retained')
    assert hasattr(check, 'load_weights'), 'restricted NumPy checkpoint loader needed'
    path = tmp_path / 'old_cnn.pth'
    torch.save({'normalization':np.array([1.,2.]),'weight':torch.ones(2)},path)
    loaded = check.load_weights(path)
    np.testing.assert_array_equal(loaded['normalization'],[1.,2.])
    assert torch.equal(loaded['weight'],torch.ones(2))


def test_upload_progress_accepts_pathlib_filename(tmp_path, monkeypatch, capsys):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    transfer = importlib.import_module('remote_archive')
    path = tmp_path / 'archive.bin'
    path.write_bytes(b'checkpoint')
    with transfer.UploadReader(path) as reader:
        reader.last_report = 0
        assert reader.read() == b'checkpoint'
    assert 'upload_progress' in capsys.readouterr().out


def test_retry_only_replaces_our_incomplete_upload_stubs(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    transfer = importlib.import_module('remote_archive')
    assert hasattr(transfer, 'incomplete_stub'), 'interrupted upload stubs need a bounded retry gate'
    allowed = {'retired-models-02.tar.gz'}
    assert transfer.incomplete_stub({'name':'retired-models-02.tar.gz','state':'starter'},allowed)
    assert not transfer.incomplete_stub({'name':'retired-models-02.tar.gz','state':'uploaded'},allowed)
    assert not transfer.incomplete_stub({'name':'unrelated.zip','state':'starter'},allowed)
