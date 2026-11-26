import builtins
import importlib.util
from pathlib import Path


def test_vmf_math_import_does_not_require_external_model4(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('model4_fft_pipeline'):
            raise ModuleNotFoundError('external project deliberately unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    source=Path(__file__).resolve().parents[2]/'research/europe_vmf3_20261126/features.py'
    spec=importlib.util.spec_from_file_location('research.europe_vmf3_20261126.features_without_external',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.background_features)


def test_new_layout_release_binds_engine_tools_and_slurm(tmp_path):
    from gnss_field.artifacts import release_digest
    (tmp_path/'src/gnss_field').mkdir(parents=True)
    (tmp_path/'src/gnss_field/__init__.py').write_text('')
    tools=tmp_path/'research/engine/tools';tools.mkdir(parents=True)
    scripts=tmp_path/'research/engine/slurm';scripts.mkdir(parents=True)
    tool=tools/'train.py';tool.write_text('version1')
    script=scripts/'run.sh';script.write_text('version1')
    first=release_digest(tmp_path);tool.write_text('version2')
    second=release_digest(tmp_path);assert second!=first
    script.write_text('version2');assert release_digest(tmp_path)!=second
