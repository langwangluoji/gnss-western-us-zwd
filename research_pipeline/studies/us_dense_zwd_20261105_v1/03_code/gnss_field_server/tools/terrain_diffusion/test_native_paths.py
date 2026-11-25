import importlib.util
import os
from pathlib import Path


def test_windows_long_output_path_can_be_written_and_reopened(tmp_path):
    path=Path(__file__).with_name('native_paths.py')
    spec=importlib.util.spec_from_file_location('native_paths',path)
    module=importlib.util.module_from_spec(spec)
    if path.exists():spec.loader.exec_module(module)
    assert hasattr(module,'file_path'), 'long experiment result paths need platform-safe access'
    original=tmp_path/('a'*100)/('b'*100)/'calibration.json'
    normalized=module.file_path(original)
    if os.name=='nt':assert str(normalized).startswith('\\\\?\\')
    normalized.parent.mkdir(parents=True)
    normalized.write_bytes(b'{"verified":true}')
    assert module.file_path(original).read_bytes()==b'{"verified":true}'
    normalized.unlink();normalized.parent.rmdir();normalized.parent.parent.rmdir()
