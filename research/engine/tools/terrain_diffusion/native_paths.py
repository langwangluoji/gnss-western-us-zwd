"""Use Windows extended absolute paths for long local experiment artifacts."""
import os
from pathlib import Path


def file_path(path):
    result=Path(path).resolve()
    if os.name!='nt' or str(result).startswith('\\\\?\\'):return result
    value=str(result)
    if value.startswith('\\\\'):return Path('\\\\?\\UNC\\'+value[2:])
    return Path('\\\\?\\'+value)
