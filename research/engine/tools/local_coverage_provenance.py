"""Conservative source and packed-identity locks for future local resumes."""
from pathlib import Path
from gnss_field.artifacts import sha256,immutable_json


def dependency_snapshot(project,runtime):
    project=Path(project)
    files={p.relative_to(project).as_posix():sha256(p)
           for folder in ('tools','src/gnss_field')
           for p in sorted((project/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    manifest=Path(runtime['data_root'])/'manifest.json'
    return dict(source_files=files,packed_manifest_sha256=sha256(manifest),
                packed_root=str(manifest.parent.resolve()))


def lock_dependencies(root,snapshot):
    root=Path(root);path=root/'dependencies.lock.json'
    if (root/'experiment.lock.json').exists() and not path.exists():
        raise ValueError('Incomplete legacy provenance: use a new experiment ID; do not retrofit the old lock')
    immutable_json(path,snapshot)
