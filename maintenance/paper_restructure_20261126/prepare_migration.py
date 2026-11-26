"""Copy only tracked lightweight evidence; filesystem moves use migrate.ps1."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path.cwd()
HERE = ROOT / 'maintenance/paper_restructure_20261126'
ENGINE = 'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def mapped(relative):
    if relative.startswith(ENGINE + '/src/'):
        return 'src/' + relative[len(ENGINE + '/src/'):]
    if relative.startswith(ENGINE + '/results/'):
        return 'experiments/us/' + relative[len(ENGINE + '/results/'):]
    if relative.startswith(ENGINE + '/'):
        tail = relative[len(ENGINE + '/'):]
        if tail.split('/')[0] in ('build', 'local_validation', 'research_pipeline'):
            return 'archive/legacy/' + relative
        return 'research/engine/' + tail
    if relative.startswith('research_pipeline/'):
        return 'archive/legacy/' + relative
    if relative.startswith('maintenance/') and not relative.startswith('maintenance/paper_restructure_20261126/'):
        tail = relative[len('maintenance/'):]
        parts = tail.split('/')
        if parts[0] in ('europe_transfer_20261126', 'europe_vmf3_20261126') and len(parts) > 1 and parts[1] in ('runs', 'probability', 'evaluation'):
            return 'experiments/europe/' + tail
        return 'research/' + tail
    return relative


def long(path):
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' else path


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode('utf-8').strip('\0').split('\0')
    mapping = {p: mapped(p) for p in tracked}
    # Source snapshots and new public evidence must exist before moving originals.
    for old, new in mapping.items():
        if new.startswith(('experiments/', 'archive/legacy/')):
            target = long(ROOT / new); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(long(ROOT / old), target)
    original = ROOT / 'archive/legacy/root_before'; original.mkdir(parents=True, exist_ok=True)
    for name in ('README.md', 'AGENTS.md', '.gitignore'):
        shutil.copy2(ROOT / name, original / name)
    (HERE / 'path_mapping.json').write_text(json.dumps(dict(source_commit='b7287371ab198d03e7a0700995f1afcca04016ee',
        paths=mapping, policy='old science locks unchanged; tracked evidence copied byte-for-byte before moves'), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(tracked_files=len(tracked), public_path_changes=sum(a != b for a,b in mapping.items()))))


if __name__ == '__main__': main()
