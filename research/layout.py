"""Explicit relocated paths and opt-in historical-path compatibility."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def paths(root=ROOT):
    root = Path(root)
    return dict(root=root, engine=root/'research/engine', package=root/'src/gnss_field',
                study=root/'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1',
                results_public=root/'experiments/us')


def bootstrap(root=ROOT):
    root = Path(root).resolve()
    if os.name == 'nt':
        subprocess.run(['powershell', '-NoProfile', '-File', str(root/'maintenance/paper_restructure_20261126/bootstrap.ps1'), '-Root', str(root)], check=True)
        return
    p = paths(root); legacy=root/'artifacts/legacy_layout/research_pipeline'
    links = {root/'research_pipeline': legacy,
             p['engine']/'results': p['study']/'03_code/gnss_field_server/results' if legacy.exists() else p['results_public'],
             p['engine']/'src': root/'src'}
    for entry in (root/'research').iterdir():
        if entry.is_dir() and entry.name != 'engine' and (entry.name.endswith(('20261123','20261125','20261126'))):
            links[root/'maintenance'/entry.name] = entry
    for link, target in links.items():
        if target.exists() and not link.exists():
            link.parent.mkdir(parents=True,exist_ok=True); link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)


if __name__ == '__main__': bootstrap()
