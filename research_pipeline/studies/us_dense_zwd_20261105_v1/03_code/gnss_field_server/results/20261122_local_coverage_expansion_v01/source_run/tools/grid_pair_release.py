"""Build and verify an isolated code release; reuse only checksum-matched prior assets."""
from pathlib import Path
import argparse
import shutil
import zipfile
from gnss_field.artifacts import sha256,read_json,write_json


def verify_grid_release(root):
    root=Path(root).resolve()
    manifest=root/'grid_release.json'
    files=read_json(manifest)['files']
    if not files:
        raise ValueError('Empty release manifest')
    for name,digest in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256(p)!=digest:
            raise ValueError(f'Changed or missing release dependency: {name}')
    return sha256(manifest)


def publish_archive(destination):
    project=Path(__file__).resolve().parents[1]
    archive=project/'dist/20261121_grid_processor_pair_v01_code.zip'
    temporary=archive.with_suffix('.staging')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file() and p.relative_to(destination).parts[0]!='bundle':
                z.write(p,p.relative_to(destination))
    temporary.replace(archive)
    write_json(project/'results/20261121_grid_processor_pair_v01/release.json',dict(
        archive=str(archive),archive_sha256=sha256(archive),release=str(destination),
        release_sha256=sha256(destination/'grid_release.json'),
        reused_bundle={p.name:sha256(p) for p in (destination/'bundle').iterdir() if p.is_file()}))
    print('ISOLATED_CODE_RELEASE_READY',archive,sha256(archive),flush=True)


def package():
    from grid_pair_data import PROJECT,local_runtime,BUNDLE_FILES
    r=local_runtime()
    destination=PROJECT/'build/20261121_grid_processor_pair_v01'
    final=destination
    if final.exists():
        verify_grid_release(final)
        return publish_archive(final)
    destination=final.with_name(final.name+'.staging')
    if destination.exists():
        raise ValueError('Incomplete staging preserved; inspect before a new build')
    destination.mkdir(parents=True)
    for category in ('src','tools'):
        for source in (PROJECT/category).rglob('*.py'):
            if '__pycache__' in source.parts:
                continue
            target=destination/source.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    resources=PROJECT/'src/gnss_field/resources'
    (destination/'src/gnss_field/resources').mkdir(parents=True,exist_ok=True)
    for source in resources.glob('*'):
        if source.is_file():
            shutil.copy2(source,destination/source.relative_to(PROJECT))
    assets=destination/'assets'; assets.mkdir()
    for key,name in [('m0','M0_005000.pt'),('design','design.json'),('outer','outer.json'),('parameters','mixture.json')]:
        shutil.copy2(r[key],assets/name)
    (destination/'bundle').mkdir()
    for name in BUNDLE_FILES:
        shutil.copy2(Path(r['bundle'])/name,destination/'bundle'/name)
    (destination/'configs').mkdir()
    shutil.copy2(PROJECT/'configs/grid_processor_pair_v01.json',destination/'configs/grid_processor_pair_v01.json')
    (destination/'docs').mkdir()
    shutil.copy2(PROJECT/'docs/superpowers/plans/2026-11-21-grid-processor-pair-v01.md',destination/'docs/PLAN.md')
    (destination/'slurm').mkdir()
    shutil.copy2(PROJECT/'slurm/grid_pair_a800.sh',destination/'slurm/grid_pair_a800.sh')
    files={p.relative_to(destination).as_posix():sha256(p) for p in destination.rglob('*') if p.is_file()}
    write_json(destination/'grid_release.json',dict(files=files,experiment='20261121_grid_processor_pair_v01',
        input='A only; B labels restricted to losses/evaluation',C2019=False,D2020=False))
    verify_grid_release(destination)
    destination.rename(final); destination=final
    publish_archive(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--verify',type=Path); a=p.parse_args()
    print(verify_grid_release(a.verify)) if a.verify else package()
