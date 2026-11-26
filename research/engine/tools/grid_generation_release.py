"""Isolated generation release; refuses missing frozen mean provenance."""
from pathlib import Path
import shutil,zipfile,argparse
from gnss_field.artifacts import read_json,write_json,sha256
from grid_pair_release import verify_grid_release

PROJECT=Path(__file__).resolve().parents[1]
NAME='20261121_grid_generation_v02'


def package():
    final=PROJECT/'build'/NAME; previous=PROJECT/'results'/NAME/'previous'
    audit=PROJECT/'results'/NAME/'previous_outer_audit.json'
    if not audit.exists(): raise ValueError('Need verified downloaded old results before release')
    for fold in (0,1):
        p=previous/f'fold_{fold}/resunet69/results.json'
        if not p.exists(): raise ValueError('Need verified downloaded old results before release')
        record=read_json(p)
        if record['fold']!=fold or record['arm']!='resunet69': raise ValueError('Old result provenance mismatch')
    if final.exists(): verify_grid_release(final)
    else:
        stage=final.with_name(final.name+'.staging')
        if stage.exists(): raise ValueError('Preserved partial staging; inspect before rebuild')
        stage.mkdir(parents=True)
        for category in ('src','tools'):
            for p in (PROJECT/category).rglob('*.py'):
                if '__pycache__' in p.parts: continue
                target=stage/p.relative_to(PROJECT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
        for category in ('configs','slurm','docs','assets','src/gnss_field/resources'): (stage/category).mkdir(parents=True,exist_ok=True)
        for name in ('grid_generation_v02.json','grid_processor_pair_v01.json'): shutil.copy2(PROJECT/'configs'/name,stage/'configs'/name)
        for p in (PROJECT/'src/gnss_field/resources').iterdir():
            if p.is_file(): shutil.copy2(p,stage/'src/gnss_field/resources'/p.name)
        shutil.copy2(PROJECT/'slurm/grid_generation_a800.sh',stage/'slurm/grid_generation_a800.sh')
        shutil.copy2(PROJECT/'docs/superpowers/specs/2026-11-21-grid-generation-v01.md',stage/'docs/SPEC.md')
        shutil.copy2(PROJECT/'docs/superpowers/plans/2026-11-21-grid-generation-v01.md',stage/'docs/PLAN.md')
        shutil.copy2(PROJECT/'results'/NAME/'previous_outer_audit.json',stage/'assets/previous_outer_audit.json')
        for fold in (0,1):
            p=previous/f'fold_{fold}/resunet69'
            shutil.copy2(p/'results.json',stage/f'assets/mean_fold{fold}_source.json')
        manifest={p.relative_to(stage).as_posix():sha256(p) for p in stage.rglob('*') if p.is_file()}
        write_json(stage/'grid_release.json',dict(files=manifest,experiment=NAME,
            source_release='1e01c3bdefb04f1f14f2bb813764829d0a0c69bd0dd23ef0994f2a0440e89035',
            source_run='20261121_us_dense_zwd_grid69_processor_pair_v01',C2019=False,D2020=False))
        verify_grid_release(stage); stage.rename(final)
    archive=PROJECT/'dist'/f'{NAME}_code.zip'; temporary=archive.with_suffix('.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(final.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts: z.write(p,p.relative_to(final))
    temporary.replace(archive)
    record=dict(release=str(final),archive=str(archive),release_sha256=sha256(final/'grid_release.json'),archive_sha256=sha256(archive))
    write_json(PROJECT/'results'/NAME/'release.json',record); print(record)


if __name__=='__main__': package()
