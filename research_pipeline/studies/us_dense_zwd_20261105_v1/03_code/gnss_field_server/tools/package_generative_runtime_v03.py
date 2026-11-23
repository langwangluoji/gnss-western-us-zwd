"""Package equivalent execution improvements; retain fixed v02 data assets."""
from pathlib import Path,PurePosixPath
import io,tarfile,json
import argparse
from gnss_field.artifacts import read_json,write_json,canonical,sha256
from gnss_field.generative.campaign import load_config,registered_development_hours
from gnss_field.generative.training import FullTrainingConfig

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',choices=('v03','v04','v05'),default='v03')
    revision=parser.parse_args().revision
    project=Path(__file__).resolve().parents[1]
    root=project/'results/20261122_full_generative_design_v01';output=root/'runtime_v03';output.mkdir(exist_ok=True)
    original=read_json(root/'config.server.json')
    base=PurePosixPath('/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117')
    release=base/('code/20261122_generative_full_'+revision)
    run=base/('runs/20261122_us_zwd_gan_diffusion_vae_full_a800_'+('v03' if revision=='v05' else 'v02'))
    original['runtime'].update(project_root=str(release),run_root=str(run),staged_preparation=True,
        preparation_workers=3,evaluation_hours_batch=16,evaluation_chunk=16,
        background_identity_source=str(base/'code/20261122_generative_full_v02/src/gnss_field/generative/background.py'),
        preflight_jobs=[dict(job_id='3697508',stage='benchmark',elapsed_seconds=356)] +
            ([dict(job_id='3698894',stage='benchmark',elapsed_seconds=277)] if revision=='v05' else []))
    original['data']['cache_root']=str(base/'cache/full_generative_v01')
    original['evaluation'].update(evaluation_hours_batch=16,evaluation_chunk=16)
    if revision=='v05':
        original['evaluation']['final_selection_protocol']='subset_locked_products'
    original['benchmark'].update(reuse_preparation_benchmark=str(run/'original_benchmark.json'),budget_seconds=3000)
    original['slurm']['max_cpus_per_gpu']=7
    config=output/('config.server.json' if revision=='v03' else f'config_{revision}.server.json');write_json(config,original)
    checked=load_config(config);registered_development_hours(checked)
    FullTrainingConfig(**{**checked['scientific'],**checked['training'],
        **{k:v for k,v in checked['evaluation'].items() if k!='final_selection_protocol'}})
    files={p.relative_to(project).as_posix():p for p in (project/'src').rglob('*.py')}
    for name in ('run_full_generative_campaign.py','run_full_generative_preflight.py','export_full_generative_results.py',
                 'local_context_features.py','grid_generation_metrics.py'):
        files['tools/'+name]=project/'tools'/name
    files['configs/full_generative_a800_v02.json']=config
    for path in (project/'tests').glob('test_full_generative_*.py'):files[path.relative_to(project).as_posix()]=path
    for name in ('ARCHITECTURE.md','OPERATIONS.md','FINAL_INTEGRATION_REVIEW.md'):
        files['protocol/'+name]=root/name
    for path in project.glob('PERFORMANCE_*_REPORT.md'):files['protocol/'+path.name]=path
    report=root/'PERFORMANCE_ORCHESTRATION_REPORT.md'
    if report.exists():files['protocol/'+report.name]=report
    manifest=dict(files={name:dict(sha256=sha256(path),bytes=path.stat().st_size) for name,path in files.items()},
        scientific_training_unchanged=True,fixed_data_release='20261122_generative_full_v02',
        previous_preflight_job='3697508',previous_card_seconds=356)
    target=output/('code.tar.gz' if revision=='v03' else f'code_{revision}.tar.gz')
    with tarfile.open(target,'w:gz',compresslevel=3) as tar:
        for name,path in sorted(files.items()):
            info=tar.gettarinfo(str(path),arcname=name);info.mtime=0
            with path.open('rb') as stream:tar.addfile(info,stream)
        content=canonical(manifest);info=tarfile.TarInfo('code_manifest.json');info.size=len(content);tar.addfile(info,io.BytesIO(content))
    with tarfile.open(target) as tar:
        import hashlib
        for name,row in manifest['files'].items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=row['sha256']:raise ValueError('Packaged source differs')
    write_json(output/('bundle.json' if revision=='v03' else f'bundle_{revision}.json'),dict(sha256=sha256(target),bytes=target.stat().st_size,release=str(release),run=str(run)))
    print('RUNTIME_RELEASE_READY',revision,len(files),target.stat().st_size,flush=True)

if __name__=='__main__':main()
