"""Build the immutable Linux release and input bundle; do not connect or submit."""
from pathlib import Path, PurePosixPath
import io
import json
import tarfile

from gnss_field.artifacts import canonical,sha256,write_json


def main():
    project=Path(__file__).resolve().parents[1]
    output=project/'results/20261122_full_generative_design_v01'
    base=PurePosixPath('/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117')
    release=base/'code/20261122_generative_full_v02'
    run=base/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v01'
    config=json.loads((output/'config.local.json').read_text(encoding='utf-8'))
    packed=Path(config['data']['packed_root'])
    files={}
    for path in sorted(packed.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(packed).as_posix()
        allowed=(len(Path(relative).parts)==1 or
                 relative.startswith('A/201') and relative[2:6] in ('2016','2017','2018','2019') or
                 relative.startswith('B/') and relative[2:6] in ('2016','2017','2018') or relative.startswith('C/2019-'))
        if allowed:files['data/gnss_field_server_v1/'+relative]=path
    source=config['data']['background_source'];server_assets={};hashes={}
    for field,name in (('network_checkpoint','network.pt'),('geometry_coefficients','geometry_correction.json'),('fusion_settings','fusion_settings.json')):
        path=Path(source[field]);relative='assets/frozen_background/'+name
        files[relative]=path;server_assets[field]=str(release/relative);hashes[str(release/relative)]=sha256(path)
    server_assets.update(expected_sha256=hashes,geometry_readout_index=0)
    manifest={name:dict(sha256=sha256(path),bytes=path.stat().st_size) for name,path in files.items()}
    def archive(target,paths,extras):
        with tarfile.open(target,'w:gz',compresslevel=3) as tar:
            for name,path in sorted(paths.items()):
                info=tar.gettarinfo(str(path),arcname=name);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                with path.open('rb') as stream:tar.addfile(info,stream)
            for name,content in sorted(extras.items()):
                info=tarfile.TarInfo(name);info.size=len(content);info.mtime=0;tar.addfile(info,io.BytesIO(content))
    data_archive=output/'server_data.tar.gz'
    reuse=False
    if data_archive.exists():
        with tarfile.open(data_archive) as previous:
            reuse=json.loads(previous.extractfile('server_data_manifest.json').read())['files']==manifest
    if not reuse:
        archive(data_archive,files,{'server_data_manifest.json':canonical(dict(files=manifest,D_values_included=False))})
    config['data'].update(packed_root=str(release/'data/gnss_field_server_v1'),cache_root=str(run/'cache'),background_source=server_assets)
    config['runtime'].update(run_root=str(run),project_root=str(release),python='/share/home/u23114/.conda/envs/ZhouSiyu/bin/python',device='cuda')
    config['slurm']=dict(gpu_partition='A800',cpu_partition='intel')
    config['evaluation']['evaluation_population_manifest']=str(release/'protocol/population/C2019.lock.json')
    write_json(output/'config.server.json',config)
    # Validate the actual release configuration, not just synthetic test configs.
    from gnss_field.generative.campaign import load_config,registered_development_hours
    from gnss_field.generative.training import FullTrainingConfig
    checked=load_config(output/'config.server.json')
    registered_development_hours(checked)
    FullTrainingConfig(**{**checked['scientific'],**checked['training'],**checked['evaluation']})
    code={p.relative_to(project).as_posix():p for p in (project/'src').rglob('*.py')}
    for name in ('local_context_features.py','grid_generation_metrics.py','run_full_generative_campaign.py','run_full_generative_preflight.py','export_full_generative_results.py'):
        code['tools/'+name]=project/'tools'/name
    for path in (project/'tests').glob('test_full_generative_*.py'):code[path.relative_to(project).as_posix()]=path
    code['configs/full_generative_a800_v01.json']=output/'config.server.json'
    for path in (output/'population').glob('*'):code['protocol/population/'+path.name]=path
    for name in ('ARCHITECTURE.md','INTERFACES.md','IMPLEMENTATION_PLAN.md','PRO_REVIEW_FULL.txt','PRO_FINAL_REVIEW.txt',
                 'real_cuda_acceptance.json','population_acceptance.json','background_acceptance.json','FINAL_INTEGRATION_REVIEW.md',
                 'RESULTS_EXPORT_REPORT.md','FINAL_TEST_RESULTS.xml','gan_gradient_acceptance.json','OPERATIONS.md'):
        code['protocol/'+name]=output/name
    code_manifest={name:dict(sha256=sha256(path),bytes=path.stat().st_size) for name,path in code.items()}
    archive(output/'server_code.tar.gz',code,{'code_manifest.json':canonical(dict(files=code_manifest))})
    for kind in ('data','code'):
        path=output/f'server_{kind}.tar.gz'
        with tarfile.open(path) as tar:
            if any(not(m.isfile() or m.isdir()) or '..' in PurePosixPath(m.name).parts for m in tar):raise ValueError('Unsafe release member')
        write_json(output/f'server_{kind}_bundle.json',dict(bytes=path.stat().st_size,sha256=sha256(path),release=str(release),run=str(run)))
    print('IMMUTABLE_RELEASE_READY',str(release),len(code),len(files),flush=True)


if __name__=='__main__':main()
