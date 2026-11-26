"""Frozen two-GP additional-date CPU inference; no training or model selection."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read,run
from tools.terrain_diffusion.data import sha
from tools.terrain_diffusion.native_model_seal import seal_completed_model,evaluation_marker
from tools.terrain_diffusion.native_weather_gp_experiment import codes
from centered_artifact_audit import joint_marker
from event_audit import validate_calendar_origins


def wait_for(path):
    while not path.exists():
        if STOP.exists():raise SystemExit(75)
        time.sleep(15)


def validate_extra_lineage(lineage,source,cache_sha,embedding_sha,code_sha,reference):
    expected=dict(source=source,cache_manifest_sha256=cache_sha,embedding_manifest_sha256=embedding_sha,code_sha256=code_sha,
        members=32,seed=261126,device='cpu',independent_iid_member_noise=True)
    if any(lineage[k]!=v for k,v in expected.items()):raise ValueError('extra GP inference recipe differs')
    if reference and (lineage['probability_weights_sha256'] is not None or lineage['model_protocol_sha256'] is not None):
        raise ValueError('Gaussian reference unexpectedly uses trained covariance weights')


def extra_marker(folder,seal,source,cache,embedding):
    if not folder.exists():return None
    marker=evaluation_marker(folder,seal)
    validate_extra_lineage(read(marker),source,sha(cache/'manifest.json'),sha(embedding/'manifest.json'),codes(),seal is None)
    return marker


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args();seed=args.seed
    lock=read(Path(__file__).with_name('weather_gp_supplement_lock.json'))
    cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar15_s{seed}'
    embedding=RESULTS/'20261126_native_weather_embeddings'/f'transformer72_physics_aux_last_calendar15_s{seed}'
    wait_for(cache/'manifest.json');wait_for(embedding/'manifest.json')
    config=read(cache/'protocol.json')['config']
    if config['calendar_day']!=15 or config['seed']!=seed or config['months']!=12:raise ValueError('registered extra calendar required')
    import numpy as np
    manifest=read(cache/'manifest.json')
    if manifest['protocol_sha256']!=sha(cache/'protocol.json'):raise ValueError('extra calendar protocol changed')
    if sha(cache/'origins.npy')!=manifest['files']['origins.npy']:raise ValueError('extra calendar hours changed')
    validate_calendar_origins(np.load(cache/'origins.npy'),15)
    if read(embedding/'manifest.json')['cache_manifest_sha256']!=sha(cache/'manifest.json'):raise ValueError('extra embedding cache differs')
    gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    common=['--cache',cache,'--embeddings',embedding,'--gaussian',gaussian,'--copula',gaussian/'copula.json',
        '--seed',seed,'--members',32,'--sample-seed',lock['sample_seed'],'--device',lock['device']]
    root=RESULTS/'20261126_weather_gp_calendar15_cpu'/f's{seed}';reference=root/'reference'
    expected_source=lock['sources'][str(seed)]['c175']['source']
    run('native_weather_gp_experiment',['--phase','reference','--out',reference,*common],extra_marker(reference,None,expected_source,cache,embedding))
    extra_marker(reference,None,expected_source,cache,embedding);targets=[reference]
    for view in lock['views']:
        model=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c{view}'
        wait_for(model/'status.json');protocol=read(model/'protocol.json')
        if protocol['config']['seed']!=seed or protocol['config']['view']!=view:raise ValueError('GP seed/view differs')
        seal=seal_completed_model(model,'COMPLETE_WEATHER_GP',codes());expected=lock['sources'][str(seed)]['c175']
        if seal['source']!=expected['source']:raise ValueError('GP mean/Gaussian source differs')
        if view==175 and (seal['weights_sha256']!=expected['weights_sha256'] or seal['protocol_sha256']!=expected['protocol_sha256']):raise ValueError('frozen G175 changed')
        target=root/f'c{view}'
        run('native_weather_gp_experiment',['--phase','evaluate','--model',model,'--out',target,*common],extra_marker(target,seal,expected_source,cache,embedding))
        extra_marker(target,seal,expected_source,cache,embedding);targets.append(target)
        # Score each completed candidate before waiting for the next one.
        for folder in targets:
            script=Path(__file__).with_name('probability_joint_audit.py')
            if joint_marker(folder,script) is None:
                if STOP.exists():raise SystemExit(75)
                subprocess.run([sys.executable,str(script),
                    '--cache',str(cache),'--evaluation',str(folder),'--out',str(folder)],cwd=ROOT,check=True)
            joint_marker(folder,script)
    print('Complete frozen extra-calendar CPU verification',flush=True)


if __name__=='__main__':main()
