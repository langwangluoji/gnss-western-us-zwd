"""Immutable input/implementation lock for the final weather-scale/tail experiment."""
from pathlib import Path
from campaign import RESULTS,read
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
from tools.terrain_diffusion.native_inducing_flow_experiment import codes as source_codes


LOCK=Path(__file__).with_name('weather_calibration_lock_v2.json')


def receipt():
    from weather_calibration_run import ROOT,cached,codes,GAMMA_SEED,VARIANTS
    sources={}
    for seed in (42,17):
        train,_,_=cached(seed);dev,_,_=cached(seed,8)
        parent=RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}'
        seal=validate_model_seal(parent,'COMPLETE_INDUCING_FLOW',source_codes());evaluation=RESULTS/f'20261126_inducing_latent_flow/calendar8/affine_time_s{seed}'
        evaluation_marker(evaluation,seal)
        sources[str(seed)]=dict(training_manifest_sha256=sha(train/'manifest.json'),development_manifest_sha256=sha(dev/'manifest.json'),
            affine_model_seal_sha256=sha(parent/'model_artifact_seal.json'),parent_evaluation_sha256=sha(evaluation/'evaluation.npz'),
            parent_lineage_sha256=sha(evaluation/'lineage.json'))
    names=['probability_joint_audit.py','inducing_calibration_audit.py','weather_controls_summary.py','event_audit.py',
        'weather_calibration_summary.py','weather_calibration_lock.py','mean_hs_summary.py','event_audit_lock.json']
    return dict(code_sha256=codes(),sources=sources,helpers={n:sha(Path(__file__).with_name(n)) for n in names},
        variants=list(VARIANTS),steps=3000,members=32,gamma_seed=GAMMA_SEED,
        primary='T_weather vs G_weather F6 fairCRPS; G_extra matched extra budget control',
        covariance_match='T_weather and G_weather same mu and s2(c)C',nu_grid=[None,3,4,6,8,12,20,40,80,160,1000],
        user_end_condition='finish this experiment then stop new experiments and deliver full research summary',
        strict_oof=False,confirmation_C_D_read=False)


def validate_lock():
    expected=read(LOCK);actual=receipt()
    if expected!=actual:raise ValueError('final experiment locked input, implementation or contract changed')
    return expected


def create_lock():
    if LOCK.exists():raise ValueError('scientific lock already exists; never overwrite')
    write_json(LOCK,receipt());print('Locked final weather-scale/tail experiment',flush=True)


def require_lineage(actual,expected):
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('evaluation family, seed, noise or input source differs')
