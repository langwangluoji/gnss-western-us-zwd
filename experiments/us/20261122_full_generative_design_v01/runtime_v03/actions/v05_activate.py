from pathlib import Path
import hashlib,json,os,sys,tarfile,shutil,time,subprocess
BASE=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
RELEASE=BASE/'code/20261122_generative_full_v05'
RUN=BASE/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03'
OLD=BASE/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v02'
SOURCE_CODE=BASE/'code/20261122_generative_full_v04'
ARCHIVE=BASE/'uploads/20261122_generative_full_v05_code.tar.gz'
PART=ARCHIVE.with_suffix(ARCHIVE.suffix+'.part')
EXPECTED='c0423d389a6e1933b42b791b07b8f0047b5e1978eaf531f3056c4e66a36f0b4e'
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for path in (RELEASE,RUN,OLD,SOURCE_CODE,ARCHIVE):
    if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Path escapes owned project')
if PART.exists():
    if digest(PART)!=EXPECTED:raise ValueError('Uploaded source hash differs')
    PART.replace(ARCHIVE)
if digest(ARCHIVE)!=EXPECTED:raise ValueError('Archive hash differs')
if RELEASE.exists():
    if json.loads((RELEASE/'release_complete.json').read_text())['code']!=EXPECTED:raise ValueError('Immutable release differs')
else:
    staging=RELEASE.parent/('._gen_v05_'+EXPECTED[:12]);staging.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        for member in archive:
            if not(staging/member.name).resolve().is_relative_to(staging.resolve()) or not(member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member')
        archive.extractall(staging,filter='data')
    for name,row in json.loads((staging/'code_manifest.json').read_text())['files'].items():
        if digest(staging/name)!=row['sha256']:raise ValueError('Code hash differs:'+name)
    (staging/'release_complete.json').write_text(json.dumps(dict(code=EXPECTED,revision='v05_fixed_products')))
    staging.rename(RELEASE)
for name,row in json.loads((RELEASE/'code_manifest.json').read_text())['files'].items():
    if digest(RELEASE/name)!=row['sha256']:raise ValueError('Released source changed:'+name)
# The timed mathematical paths are unchanged; only the registered annual
# product-selection protocol and its cost bound differ from the timed release.
for name in ('architecture.py','observations.py','training.py','evaluation.py','preparation.py','data.py','background.py','benchmark.py'):
    relative=Path('src/gnss_field/generative')/name
    if digest(RELEASE/relative)!=digest(SOURCE_CODE/relative):raise ValueError('A timed model/data path changed:'+name)
RUN.mkdir(exist_ok=True)
sys.path[:0]=[str(RELEASE/'src'),str(RELEASE/'tools')]
os.environ.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(RELEASE/'src')+':'+str(RELEASE/'tools'),
                  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
from gnss_field.artifacts import read_json,write_json,immutable_json,fingerprint
from gnss_field.generative.campaign import Campaign,load_config,build_data
config_path=RELEASE/'configs/full_generative_a800_v02.json'
config=load_config(config_path)
for name in ('audit_B.json','audit_C.json'):
    immutable_json(RUN/name,read_json(OLD/name))
measured=read_json(OLD/'benchmark.json')
immutable_json(RUN/'original_benchmark.json',measured)
data=build_data(config)
if fingerprint(data.metadata)!=measured['condition_identity']:raise ValueError('Measured input contract changed')
if data._label_identity('B')!=read_json(RUN/'audit_B.json')['labels_identity']:raise ValueError('Training labels changed')
campaign=Campaign(config)
accounting=campaign.refresh_accounting()
revised=dict(measured)
revised.update(max_final_candidates=10,final_selection_protocol='subset_locked_products',
    final_evaluation_seconds=10*8760*max(measured['final_evaluation_seconds_per_hour_K128'].values())+measured['representation_check_seconds'],
    pricing_revision=dict(source_sha256=digest(OLD/'benchmark.json'),change='40 repeated annual candidates to10 precommitted products',
        scientific_training_unchanged=True,timed_mathematical_sources_equal=True,full_year_selection=False,
        previous_billed_card_seconds=accounting['card_seconds'],no_new_GPU_benchmark=True))
immutable_json(RUN/'benchmark.json',revised)
lock=campaign.lock_budget(revised,accounted_seconds=accounting['card_seconds'])
immutable_json(RUN/'product_selection_protocol.lock.json',dict(protocol='subset_locked_products',max_products=10,
    point_selection='lowest fixed576hour development station macro RMSE',joint_selection='lowest fixed576hour group ES',
    annual_checkpoint_selection=False,annual_probability_members=128,member_sensitivity_prefix=64,
    control_products=['point'],shared_products=['joint'],complete_products=['point','joint']))
write_json(RUN/'preflight_status.json',dict(state='COMPLETE',mode='measured_runtime_reused_fixed_products',
    measured_total_card_hours=lock['total_seconds']/3600,already_billed_card_seconds=accounting['card_seconds'],updated_unix=time.time()))
queue=subprocess.run(['squeue','-u',os.environ.get('USER','u23114'),'-h','-o','%i|%P|%T|%j'],check=True,capture_output=True,text=True).stdout
if any('a800' in line.split('|')[1].lower() and line.split('|')[-1].startswith(('gvd-','fg-')) for line in queue.splitlines()):
    raise ValueError('An allocation from this GNSS campaign is still active')
result=campaign.submit(config_path)
write_json(RUN/'first_submission.json',dict(result=result,total_budget_card_hours=lock['total_seconds']/3600,
    release=str(RELEASE),run=str(RUN),config=str(config_path),created_unix=time.time()))
print(json.dumps(dict(stage='FORMAL_PIPELINE_SUBMITTED',result=result,run=str(RUN),
    budget_card_hours=lock['total_seconds']/3600,full_year_products=10),ensure_ascii=False),flush=True)
print('RUNTIME_V05_ACTIVATION_COMPLETE',flush=True)
