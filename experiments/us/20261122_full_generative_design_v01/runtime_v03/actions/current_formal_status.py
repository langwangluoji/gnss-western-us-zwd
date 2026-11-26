from pathlib import Path
import json,subprocess,time
base=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
run=base/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03'
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
result={'checked_unix':time.time(),'run':str(run),'queue':subprocess.run(['squeue','-u','u23114','-h','-o','%i|%P|%T|%M|%j|%R'],capture_output=True,text=True,check=True).stdout,'campaign':read(run/'campaign.json'),'first_submission':read(run/'first_submission.json'),'preflight':read(run/'preflight_status.json'),'progress':{str(p.relative_to(run)):read(p) for p in run.glob('stages/*/progress.json')},'logs':{p.name:p.read_text(errors='replace')[-4500:] for p in sorted(run.glob('*.out'),key=lambda p:p.stat().st_mtime)[-6:]}}
print(json.dumps(result,ensure_ascii=False))

