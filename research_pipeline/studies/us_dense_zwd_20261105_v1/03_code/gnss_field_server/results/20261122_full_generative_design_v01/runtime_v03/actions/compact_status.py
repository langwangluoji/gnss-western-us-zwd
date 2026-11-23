from pathlib import Path
import json,subprocess,time
base=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117';run=base/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def compact(v):
 if isinstance(v,dict):return {k:compact(x) for k,x in v.items() if k not in ('hours','hour_ids','eligible_hours','evaluation_history','history','evaluations','records','development_hours')}
 if isinstance(v,list):return [compact(x) for x in v] if len(v)<24 else {'length':len(v),'first':compact(v[:2]),'last':compact(v[-2:])}
 return v
c=read(run/'campaign.json')
r={'checked_unix':time.time(),'queue':subprocess.run(['squeue','-u','u23114','-h','-o','%i|%P|%T|%M|%j|%R'],capture_output=True,text=True,check=True).stdout,'campaign':compact(c),'files':[str(p.relative_to(run)) for p in run.glob('*.json')],'progress':{str(p.relative_to(run)):compact(read(p)) for p in run.glob('stages/*/progress.json')},'logs':{p.name:p.read_text(errors='replace')[-1800:] for p in sorted(run.glob('*.err'),key=lambda p:p.stat().st_mtime)[-4:]} ,'dispatch_logs':{p.name:p.read_text(errors='replace')[-1800:] for p in sorted(run.glob('dispatch-*.out'),key=lambda p:p.stat().st_mtime)[-3:]}}
print(json.dumps(r))
