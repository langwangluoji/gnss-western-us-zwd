from pathlib import Path
import json,time
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
c=read(root/'campaign.json');p=read(root/'stages/control/progress.json');s=c['stages']['control'];b=read(root/'budget.lock.json');d=read(root/'diagnostics.json')
def brief(x):return {k:x.get(k) for k in ['status','step','epoch','hours_in_epoch','total_training_hours','elapsed_seconds','telemetry'] if k in x}
def candidate(x):return {'step':x.get('step'),'epoch':x.get('epoch'),'weight_source':x.get('weight_source'),'rmse':x.get('metrics',{}).get('station_macro_rmse_mm'),'mae':x.get('metrics',{}).get('station_macro_mae_mm'),'crps':x.get('metrics',{}).get('CRPS_mm')}
curve=[]
for x in s.get('evaluations',s.get('evaluation_history',[])):curve.append(candidate(x))
if not curve:
 for f in (root/'stages/control').glob('*.jsonl'):
  for line in f.read_text().splitlines():
   try:
    x=json.loads(line)
    if 'metrics' in x:curve.append(candidate(x))
   except Exception:pass
r={'checked':time.time(),'progress':brief(p),'stage':brief(s),'baseline':candidate(s.get('baseline',{}).get('point',{})),'best':candidate(s.get('best',{}).get('point',{})),'latest':[candidate(x) for x in p.get('development',[])],'curve':curve,'stage_files':[x.name for x in (root/'stages/control').iterdir()],'budget_keys':list(b),'budget_totals':{k:v for k,v in b.items() if isinstance(v,(str,int,float))},'control_budget':{k:v.get('control') for k,v in b.items() if isinstance(v,dict) and 'control' in v},'diagnostics_keys':list(d),'diagnostic_scalars':{k:v for k,v in d.items() if isinstance(v,(str,int,float,bool))},'accounted_card_hours':c['accounted_card_seconds']/3600}
print(json.dumps(r))
