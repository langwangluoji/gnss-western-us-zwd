from pathlib import Path
import json,statistics,time
import torch
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03'
s=torch.load(root/'stages/control/last.pt',map_location='cpu',weights_only=False)
h=s['history'];rows=[]
for epoch in sorted(set(x['epoch'] for x in h)):
 r=[x for x in h if x['epoch']==epoch]
 rows.append({'epoch_index':epoch,'updates':len(r),'mean_B_mse_mm2':statistics.mean(x['mse_mm2'] for x in r),'first100_B_mse':statistics.mean(x['mse_mm2'] for x in r[:100]),'last100_B_mse':statistics.mean(x['mse_mm2'] for x in r[-100:]),'lr_min':min(x['learning_rate'] for x in r),'lr_max':max(x['learning_rate'] for x in r)})
print(json.dumps({'checked':time.time(),'keys':list(s),'loss_by_epoch':rows,'first_updates':h[:3],'last_updates':h[-3:],'config':{k:v for k,v in s.get('config',{}).items() if k in ['stage','train_years','lr','weight_decay','epochs','patience','min_epochs','late_decay_fraction']}}))
