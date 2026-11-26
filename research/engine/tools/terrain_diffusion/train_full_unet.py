from pathlib import Path
import json,time,torch
from tools.terrain_diffusion.unet_foundation import Experiment

def main():
 out=Path('results/20261124_full_data_unet24_long'); out.mkdir(parents=True,exist_ok=True)
 data=Path('results/20261124_terrain_diffusion_ab_full_data'); exp=Experiment(data)
 (out/'protocol.json').write_text(json.dumps({'data_path':str(data),'train_sequences':len(exp.data.a['train']),'train_events':len(exp.fit),'validation_sequences':len(exp.data.a['val_y']),'budget_per_seed':600,'seeds':[42,17],'model':'GridUNet24','confirmation_C_D_read':False},indent=2),encoding='utf-8')
 rows=[]
 for s in (42,17):
  f=out/f's{s}'; m=exp.fit_model('unet',24,s,f,600,999); metrics=exp.evaluate(m,'unet',f); rows.append({'seed':s,'metrics':metrics}); (f/'completion.json').write_text(json.dumps({'status':'completed','metrics':metrics},indent=2),encoding='utf-8'); del m; torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
main()
