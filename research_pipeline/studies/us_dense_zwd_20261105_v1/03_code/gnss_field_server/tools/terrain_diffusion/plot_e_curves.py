import json,re,glob
from pathlib import Path
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5)); any=False
for p in glob.glob('results/20261124_E*_short_console.log'):
 e=Path(p).name.split('_')[1]; s=Path(p).read_text(errors='ignore'); ys=[]
 for line in s.splitlines():
  try:
   d=json.loads(line); ys.append(d.get('monitor_rmse_mm'))
  except: pass
 ys=[x for x in ys if x is not None]
 if ys:
  plt.plot(ys,label=e); any=True
if any:
 plt.xlabel('Logged monitor event'); plt.ylabel('Monitor RMSE (mm)'); plt.legend(ncol=3); plt.tight_layout(); plt.savefig('results/20261124_E_training_curves.png',dpi=180); plt.close()
