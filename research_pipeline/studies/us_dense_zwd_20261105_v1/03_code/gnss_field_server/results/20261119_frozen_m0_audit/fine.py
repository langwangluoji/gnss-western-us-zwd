from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
d=json.loads((p/'release/seed_42/mean_base/full2019/5000/metrics.json').read_text())['metrics']['fine']
fig,ax=plt.subplots(1,2,figsize=(9,4))
for i,(key,key100) in enumerate([('model','coarsened'),('model_increments','coarsened_increments')]):
 values=([d[key]['station_macro_rmse_mm'],d[key100]['station_macro_rmse_mm']] if i==0 else [d[key]['0-50']['group_macro_rmse_mm'],d[key100]['0-50']['group_macro_rmse_mm']])
 bars=ax[i].bar(['M0, 25 km','M0, regridded 100 km'],values,color=['steelblue','grey']);ax[i].bar_label(bars,fmt='%.3f');ax[i].set_ylim(0,max(values)*1.17)
ax[0].set(title='Point skill on complete stencils',ylabel='Station-macro surface RMSE (mm)',xlabel=str(d['model']['n_stations'])+' stations')
z=d['model_increments']['0-50'];ax[1].set(title='0-50 km paired increments',ylabel='Group-macro common-height RMSE (mm)',xlabel=f"{z['n_pairs']} pair, {z['n_pair_hours']} pair-hours")
fig.tight_layout();fig.savefig(p/'m0_fine_scale.png',dpi=160)
