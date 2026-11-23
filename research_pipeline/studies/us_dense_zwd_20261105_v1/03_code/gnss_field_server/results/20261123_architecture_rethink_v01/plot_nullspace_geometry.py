from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parent
p=ROOT/'nullspace_geometry_demo.npz'
a=np.load(p);r=json.loads((ROOT/'nullspace_geometry_demo.json').read_text())
proj=ROOT.parent.parent
cfg=json.loads((proj/'results/20261122_full_generative_design_v01/config.local.json').read_text())
packed=Path('\\\\?\\'+cfg['data']['packed_root']); st=pd.read_csv(packed/'stations.csv')
with np.load(packed/'geometry.npz') as g:xy=np.c_[g['x'].ravel(),g['y'].ravel()]
fig,axs=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
field=np.where(a['land'],a['delta_Z0'],np.nan);limit=12
extent=[xy[:,0].min()/1000,xy[:,0].max()/1000,xy[:,1].min()/1000,xy[:,1].max()/1000]
im=axs[0].imshow(field,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='equal')
ids=a['fit_ids'];cs=a['C_ids'];axs[0].scatter(st.x.iloc[ids]/1000,st.y.iloc[ids]/1000,s=5,c='black',alpha=.45,label='A+B constraints')
axs[0].scatter(st.x.iloc[cs]/1000,st.y.iloc[cs]/1000,s=18,marker='^',facecolors='none',edgecolors='#22a87a',linewidths=.8,label='C locations (no labels read)')
axs[0].set(title='A smooth perturbation invisible to A+B',xlabel='Lambert x (km)',ylabel='Lambert y (km)');axs[0].legend(fontsize=8,loc='upper left')
fig.colorbar(im,ax=axs[0],label='Common-height field change (mm)',shrink=.8)
change=a['station_delta'];rng=np.random.default_rng(42)
for i,(label,sel,color) in enumerate([('A+B',ids,'#426b9c'),('C',cs,'#c96138')]):
 axs[1].scatter(i+rng.uniform(-.13,.13,len(sel)),change[sel],s=10,c=color,alpha=.6)
axs[1].axhline(0,c='#8a8a8a',lw=.8);axs[1].set_xticks([0,1],['A+B query changes','C query changes']);axs[1].set(ylabel='Surface ZWD change (mm)',title='Same observed values;\ndifferent unobserved values');axs[1].grid(axis='y',alpha=.2)
axs[1].text(.04,.96,f"A+B max: {r['maximum_A_B_query_change_mm']:.1e} mm\nC RMS: {r['C_query_change_rms_mm']:.3f} mm\nField RMS on land: 5.0 mm",transform=axs[1].transAxes,va='top',fontsize=9)
fig.suptitle('Geometry-only identifiability demonstration - not a model forecast',fontsize=12)
fig.savefig(ROOT/'nullspace_geometry_demo.png',dpi=180);plt.close(fig)
print(str(ROOT/'nullspace_geometry_demo.png'))
