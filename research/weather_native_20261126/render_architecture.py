"""Readable repo-native architecture and Gaussian-reference solver diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json


def main():
    out=ROOT/'docs/figures/weather_native_20261126';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(15,11));ax.set(xlim=(0,15),ylim=(0,11));ax.axis('off')
    colors=dict(input='#e4edf7',source='#e6eee7',encoder='#e5e4f4',prob='#fff0db',result='#e0f1ed')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=colors[color],edgecolor='#536677',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,linespacing=1.6,color='#1c3243')
    def arrow(a,b,label=''):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color='#536677'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2,label,fontsize=8,ha='center',backgroundcolor='white')
    ax.text(.2,10.6,'Weather-conditioned native station encoders and residual diffusion',fontsize=18,weight='bold',color='#1c3243')
    ax.text(.2,10.15,'Current reconstruction R (h=0) and future prediction F (h=1...6) are scored separately.',fontsize=11,color='#536677')
    box(.25,8.25,4.0,1.35,'Visible GNSS ZWD / T / P\nMasks + ages + fixed station identities\nTerrain, coast, coordinates and calendar','input')
    box(5.1,8.25,4.2,1.35,'Frozen GRU+p14 and I1\nSame-station observation - model error\n72h state history / 146h raw span','source')
    box(10.15,8.25,4.25,1.35,'One chronological cache\n12h blocks; same condition set throughout\nNo future observations in conditions','input')
    arrow((4.32,8.9),(5.03,8.9));arrow((9.37,8.9),(10.08,8.9))
    for x,kind in [(1.,'GRU'),(5.5,'LSTM'),(10.,'Causal Transformer')]:
        box(x,6.1,3.4,1.1,f'{kind}\nPer-station causal encoding\nSame 72h / width64 / 3000 blocks','encoder')
        arrow((12.25,8.18),(x+1.7,7.28))
    box(3.6,4.2,7.8,1.15,'Terrain/coast-conditioned query attention\nKeep station direction, height contrast, identity and error history\nZero-initialized mean head: I1 at h=0; origin persistence at h=1...6','encoder')
    for x in (2.7,7.2,11.7):arrow((x,6.03),(7.5,5.42))
    box(.4,2.1,3.2,1.1,'Conditional Gaussian\nWeather-dependent mean / scale\nIndependent + correlated controls','prob')
    box(4.1,2.1,3.2,1.1,'Native-space v diffusion\nNo target-to-grid projection\nFull station trajectories','prob')
    box(7.8,2.1,3.2,1.1,'Graph + frequency-token v\nAll GFT and DCT coefficients\nLocal band mixing; one chain','prob')
    box(11.5,2.1,3.0,1.1,'Gaussian-preconditioned v\nResidual shape + dependencies\nDDIM40 vs Heun20 (40 NFE)','prob')
    for x in (2.,5.7,9.4,13.):arrow((7.5,4.13),(x,3.28))
    box(2.,.35,11.,.95,'Matched seeds and dates: RMSE + increments + fair CRPS + coverage + station-pair differences\nPrimary / frozen supplementary calendars; difficult weather, coast, terrain and sparse support','result')
    for x in (2.,5.7,9.4,13.):arrow((x,2.03),(7.5,1.37))
    fig.savefig(out/'architecture.png',dpi=170,bbox_inches='tight');fig.savefig(out/'architecture.svg',bbox_inches='tight');plt.close(fig)
    svg=out/'architecture.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    counts=np.array([4,10,20,40,100]);ratios=[]
    t=np.linspace(0,1,101);alpha=np.maximum(np.cos((t+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2,1e-5)
    for steps in counts:
        schedule=np.round(np.linspace(100,0,steps+1)).astype(int)
        theta=np.arccos(np.sqrt(alpha[schedule]));ratios.append(float(np.prod(np.cos(np.diff(theta))**2)))
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.plot(counts,ratios,'o-',label='Zero-v deterministic DDIM');ax.axhline(1,color='#35836c',label='Zero-v probability flow / Heun')
    ax.set(xlabel='Sampling steps',ylabel='Output variance / reference variance',title='Gaussian-reference numerical control; not a GNSS accuracy result',ylim=(.45,1.03))
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'reference_variance.png',dpi=170);plt.close(fig)
    write_json(out/'manifest.json',dict(script_sha256=sha(Path(__file__)),figures={p.name:sha(p) for p in out.glob('*') if p.suffix in ('.png','.svg')},
        zero_velocity_reference=dict(steps=counts.tolist(),ddim_variance_ratio=ratios,heun_variance_ratio=[1.]*len(counts))))
    print(json.dumps(dict(directory=str(out),ddim20_variance_ratio=ratios[2])))


if __name__=='__main__':main()
