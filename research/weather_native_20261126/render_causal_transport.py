"""Render the tested causal-state / physical-value architecture, without Mermaid."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/figures/weather_native_20261126'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(16,11));ax.set(xlim=(0,16),ylim=(0,11));ax.axis('off')
    def box(x,y,w,h,text,color='#e5edf6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=color,edgecolor='#506576'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5,color='#183343')
    def arrow(start,end,dashed=False):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.3,color='#506576',linestyle='--' if dashed else '-'))
    ax.text(.2,10.6,'Causal station-state learning with explicit innovation / increment transport',fontsize=17,weight='bold')
    ax.text(.2,10.15,'NativePhysicsLM: same chronological cache, same 72h state history, zero-initialized mean corrections',fontsize=11)
    box(.4,8.3,5.5,1.25,'Visible station sequences: GNSS ZWD / T / P\nSeparate masks and ages; source-model errors\nIdentity, coordinates, terrain, coast, calendar')
    box(8.7,8.3,6.5,1.25,'Query features and station-query relations\nSigned east / north, distance, height, roughness, coast\nNo target observations or future weather')
    box(.7,6.5,5,1.05,'Per-station GRU / LSTM / causal Transformer\nRetain each station identity through time','#e8e5f6');arrow((3.2,8.23),(3.2,7.62))
    box(.4,4.45,4.6,1.15,'Historical-prefix auxiliary head\nPredict +1...6h GNSS / T / P changes\nTargets lie inside observed history','#f8e7d8');arrow((2.5,6.43),(2.5,5.67),True)
    box(6.1,6.35,4.5,1.35,'Terrain / coast query attention\n4 radii: 50 / 100 / 200 / 400 km\nDirectional bias and values','#e1eee5');arrow((5.77,7),(6.03,7));arrow((11.9,8.23),(8.35,7.77))
    box(11.6,6.35,3.9,1.35,'Explicit physical value paths\nObserved lagged innovations\nPredicted future ZWD increments','#e1eee5');arrow((10.67,7),(11.53,7))
    box(5.7,4.25,4.9,1.4,'Current head: zero initial gate\nLag 0 / 1 / 3 / 6 / 12 / 24h innovations\nHeight-adjusted transport; +/-5 mm','#e5edf6')
    box(11.5,4.25,4.1,1.4,'Future head: zero initial output\nStation-predicted +1...6h changes\nHeight-adjusted transport; +/-30 mm','#e5edf6')
    arrow((8.35,6.28),(8.35,5.72));arrow((13.55,6.28),(13.55,5.72))
    box(5.7,2.5,9.9,1.0,'Frozen I1 current field + learned correction -> deterministic 0...6h mean\nFreeze candidate -> training-only residual centering / scaling','#e2efeb')
    arrow((8.35,4.18),(8.35,3.57));arrow((13.55,4.18),(13.55,3.57))
    box(.5,.45,15,.95,'Conditional Gaussian / Gaussian copula / native-space v / full graph-time frequency tokens\nGaussian-preconditioned residual diffusion; matched DDIM40 and Heun20; restore physical mm','#fff0d9')
    arrow((10.65,2.43),(10.65,1.47))
    ax.text(.6,2.7,'Dashed branch: training supervision\nnever a future inference input',fontsize=9,color='#506576')
    for extension in ('png','svg'):fig.savefig(OUT/f'causal_transport.{extension}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    path=OUT/'causal_transport.svg';path.write_bytes(('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))


if __name__=='__main__':main()
