"""Static scientific architecture: station identity to projective trajectories."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]


def main():
    fig,ax=plt.subplots(figsize=(15,8));ax.set(xlim=(0,15),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,text,color='#edf4f2'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',fc=color,ec='#527675',lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#5c696a'))
    ax.text(.2,7.7,'Station-causal encoder + weather GP + local trajectory diffusion',fontsize=18,weight='bold')
    box(.2,5.2,3.25,1.7,'Visible history: 72 hourly states\n82 stations, 15 dynamic channels\nGNSS / T / P, mask / age, errors\n9 static + 8 station-ID channels')
    box(4.0,5.2,3.0,1.7,'Per-station temporal encoder\nGRU / LSTM / causal Transformer\nMatched effective history\n64-D station states')
    box(7.55,5.2,3.0,1.7,'Terrain / coast query attention\nRetain directed edges + station ID\nRelative xy / height / coast\n64-D fused + 111-D query = c175')
    box(11.1,5.2,3.55,1.7,'Frozen mean + S175 marginals\nZero-init new mean corrections\n7 leads: current through +6 h\nmu(q,h), sigma(q,h)', '#f7eddf')
    arrow((3.45,6.05),(4,6.05));arrow((7,6.05),(7.55,6.05));arrow((10.55,6.05),(11.1,6.05))
    box(.2,2.7,3.25,1.7,'Visible-weather summary cw90\nMasked 1 / 24 / 72 h statistics\nFixed visible stations only\nc175 + cw90 = c265')
    box(4.0,2.7,3.0,1.7,'Weather Gaussian correlation K\nLead-specific spatial ranges\nWarp / rho + weather time kernel\nPSD; unit diagonal; no grid')
    box(7.55,2.7,3.0,1.7,'Joint GP initial trajectory u0\nAll 137 queries x 7 leads\nLocal principal root Aq: 7 x 7\nz0,q = Aq^-1 u0,q')
    box(11.1,2.7,3.55,1.7,'Local v trajectory flow (c265)\nNonlinear / affine Gaussian control\nHeun0 x 20: 39 NFE\nDDIM x 40: 40 NFE', '#f7eddf')
    arrow((1.8,5.2),(1.8,4.4));arrow((3.45,3.55),(4,3.55));arrow((7,3.55),(7.55,3.55));arrow((10.55,3.55),(11.1,3.55))
    box(4.0,.55,6.55,1.3,'Physical ensemble: y = mu + sigma * Aq zq\nShared GP coupling + query-local temporal transport\nAdding queries preserves existing marginal laws', '#edf1f8')
    arrow((12.85,2.7),(10.55,1.2))
    ax.plot([14.65,14.9,14.9],[6.05,6.05,1.2],color='#5c696a',lw=1.5)
    arrow((14.9,1.2),(10.55,1.2))
    ax.text(.2,.0,'2016-2019: 3000 consecutive 12 h blocks; sample station IDs only.  2020: exposed development.  Diffusion index is not physical time.',fontsize=9,color='#525e63')
    folder=ROOT/'docs/figures/weather_native_20261126';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/f'weather_projective_trajectory.{suffix}',dpi=180,bbox_inches='tight')
    svg=folder/'weather_projective_trajectory.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


if __name__=='__main__':main()
