"""Architecture artifact for the fixed directed-lag experiment contract."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def main():
    out=Path(__file__).resolve().parents[2]/'docs/figures/weather_native_20261126';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(15,8.5));fig.patch.set_facecolor('#f7f9fc');ax.set(xlim=(0,15),ylim=(0,8.5));ax.axis('off')
    def box(x,y,w,h,text,color='#e5eefb'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',facecolor=color,edgecolor='#58718e',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#152d48')
    def arrow(a,b,color='#58718e'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.3))
    ax.text(.35,8.1,'Directed station-lag transport on the H_delta carrier',fontsize=18,weight='bold',color='#152d48')
    ax.text(.35,7.65,'Same TF3000 source | +3000 consecutive blocks per arm/seed | raw final3000 primary; EMA parallel',fontsize=10,color='#52667b')
    box(.4,5.65,2.7,1.35,'Visible history\nGNSS / T / P, errors\nmask / age, station ID\n[B,72,C,15]')
    box(3.65,5.75,2.8,1.15,'Causal station Transformer\nFull terrain / coast / ID\n[B,C,72,64]')
    box(.4,3.55,2.7,1.25,'Query conditions + geometry\nPast observations only\n[B,Q,111], [Q,9]')
    box(3.65,3.5,2.8,1.35,'Original H carrier\nCurrent / future readout\nSource embedding h_q\nH correction [B,Q,7]')
    arrow((3.1,6.3),(3.65,6.3));arrow((3.1,4.15),(3.65,4.15));arrow((5.05,5.75),(5.05,4.85))
    box(7.1,5.75,3.1,1.2,'Station x lag keys\nlag = 0,1,3,6,12,24h\n[B,C,6,64]','#e0f0eb')
    box(7.1,3.7,3.1,1.25,'Lead-aware terrain query\nDirection, height, coast\nlag and forecast lead h\nScores [B,Q,7,C,6,4]','#e0f0eb')
    box(7.1,1.55,3.1,1.3,'Four numeric values per token\nself error; hourly GNSS/T/P\nVariable-specific masks\nPool [B,Q,7,4,4]','#fff0dc')
    arrow((6.45,6.3),(7.1,6.3));arrow((6.45,4.15),(7.1,4.3));arrow((8.65,5.75),(8.65,4.95));arrow((8.65,3.7),(8.65,2.85))
    ax.plot([.4,.15,.15,6.75,6.75],[6.3,6.3,1.15,1.15,2.2],color='#ad8450',linewidth=1.3)
    arrow((6.75,2.2),(7.05,2.2),'#ad8450')
    box(11.1,4.65,3.25,1.6,'Matched transport arms\nFactorized: separable scores,\nterminal key shared over lags\nJoint: historical keys and\nspace / lag / lead interactions','#e0f0eb')
    box(11.1,2.05,3.25,1.55,'Zero-initialized lead readout\nBounded extra correction\ncurrent +/-3mm; future +/-6mm\nFinal = I1 cache base + H + extra','#fff0dc')
    arrow((10.2,4.3),(11.1,5.05));arrow((10.2,2.2),(11.1,2.8));arrow((12.7,4.65),(12.7,3.6))
    ax.text(.45,.7,'Loss: original field + cross-origin h0 difference + 0.5 within-origin lead difference + 0.25 causal station auxiliary',fontsize=11,color='#152d48')
    ax.text(.45,.25,'2016-2019 training; exposed sampled2020 development. Confirmation/C/D isolated. Probability sources remain frozen.',fontsize=10,color='#52667b')
    fig.savefig(out/'directed_lag_architecture.png',dpi=180,bbox_inches='tight')
    svg=out/'directed_lag_architecture.svg';fig.savefig(svg,bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))


if __name__=='__main__':main()
