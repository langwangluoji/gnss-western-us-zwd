"""Standalone scientific architecture figure, with PNG fallback for GitHub."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def main():
    out=Path(__file__).resolve().parents[2]/'docs/figures/weather_native_20261126';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=color,edgecolor='#334155',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#475569',linewidth=1.3))
    ax.text(.1,7.65,'Coupled inducing-innovation flow for continuous-query ZWD fields',fontsize=17,weight='bold')
    ax.text(.1,7.25,'Frozen point mean | Complete Gaussian residual retained | Joint endpoint scoring',fontsize=11,color='#475569')
    box(.2,5.35,3.2,1.25,'Observed GNSS / T / P history\nMasks, age, station identities\nTerrain and coast relations','#e2e8f0')
    box(4.1,6,3.5,.8,'Frozen TF3000 point mean mu','#dbeafe')
    box(4.1,4.8,3.5,.8,'Frozen S175 input scale sigma\nWeather-conditioned GP kernel R','#dbeafe')
    box(8.5,5.1,4.9,1.5,'8 fixed geometry-only anchors\nR = R_perp + W W^T\nGP path = epsilon_perp + W eta','#dbeafe')
    box(.2,2.65,3.2,1.4,'90-D historical summary\n265-D query conditions\nNo future target values','#e2e8f0')
    box(4.1,2.55,4,1.6,'TRAIN: shared latent flow g(eta,c)\nAffine / odd nonlinearity\nTime / complete DCT7\n4 Heun steps; 8 modes x 7 leads','#dcfce7')
    box(8.75,2.55,4.65,1.6,'TRAIN: continuous query decoder\nZ = a[epsilon_perp + W g] + Phi g\nY = mu + sigma Z\nInitial g=eta, a=1, Phi=0','#dcfce7')
    box(2.7,.35,9.6,1.05,'Joint fair scores: marginal CRPS + hourly increment CRPS + station-pair CRPS + energy\nControls: frozen Gaussian and matched affine | Analytic point mean remains mu','#fef3c7')
    arrow((3.5,6),(4,6.4));arrow((3.5,5.7),(4,5.2));arrow((7.7,5.2),(8.4,5.7))
    arrow((3.5,3.35),(4,3.35));arrow((8.2,3.35),(8.65,3.35));arrow((10.9,5),(10.9,4.3))
    ax.plot([7.7,7.95,7.95,13.7,13.7],[6.4,6.4,6.95,6.95,3.4],color='#475569',linewidth=1.3)
    arrow((13.7,3.4),(13.5,3.4));arrow((1.8,5.2),(1.8,4.2))
    ax.plot([3.5,3.7,3.7,8.5,8.5],[2.95,2.95,2.2,2.2,2.95],color='#475569',linewidth=1.3)
    arrow((8.5,2.95),(8.65,2.95))
    arrow((11,2.45),(11,1.5));arrow((5.9,2.45),(5.9,1.5))
    ax.text(.2,.6,'Blue: frozen\nGreen: trained',fontsize=10,color='#334155')
    fig.tight_layout();fig.savefig(out/'inducing_innovation_flow.png',dpi=180);fig.savefig(out/'inducing_innovation_flow.svg');plt.close(fig)
    svg=out/'inducing_innovation_flow.svg';svg.write_bytes(b'\n'.join(line.rstrip() for line in svg.read_bytes().splitlines())+b'\n')


if __name__=='__main__':main()
