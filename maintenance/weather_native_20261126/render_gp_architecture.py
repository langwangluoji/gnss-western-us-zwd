"""Static scientific flowchart of the tested station/GP trajectory pipeline."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parents[2]/'docs/figures/weather_native_20261126'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(16,11));ax.set(xlim=(0,16),ylim=(0,11));ax.axis('off')
    def box(x,y,w,h,text,color='#e6edf5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=color,edgecolor='#526879'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5,color='#153447')
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526879',lw=1.3))
    ax.text(.3,10.6,'Station-causal encoding and Gaussian-reference trajectory diffusion',fontsize=17,weight='bold')
    ax.text(.3,10.15,'2016-2019 continuous-hour windows; 2020 exposed development; current reconstruction and +1...6h forecast',fontsize=11)
    box(.4,8.3,5.2,1.2,'Visible station history [B,72,C,15]\nGNSS / T / P, masks / ages, model errors\nStatic terrain / coast [C,9], ID embedding [C,8]')
    box(6.2,8.3,4.4,1.2,'Per-station GRU / LSTM / causal TF\nWidth64; same effective history\nCausal auxiliary station increments','#e9e6f5')
    box(11.2,8.3,4.4,1.2,'Terrain / coast query attention\nSigned direction and height relations\nExplicit innovation / increment values','#e3eee6')
    arrow((5.67,8.9),(6.13,8.9));arrow((10.67,8.9),(11.13,8.9))
    box(.4,6.5,6.7,1.1,'Separate mean-continuation experiment: C / H / S\nEach source3000 + extra3000 matched chronological blocks\nGP comparison here keeps the original source3000 mean','#e3eee6')
    box(8,6.5,7.6,1.1,'Freeze deterministic mean f, then r = (y - f - b) / s\nS175 condition = fused64 + known-query111\nFrozen Gaussian g(c), d(c); standardized u = (r - g) / d','#fff0d9')
    arrow((13.4,8.23),(11.8,7.67))
    box(.4,4.4,4.5,1.3,'G-ref: no new training\nGP initial state As eta At\nAs = principal square root of Ks\nAt = principal square root of Kt','#edf0f1')
    box(5.7,4.4,4.5,1.3,'Full: space + time whitening\nz = inverse(As) u inverse(At)\nGraph / time v, 45,639 parameters\nGenerate, then As z At','#e9e6f5')
    box(11,4.4,4.6,1.3,'Local / Affine: time whitening\nw = u inverse(At); GP initial As eta\nNonlinear v:44,692; affine v:44,944\nGenerate locally, then w At','#e9e6f5')
    for x in (2.65,7.95,13.3):arrow((11.8,6.43),(x,5.77))
    box(.5,2.55,15,.95,'Missing labels: group complete7h trajectories BEFORE whitening; inference always keeps all137 registered queries\nShared k per origin and IID epsilon streams; exact alpha endpoints 1 / 0; zero initial velocity inherits G-ref under Heun0','#f8e7dc')
    for x in (2.65,7.95,13.3):arrow((x,4.33),(x,3.57))
    box(.5,.55,15,1.15,'Heun0 x20 =39 neural calls; DDIM x40 =40 neural calls; fixed K32 IID members\nRestore y = f + b + s * [ g(c) + d(c) * generated_u ]  (* means elementwise)\nReport mean RMSE, fair CRPS, analytic Gaussian / rank coverage, station-pair and trajectory-increment scores','#e3eee6')
    arrow((8,2.48),(8,1.77))
    for extension in ('png','svg'):fig.savefig(OUT/f'gp_trajectory_process.{extension}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    svg=OUT/'gp_trajectory_process.svg'
    svg.write_bytes(('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))


if __name__=='__main__':main()
