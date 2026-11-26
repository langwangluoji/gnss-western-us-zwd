"""Architecture figure for exact-mean shape/scale Gaussian-process controls."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    root=Path(__file__).resolve().parents[2];out=root/'docs/figures/weather_native_20261126';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,text,color='#edf4f2'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.12',fc=color,ec='#527675',lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.2,linespacing=1.5)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#5c696a'))
    ax.text(.2,6.65,'Mean-preserving probability controls: scale versus non-Gaussian shape',fontsize=17,weight='bold')
    box(.2,4.35,3.0,1.55,'Frozen source\nTransformer3000 mean: mu_base\nS175 standard deviation: sigma0\nStation-causal condition c265')
    box(3.75,4.35,2.8,1.55,'Frozen weather GP G265\nJoint Gaussian eta with unit diag\n137 queries x 7 leads\nSame initial coupling in all arms')
    box(7.1,4.35,2.8,1.55,'Condition-only coefficient head\nTwo SiLU layers, zero last head\na,b,c: normalized Hermite weights\nOptional positive scale multiplier')
    box(10.45,4.35,3.2,1.55,'Physical ensemble\ny = mu_base + sigma0 * mult * z\nE[y | known history] = mu_base\nStd[y | known history] known', '#f7eddf')
    arrow((3.2,5.12),(3.75,5.12))
    ax.plot([1.7,1.7,8.5],[5.9,6.2,6.2],lw=1.5,color='#5c696a');arrow((8.5,6.2),(8.5,5.9))
    box(2.1,2.45,9.8,1.1,'z = a*eta + b*(eta^2 - 1)/sqrt(2) + c*(eta^3 - 3*eta)/sqrt(6)\na^2 + b^2 + c^2 = 1     =>     E[z] = 0, Var[z] = 1', '#edf1f8')
    arrow((5.15,4.35),(5.15,3.55));arrow((8.5,4.35),(8.5,3.55));arrow((11.9,3.0),(12.05,4.35))
    for x,title,text in [(0.4,'ScaleOnly','44,239 parameters\nLearn mult; b=c=0\nStrict Gaussian, analytic CRPS'),
        (4.9,'ShapeOnly','44,030 parameters\nLearn b,c; mult=1\nFixed mean and marginal variance'),
        (9.4,'ScaleShape','43,801 parameters\nLearn mult,b,c\nCompare with ScaleOnly for shape gain')]:
        box(x,.45,4.1,1.35,title+'\n'+text)
    ax.text(.2,.03,'Train: K16 IID marginal fair CRPS, consecutive2016-2019 blocks. Evaluate: K32 GP-coupled fields, exposed2020 development.',fontsize=9,color='#525e63')
    for extension in ('png','svg'):fig.savefig(out/f'centered_shape_control.{extension}',dpi=180,bbox_inches='tight')
    path=out/'centered_shape_control.svg';path.write_bytes(('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


if __name__=='__main__':main()
