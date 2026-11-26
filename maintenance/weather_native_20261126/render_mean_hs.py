"""Static, source-native HS architecture; no raster generation or model scoring."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from campaign import ROOT


def main():
    fig,ax=plt.subplots(figsize=(15,7.8),layout='constrained');ax.set(xlim=(0,15),ylim=(0,8));ax.axis('off')
    palette=dict(old='#edf1f5',new='#e3f1ed',readout='#fbefdf',loss='#f5f0f5')
    boxes={}
    def box(name,x,y,w,h,text,kind='old'):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.10',fc=palette[kind],ec='#425966',lw=1.1)
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#183340',linespacing=1.45);boxes[name]=(x,y,w,h)
    def arrow(a,b,side_a='right',side_b='left',color='#526a76',style='-'):
        def port(name,side):
            x,y,w,h=boxes[name];return {'right':(x+w,y+h/2),'left':(x,y+h/2),'top':(x+w/2,y+h),'bottom':(x+w/2,y)}[side]
        start,end=port(a,side_a),port(b,side_b)
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,lw=1.25,color=color,linestyle=style,connectionstyle='arc3,rad=0.0'))
    ax.text(.25,7.65,'HS: centered innovation readout + same-time spatial messages',fontsize=18,fontweight='bold',color='#183340')
    ax.text(.25,7.25,'Warm start: Transformer step3000; add3000 updates per seed42/17. Source model is trainable; cached p14/I1 field stays fixed.',fontsize=10.5,color='#526a76')
    box('history',.25,5.45,2.3,1.1,'Visible station history\n72 states;15 variables\nstatic9 + station ID8')
    box('S',3.15,5.45,2.2,1.1,'S: same-hour graph\n8 geometry neighbors\nzero-initialized gate','new')
    box('time',5.95,5.45,2.1,1.1,'Causal Transformer\nper-station states64\nno future observations')
    box('query',8.65,5.45,2.25,1.1,'Terrain/coast query\n+ forecast fusion\n111 conditions;4 heads')
    box('nodes',5.95,3.7,2.1,.85,'Inherited node head\n6 leads x GNSS/T/P')
    box('H',8.65,3.35,2.25,1.3,'H: centered innovations\n0/1/3/6/12/24h lags\nzero-initialized readout','readout')
    box('output',11.65,4.35,3.05,1.45,'7-lead mean field\ncached base + bounded\ncorrection from inherited\nforecast route and HS')
    box('base',.25,2.4,2.3,1.05,'Cached p14/I1 field\nand observed innovations\noriginal source retained')
    box('loss',3.2,.75,7.75,1.05,'Matched training objective\nQuery level MSE +0.1 origin-to-origin h0 delta MSE\n+0.25 causal station auxiliary loss','loss')
    box('export',11.65,1.3,3.05,1.15,'Final3000 raw + EMA\n137 queries; exposed2020\nformal accuracy evaluated\nagainst C/H/S controls')
    for a,b in [('history','S'),('S','time'),('time','query'),('query','output')]:arrow(a,b)
    arrow('time','nodes','bottom','top');arrow('query','H','bottom','top',color='#a57538')
    ax.plot([8.05,8.35,8.35],[4.125,4.125,5.9],color='#526a76',lw=1.2)
    ax.add_patch(FancyArrowPatch((8.35,5.9),(8.65,5.9),arrowstyle='-|>',mutation_scale=13,color='#526a76',lw=1.2))
    arrow('H','output',color='#a57538')
    ax.plot([2.55,3.0,8.25,8.25],[2.925,3.1,3.1,4.0],color='#a57538',lw=1.2)
    ax.add_patch(FancyArrowPatch((8.25,4.0),(8.65,4.0),arrowstyle='-|>',mutation_scale=13,color='#a57538',lw=1.2))
    arrow('nodes','loss','bottom','top',style='--')
    ax.plot([2.55,12.1],[2.925,2.925],color='#526a76',lw=1.2)
    ax.add_patch(FancyArrowPatch((12.1,2.925),(12.8,4.35),arrowstyle='-|>',mutation_scale=13,color='#526a76',lw=1.2))
    arrow('output','export','bottom','top')
    ax.plot([11.65,11.2,11.2],[4.35,4.35,1.275],color='#526a76',lw=1.2,ls='--')
    ax.add_patch(FancyArrowPatch((11.2,1.275),(10.95,1.275),arrowstyle='-|>',mutation_scale=13,color='#526a76',lw=1.2,linestyle='--'))
    ax.text(.25,.23,'Same visible information and effective history as C/H/S. The new HS branch is evaluated separately from probability-center postprocessing.',fontsize=10,color='#526a76')
    folder=ROOT/'docs/figures/weather_native_20261126';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/f'mean_hs_architecture.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
