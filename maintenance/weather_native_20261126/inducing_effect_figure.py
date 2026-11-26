"""Paired effect sizes and month intervals, complementing absolute-score plots."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from campaign import RESULTS


def main():
    out=RESULTS/'20261126_inducing_latent_flow/paired_calendar8_v2'
    data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    edges=['affine_time_vs_gaussian_time','odd_frequency_vs_affine_frequency']
    labels=['Affine time - Gaussian','Odd frequency - Affine frequency']
    panels=[('crps_h6','F6 CRPS difference (mm)'),('increment','Hourly increment CRPS difference (mm)'),
        ('pair_0-50_h6','Near-pair F6 CRPS difference (mm)'),('energy','Whole-field energy difference (mm)'),
        ('event20_increase_brier','20 mm increase: Brier difference'),('event20_decrease_brier','20 mm decrease: Brier difference')]
    fig,axes=plt.subplots(3,2,figsize=(13,8),layout='constrained')
    for ax,(key,title) in zip(axes.flat,panels):
        for j,edge in enumerate(edges):
            row=data['comparisons'][edge][key];value=row['delta'];lo,hi=row['delta_ci95']
            assert lo<=value<=hi
            ax.errorbar(value,1-j,xerr=[[value-lo],[hi-value]],fmt='o',capsize=4,color=['#2563eb','#d97706'][j])
        ax.axvline(0,color='#64748b',lw=1);ax.set(yticks=[1,0],yticklabels=labels,ylim=(-.5,1.5),title=title)
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y',labelsize=8);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis='x',style='sci' if key.startswith('event') else 'plain',scilimits=(0,0),useMathText=True)
    fig.suptitle('Negative differences favor the first model; shared-month 95% intervals, exposed 2020',fontsize=12)
    fig.savefig(out/'paired_effects.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
