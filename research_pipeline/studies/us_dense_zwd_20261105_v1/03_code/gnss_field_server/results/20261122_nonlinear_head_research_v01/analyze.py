"""Reproduce spike metrics, uncertainty, full-grid publication, and figures."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from local_innovation_pilot import Pilot
from local_innovation_report import paired_bootstrap,export_reviewed_case
from gnss_field.artifacts import read_json,write_json,sha256
from gnss_field.runtime import configure

ROOT=Path(__file__).resolve().parent


def main():
    configure(type('R',(),{'cpus':2})(),42)
    d=Pilot();control=ROOT/'pro_optimizer_control_v01';r=read_json(control/'results.json')
    spec=importlib.util.spec_from_file_location('research_spike',control/'experiment.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    hours=np.asarray(d.lock['outer_hours']);oi=d.indices(hours)
    week=read_json(ROOT.parents[1]/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json')
    starts=np.array([np.datetime64(x,'h').astype('i8') for x in week['week_starts']])
    checks={};intervals={}
    for fold in (0,1):
        directory=control/f'fold_{fold}';saved=np.load(directory/'predictions.npz');rr=r['folds'][str(fold)]
        assert np.array_equal(saved['hours'],d.a['hours'])
        for arm,row in rr['rows'].items():
            value=d.metric(saved[arm][oi],hours,d.splits[fold]['outer'])['station_macro_rmse_mm']
            assert abs(value-row['outer']['station_macro_rmse_mm'])<1e-10
        chosen=rr['selected'];pop=d.a['valid'][oi]&d.splits[fold]['outer'][None]
        intervals[str(fold)]=dict(selected=paired_bootstrap(d.a['truth'][oi],saved[chosen][oi],d.a['mean'][oi],pop,hours,starts,d.a['xy']),
            nonlinear_vs_linear=paired_bootstrap(d.a['truth'][oi],saved['nonlinear_ridge_finetune'][oi],saved['linear_ridge'][oi],pop,hours,starts,d.a['xy']))
        if chosen=='linear_ridge':
            z=np.load(directory/'linear_ridge.npz');center=torch.tensor(z['center'],device='cuda');std=torch.tensor(z['std'],device='cuda')
            coeff=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
            net=None
        else:
            z=torch.load(directory/f'{chosen}.pt',map_location='cpu',weights_only=False)
            center=z['center'].cuda();std=z['std'].cuda();net=m.Correction(m.ResidualFeatures(),np.zeros(33)).cuda();net.load_state_dict(z['state'])
            coeff=None
        class Published(nn.Module):
            def forward(self,features,J):
                x=(torch.cat([features,J/d.settings['grid_scale']],-1)-center)/std
                return d.settings['grid_scale']*(net(x) if net is not None else x@coeff[:-1]+coeff[-1])
        wrapper=Published().cuda().eval()
        predicted=d.predict(wrapper,d.a['hours'])
        error=float(np.max(abs(predicted-saved[chosen])))
        if error>.001:raise ValueError('Reloaded nonlinear predictor mismatch')
        out=directory/'publication';out.mkdir(exist_ok=True)
        export_reviewed_case(d,fold,wrapper,out)
        field=read_json(out/'publication_check.json')
        checks[str(fold)]=dict(reloaded_prediction_max_mm=error,full_grid_query_max_mm=field['max_query_error_mm'])
    write_json(ROOT/'paired_intervals.json',intervals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors={'nonlinear_zero_finetune':'#d95f02','nonlinear_ridge_finetune':'#1b9e77'}
    for fold in (0,1):
        rr=r['folds'][str(fold)];base=rr['base_inner_RMSE']
        for arm,color in colors.items():
            row=rr['rows'][arm];h=[x for x in row['history'] if x['product']=='raw']
            axes[fold,0].plot([x['step'] for x in h],[x['inner']['station_macro_rmse_mm']-base for x in h],label=arm.replace('nonlinear_',''),color=color)
        for arm,c in [('linear_ridge','#377eb8'),('nonlinear_ridge','#7570b3')]:
            axes[fold,0].axhline(rr['rows'][arm]['inner']['station_macro_rmse_mm']-base,label=arm,ls='--',color=c)
        axes[fold,0].axhline(0,color='grey',lw=.8);axes[fold,0].set(title=f'Fold {fold}: spatial x weather inner',xlabel='Update, constant LR 1e-4',ylabel='RMSE minus base (mm)');axes[fold,0].legend(fontsize=8)
        keys=['zero','linear_ridge','nonlinear_ridge','nonlinear_zero_finetune','nonlinear_ridge_finetune']
        axes[fold,1].bar(range(5),[rr['rows'][k]['outer']['station_macro_rmse_mm']-rr['rows']['zero']['outer']['station_macro_rmse_mm'] for k in keys],color=['#aaa','#377eb8','#7570b3','#d95f02','#1b9e77'])
        axes[fold,1].set_xticks(range(5),['Zero','Linear','Nonlinear\nridge','Zero init','Ridge init']);axes[fold,1].set(title=f'Fold {fold}: locked outer288',ylabel='RMSE minus base (mm)')
    fig.savefig(ROOT/'nonlinear_optimization_results.png');plt.close(fig)
    crit=read_json(ROOT/'critic_diagnostics.json');fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for fold in (0,1):
        a=[x for x in crit[str(fold)]['rows'] if x['population']=='outer']
        axes[0].scatter(np.full(len(a),fold),[x['weighted_adversarial_to_ES_gradient_ratio'] for x in a],s=18,alpha=.6)
        axes[1].scatter(np.full(len(a),fold),[x['gradient_cosine'] for x in a],s=18,alpha=.6)
    for ax in axes:ax.set_xticks([0,1],['Fold 0','Fold 1'])
    axes[0].set(ylabel='||gradient .1 adversarial|| / ||gradient ES||',title='Archived GAN: observation-space gradients')
    axes[1].set(ylabel='Gradient cosine',title='19 cached weather-held hours');axes[1].axhline(0,color='grey',lw=.8)
    fig.savefig(ROOT/'critic_gradient_diagnostics.png');plt.close(fig)
    write_json(ROOT/'validation.json',dict(complete=True,kind='research spike numerical validation; no production model change',
        metric_reproduction_max_mm=1e-10,query_checks=checks,local_gate_passed=r['local_gate_passed'],
        neural_updates_total=4752,server_submissions=False,C2019_evaluated=False,D2020=False,
        source_files={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'experiment.py',control/'experiment.py',ROOT/'critic_probe.py',ROOT/'analyze.py']},
        main_production_tests='previous323passed; production source unchanged this task'))
    print('VALIDATED',checks,flush=True)


if __name__=='__main__':main()
