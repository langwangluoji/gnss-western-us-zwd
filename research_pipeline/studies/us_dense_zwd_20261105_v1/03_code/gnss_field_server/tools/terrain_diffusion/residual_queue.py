"""Serial overnight queue; atomic progress, fail-fast dependencies and result receipts."""
import argparse,csv,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import Dataset,sha,write_json,ROOT as REPO
from .metrics import evaluate
from .residual_experiment import ROOT,prepare,foundation_run,choose_base,fit_probability,tree_correction,load_pack,score_samples

EXPERIMENTS=[('E1201_mlp','foundation','mlp'),('E1202_cnn_lstm','foundation','cnn_lstm'),('E1203_xgboost','foundation','xgboost'),('E1204_xgb_residual','tree',False),('E1205_correlated_gaussian','probability','gaussian'),('E1206_diffusion','probability','diffusion'),('E1207_diffusion_xgb','tree',True)]


def progress(root,**values):
    values['updated_unix']=time.time();tmp=root/'status.tmp';write_json(tmp,values);os.replace(tmp,root/'status.json');print(json.dumps(values),flush=True)


def report(root):
    data=Dataset(device='cpu');records=[];audits=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for eid,_,_ in EXPERIMENTS:
        for seed in (42,17):
            sd=root/eid/f's{seed}'
            for scope in ('selection','expanded'):
                path=sd/scope/'metrics.json'
                if not path.exists():continue
                score=json.loads(path.read_text());p=load_pack(path.parent/'evaluation.npz');difference=0
                for li in range(3):
                    ix=np.flatnonzero(p['layout']==li);new=evaluate(p['samples'][:,ix],p['truth'][ix])
                    for k,v in new.items():difference=max(difference,abs(v-score['layouts'][li][k]))
                assert np.isfinite(p['samples']).all() and difference<1e-5
                canonical=data.a['val_y'][p['row'],12+p['hour']]
                np.testing.assert_allclose(p['truth'],canonical,rtol=1e-6,atol=5e-5,equal_nan=True)
                audits.append(dict(experiment=eid,seed=seed,scope=scope,max_difference=difference,canonical_truth_matches=True))
                records.append(dict(experiment=eid,seed=seed,scope=scope,macro_rmse_mm=score['macro_rmse_mm'],crps_mm=score['crps_mm'],increment_crps_mm=float(np.mean([l['increment']['fair_crps_mm'] for l in score['layouts']]))))
            for hp in sd.rglob('history.csv'):
                import pandas as pd
                h=pd.read_csv(hp);x=h['step'];key='validation_rmse_mm' if 'validation_rmse_mm' in h else 'monitor_rmse_normalized'
                fig2,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(x,h.loss);ax[1].plot(x,h[key]);ax[0].set(title='Training objective',xlabel='Updates');ax[1].set(title=key,xlabel='Updates');fig2.tight_layout();fig2.savefig(hp.parent/'learning_curve.png',dpi=140);plt.close(fig2)
            for hp in sd.rglob('tree_history.json'):
                h=json.loads(hp.read_text());fig2,ax=plt.subplots(figsize=(6,4))
                for name,v in h.items():ax.plot(v['rmse'],label=name)
                ax.legend();ax.set(xlabel='Boosting round',ylabel='Normalized residual/target RMSE');fig2.tight_layout();fig2.savefig(hp.parent/'learning_curve.png',dpi=140);plt.close(fig2)
    if records:
        with (root/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
        for ax,key in zip(axes,['macro_rmse_mm','crps_mm']):
            subset=[r for r in records if r['scope']=='expanded'];labels=sorted(set(r['experiment'] for r in subset));means=[np.mean([r[key] for r in subset if r['experiment']==l]) for l in labels]
            ax.bar(range(len(labels)),means);ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set(ylabel=key,title='12 month-start days x24h')
        fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150)
    plt.close(fig);write_json(root/'artifact_audit.json',dict(status='passed',records=audits))
    rows=[]
    for eid,_,_ in EXPERIMENTS:
        rs=[r for r in records if r['experiment']==eid and r['scope']=='expanded']
        if rs:rows.append('| '+eid+' | '+f"{np.mean([r['macro_rmse_mm'] for r in rs]):.4f} | {np.mean([r['crps_mm'] for r in rs]):.4f} | {np.mean([r['increment_crps_mm'] for r in rs]):.4f} |")
    text='# v12 overnight residual field results\n\n'+('All experiments completed.\n' if len(records)==28 else 'Queue partial; status.json is authoritative.\n')
    text+='\nTraining410/validation137; allowed ZWD/T/P only. Frozen confirmation138 and C/D unread. Three spatial-fold foundation predictions train downstream models; final foundations are refitted. Same2018 development calendar, not a blind test. Radial CNN-LSTM uses12h observable history; stochastic temporal coherence is not trained.\n\n| Experiment | Macro RMSE/mm | Fair CRPS/mm | 1h increment CRPS/mm |\n|---|---:|---:|---:|\n'+'\n'.join(rows)
    text+='\n\nK1 deterministic CRPS is MAE; probability arms K16 at expansion (matches post-calibration ensemble features). Independent selection K16. Baseline is selected by two-seed quarterly RMSE. Model-family budgets include its own fold fitting; downstream dependencies are extra cost. Trees can early-stop. Fixed spatial Gaussian covariance is a meaningful correlated baseline, not a learned temporal model. Diffusion has full point residual support but retains query-graph dependence. Post-XGBoost common correction preserves ensemble spread. Historical PWV station-cohort selection remains.\n\n![Comparison](comparison.png)\n\nSource plan: docs/superpowers/plans/2026-11-24-residual-field-v12.md. Per-run folders contain metrics, curves, timing and checkpoints (weights/arrays local). Final interpretation to follow after frozen evaluations and diagnostic review.\n'
    (root/'RESULTS.md').write_text(text,encoding='utf-8',newline='\n')


def sync(root,label):
    rel=root.relative_to(REPO).as_posix()
    subprocess.run(['git','add','--',rel],cwd=REPO,check=True)
    check=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO)
    if check.returncode==1:
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m',label,'--only','--',rel],cwd=REPO,check=True)
        subprocess.run(['git','push','origin','main'],cwd=REPO,check=True)


def sync_safely(root,label):
    try:
        sync(root,label)
        write_json(root/'sync_pending.json',dict(pending=False,label=label,updated_unix=time.time()))
        return True
    except Exception as exc:
        write_json(root/'sync_pending.json',dict(pending=True,label=label,error=repr(exc),updated_unix=time.time()))
        print(json.dumps(dict(stage='sync_pending',error=repr(exc))),flush=True)
        return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--budget-seconds',type=float,default=3600);p.add_argument('--smoke',action='store_true');p.add_argument('--sync',action='store_true');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    lock=root/'queue.lock'
    # Exclusive creation prevents two queues sharing output or GPU.
    started=time.time();completed=[]
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('residual*.py')}
    existing=root/'protocol.json'
    if existing.exists():
        protocol=json.loads(existing.read_text());assert protocol['source_sha256']==sources and protocol['smoke']==a.smoke and protocol['budget_seconds']==a.budget_seconds,'Resume protocol differs; use new experiment directory'
        ready=json.loads((root/'cache/ready.json').read_text())
        assert all(sha(root/'cache'/name)==value for name,value in ready['packs'].items()),'Cache hash mismatch'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        progress(root,state='preparing',completed=completed);prepare(root,a.smoke)
        write_json(root/'protocol.json',dict(experiments=EXPERIMENTS,budget_seconds=a.budget_seconds,smoke=a.smoke,data_sha256=Dataset(device='cpu').contract['data_sha256'],source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('residual*.py')},started_unix=started))
        for eid,stage,kind in EXPERIMENTS:
            if (root/eid/'completion.json').exists():completed.append(eid);continue
            if stage!='foundation' and not (root/'foundation_selection.json').exists():choose_base(root)
            progress(root,state='running',current=eid,completed=completed)
            if stage=='foundation':foundation_run(root,eid,kind,a.budget_seconds,a.smoke)
            elif stage=='probability':fit_probability(root,eid,kind,a.budget_seconds,a.smoke)
            else:tree_correction(root,eid,kind,a.budget_seconds,a.smoke)
            completed.append(eid);report(root);progress(root,state='stage_completed',current=eid,completed=completed)
            if a.sync:sync_safely(root,f'Complete v12 {eid} real-data experiment')
        report(root);progress(root,state='completed',completed=completed,total_seconds=time.time()-started)
        if a.sync:sync_safely(root,'Complete v12 serial residual-field queue')
    except BaseException as e:
        (root/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n');progress(root,state='failed',completed=completed,error=repr(e));raise
    finally:
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
