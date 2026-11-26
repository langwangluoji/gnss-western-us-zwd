"""Final read-only reconciliation of existing calendars; never join time gaps for increments."""
from pathlib import Path
import numpy as np
from campaign import RESULTS,read
from directed_v2_summary import locations
from paired_mean_summary import paired_interval
from tools.terrain_diffusion.data import sha,write_json


def level(error):
    valid=np.isfinite(error);n=valid.sum((2,3));square=np.nansum(error*error,(2,3))
    station=np.sqrt(np.divide(square,n,out=np.full(n.shape,np.nan),where=n>0))
    count=valid.sum((1,2,3,4));mse=np.nansum(error*error,(1,2,3,4))/count
    return dict(macro_rmse_mm=np.nanmean(station,(0,1,2)).tolist(),micro_rmse_mm=np.sqrt(mse).mean(0).tolist(),
        mae_mm=(np.nansum(abs(error),(1,2,3,4))/count).mean(0).tolist(),bias_mm=(np.nansum(error,(1,2,3,4))/count).mean(0).tolist())


def main():
    root=RESULTS/'20261126_directed_lag_v2';errors={};sources={};truths={};refs={}
    for day in (8,15):
        summary=read(root/f'paired_calendar{day}/summary.json')
        for arm in ('H_delta','factorized','joint'):
            per_seed=[];reference=[]
            for seed in (42,17):
                _,_,folder=locations(RESULTS,arm,seed,'last',day)
                expected=summary['sources'][f'{arm}/last/s{seed}']['prediction_sha256']
                if sha(folder/'evaluation.npz')!=expected:raise ValueError('existing prediction changed')
                with np.load(folder/'evaluation.npz') as z:
                    p=z['prediction'].astype('float64');y=z['truth'].astype('float64');r=z['reference'].astype('float64')
                if (day,seed) in truths:
                    np.testing.assert_array_equal(y,truths[day,seed]);np.testing.assert_array_equal(r,refs[day,seed])
                else:truths[day,seed]=y;refs[day,seed]=r
                per_seed.append(p-y);reference.append(r-y);sources[f'{day}/{arm}/{seed}']=expected
            errors[day,arm]=np.stack(per_seed)
            if arm=='joint':errors[day,'I1_persistence']=np.stack(reference)
    rows=[]
    for arm in ('I1_persistence','H_delta','factorized','joint'):
        a=errors[8,arm];b=errors[15,arm];joined=np.concatenate([a,b],axis=3)
        metrics=level(joined)
        delta=np.concatenate([np.diff(e[...,0],axis=3) for e in (a,b)],axis=3)
        valid=np.isfinite(delta);count=valid.sum((2,3));mse=np.divide(np.nansum(delta*delta,(2,3)),count,out=np.full(count.shape,np.nan),where=count>0)
        metrics['current_increment_macro_rmse_mm']=float(np.nanmean(np.sqrt(mse)))
        inc=np.diff(joined,axis=-1);metrics['internal_increment_micro_rmse_mm']=float(np.mean(np.sqrt(np.nanmean(inc*inc,axis=(1,2,3,4,5)))))
        metrics['main_R_F6']=np.asarray(level(a)['macro_rmse_mm'])[[0,6]].tolist();metrics['supplement_R_F6']=np.asarray(level(b)['macro_rmse_mm'])[[0,6]].tolist()
        rows.append(dict(arm=arm,**metrics))
    ref=next(r for r in rows if r['arm']=='I1_persistence')
    np.testing.assert_allclose(ref['main_R_F6'][0],9.817064,atol=1e-6,rtol=0)
    np.testing.assert_allclose(ref['supplement_R_F6'][0],10.740343,atol=1e-6,rtol=0)
    np.testing.assert_allclose(ref['macro_rmse_mm'][0],10.319804,atol=1e-6,rtol=0)
    comparisons={}
    for candidate,reference in [('joint','I1_persistence'),('joint','H_delta'),('joint','factorized')]:
        a=np.concatenate([errors[d,candidate] for d in (8,15)],3)[...,[0,6]]
        b=np.concatenate([errors[d,reference] for d in (8,15)],3)[...,[0,6]]
        comparisons[candidate+'_vs_'+reference]=paired_interval(a,b)
    out=RESULTS/'20261126_final_research_summary';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'point_metrics.json',dict(rows=rows,combined_R_F6_delta_ci95_mm=comparisons,sources=sources,code_sha256=sha(__file__),
        population='existing72 exposed sample days,1728 unique nominal hours x3 layouts, two seeds; no new fitting or dates',
        increment='differences separately inside each72h calendar segment; never across date gaps',strict_oof=False,confirmation_C_D_read=False))
    print(rows,flush=True);print(comparisons,flush=True)


if __name__=='__main__':main()
