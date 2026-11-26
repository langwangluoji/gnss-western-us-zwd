"""B-only follow-up: exact station-RMSE convex readout and simple controls."""
import spike as s
import numpy as np
from scipy.optimize import minimize
import time,json


def macro_fit(x,y,mask,strength):
    x=np.asarray(x,'f8');y=np.asarray(y,'f8');counts=mask.sum(0);active=counts>0
    x=x[:,active];y=np.where(mask[:,active],y[:,active],0.);mask=mask[:,active]
    w=mask/np.maximum(counts[active],1)[None]
    rms=np.sqrt(np.einsum('tn,tnc->c',w,x*x)/len(counts[active])).clip(1e-6)
    x=x/rms
    gram=np.einsum('tn,tnc,tnd->ncd',w,x,x,optimize=True)
    rhs=np.einsum('tn,tnc,tn->nc',w,x,y,optimize=True)
    yy=np.einsum('tn,tn->n',w,y*y)
    def objective(b):
        gb=gram@b
        r=np.sqrt(np.maximum(yy-2*(rhs@b)+np.einsum('nc,c->n',gb,b),0.)+.01**2)
        return r.mean()+strength*(b@b),((gb-rhs)/r[:,None]).mean(0)+2*strength*b
    result=minimize(objective,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,gtol=1e-9,ftol=1e-13,maxls=40))
    if not result.success:raise ValueError(result.message)
    return result.x/rms,dict(iterations=result.nit,gradient_max=float(abs(result.jac).max()),objective=result.fun)


def main():
    start=time.monotonic();root=s.ROOT/'objective_probe';root.mkdir(exist_ok=True)
    s.immutable_json(root/'plan.lock.json',dict(code=s.sha256(__file__),sources=s.read_json(s.ROOT/'experiment.lock.json')['sources'],
        candidates=['constant','amplitude','geo3_no_height','macro_geo6','macro_geo24','macro_deep_geo24'],
        strengths=[.0001,.01,1.],epsilon_mm=.1,selection='B inner only',C_scoring=False,D_scoring=False,
        purpose='distinguish geometric information from scalar calibration and station-MSE vs station-RMSE objective'))
    data=s.PilotData();tr=data.rows('train');ir=data.rows('inner');fit=data.a['split_fit'];vi=data.a['split_inner'];outer=data.rows('outer');oi=data.a['split_outer']
    allresults={}
    for name in s.sources():
        a=np.load(s.ROOT/name/'features.npz');base,design=a['prediction'],a['design']
        hf=np.exp(-data.a['heights']/3000)
        scalar=np.stack([np.broadcast_to(hf,base.shape),(base-data.a['baseline'][:,0])/10],-1)
        candidates={};preds={};mask=data.a['mask'][np.ix_(tr,fit)];y=(data.a['labels'][np.ix_(tr,fit)]-base[np.ix_(tr,fit)])/10
        variants={'constant':scalar[:,:,:1],'amplitude':scalar,'geo3_no_height':design[:,:,[17,25,33]],
                  'macro_geo6':design[:,:,list(range(17,41,4))], 'macro_geo24':design[:,:,17:], 'macro_deep_geo24':design}
        for method,x in variants.items():
            trials=[]
            for strength in (.0001,.01,1.):
                if method.startswith('macro'): beta,diagnostic=macro_fit(x[np.ix_(tr,fit)],y,mask,strength)
                else:
                    beta,error=s.solve_ridge(x[np.ix_(tr,fit)],y,mask,strength);diagnostic={'normal_error':error}
                p=base+10*(x@beta);score=s.score(data,p,ir,vi)
                trials.append(dict(inner=score,ridge=strength,beta=beta,diagnostic=diagnostic))
            choice=min(trials,key=lambda t:t['inner']['macro_rmse']);candidates[method]=choice
            preds[method]=base+10*(x@choice['beta'])
        selected=min(candidates,key=lambda n:candidates[n]['inner']['macro_rmse'])
        s.immutable_json(root/f'{name}_selection.json',dict(selected=selected,candidates=candidates))
        # Outer scores are reported after lock; none changes first-round selection.
        res={method:dict(inner=candidates[method]['inner'],outer=s.score(data,p,outer,oi)) for method,p in preds.items()}
        allresults[name]=dict(selected=selected,models=res)
        s.write_json(root/f'{name}_results.json',allresults[name])
        np.savez(root/f'{name}_predictions.npz',**{n:p[np.ix_(outer,oi)] for n,p in preds.items()})
        print(json.dumps(dict(source=name,selected=selected,inner=res[selected]['inner']['macro_rmse'],outer=res[selected]['outer']['macro_rmse'])),flush=True)
    s.write_json(root/'results.json',allresults);s.write_json(root/'status.json',dict(stage='COMPLETE',seconds=time.monotonic()-start))


if __name__=='__main__':main()
