"""Verify and visualize v13 frozen foundation outputs; no fitting or selection."""
import json,csv,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import Dataset,write_json,sha
from .residual_experiment import load_pack
from .unet_foundation import ROOT,Experiment,GridUNet,ResidualMLP,GEOMETRY
from .unet_hybrid import Hybrid,HybridExperiment


def run():
    torch.set_num_threads(4);data=Dataset(device='cpu');a=data.a;records=[];terrain=[];receipts=[]
    for folder in sorted(ROOT.glob('E130*/s[0-9]*')):
        p=load_pack(folder/'expanded/evaluation.npz');truth=a['val_y'][p['row'],12+p['hour']]
        np.testing.assert_allclose(p['truth'],truth,atol=3e-5,rtol=1e-6,equal_nan=True);pred=p['samples'][0];assert np.isfinite(pred).all();err=pred-truth
        metrics=json.loads((folder/'expanded/metrics.json').read_text());r=[]
        for li in range(3):
            e=err[p['layout']==li];r.append(np.nanmean(np.sqrt(np.nanmean(e**2,axis=0))))
        np.testing.assert_allclose(np.mean(r),metrics['macro_rmse_mm'],atol=1e-5)
        fit=json.loads((folder/'fit.json').read_text());records.append(dict(experiment=folder.parent.name,seed=int(folder.name[1:]),macro_rmse_mm=metrics['macro_rmse_mm'],micro_rmse_mm=float(np.mean([m['micro_rmse_mm'] for m in metrics['layouts']])),mae_mm=float(np.mean([m['mae_mm'] for m in metrics['layouts']])),bias_mm=float(np.mean([m['bias_mm'] for m in metrics['layouts']])),increment_rmse_mm=float(np.mean([m['increment']['station_macro_rmse_mm'] for m in metrics['layouts']])),**{k:fit[k] for k in ('parameters','steps','best_step','seconds','early_stopped','epochs')}))
        for lo,hi,label in [(-1e6,500,'below500m'),(500,1500,'500to1500m'),(1500,1e6,'above1500m')]:
            ix=(a['val_geo'][:,2]*1000>=lo)&(a['val_geo'][:,2]*1000<hi);scores=[np.nanmean(np.sqrt(np.nanmean(err[p['layout']==li][:,ix]**2,axis=0))) for li in range(3)]
            terrain.append(dict(experiment=folder.parent.name,seed=int(folder.name[1:]),terrain=label,stations=int(ix.sum()),macro_rmse_mm=float(np.mean(scores))))
        station=np.mean([np.sqrt(np.nanmean(err[p['layout']==li]**2,axis=0)) for li in range(3)],axis=0)
        fig,ax=plt.subplots(figsize=(5,6));sc=ax.scatter(a['val_geo'][:,0],a['val_geo'][:,1],c=station,vmin=0,vmax=25,cmap='viridis',s=22);fig.colorbar(sc,ax=ax,label='Station RMSE (mm)',extend='max');ax.set(xlabel='Projected x (km)',ylabel='Projected y (km)',title=folder.parent.name+' '+folder.name);ax.set_aspect('equal');fig.tight_layout();fig.savefig(folder/'station_rmse_map.png',dpi=140);plt.close(fig)
        ix=(p['layout']==0)&(p['row']==6);station_id=int(np.argmax(np.isfinite(truth[ix]).sum(0)));fig,ax=plt.subplots(figsize=(8,3));ax.plot(p['hour'][ix],truth[ix,station_id],label='GNSS');ax.plot(p['hour'][ix],pred[ix,station_id],label='Prediction');ax.set(xlabel='Hour (2018-07-01)',ylabel='ZWD (mm)',title=str(a['val_ids'][station_id]));ax.legend();fig.tight_layout();fig.savefig(folder/'example_sequence.png',dpi=140);plt.close(fig)
        preview=folder/'example_field_mm.npy' if (folder/'example_field_mm.npy').exists() else folder/'native_preview_field_mm.npy'
        if preview.exists():
            grid=np.load(preview).squeeze();g=np.load(GEOMETRY);fig,ax=plt.subplots(figsize=(5,6));im=ax.imshow(np.where(g['region']&g['land'],grid,np.nan),origin='lower',cmap='viridis',vmin=0,vmax=200);fig.colorbar(im,ax=ax,label='ZWD (mm)',extend='both');ax.set(title='2018-07-01 12h: '+folder.parent.name,xlabel='Grid column',ylabel='Grid row');fig.tight_layout();fig.savefig(folder/'example_field.png',dpi=140);plt.close(fig)
        done=json.loads((folder/'completion.json').read_text());assert sha(folder/'best.pt')==done['best_sha256'];receipts.append(dict(experiment=folder.parent.name,seed=int(folder.name[1:]),checkpoint_sha256=done['best_sha256'],canonical_truth_verified=True,macro_rmse_recomputed=True))
    for name,rows in [('summary.csv',records),('terrain.csv',terrain)]:
        with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,4));labels=['MLP residual','U-Net16','U-Net24','Point + U-Net16']
    for i,eid in enumerate(sorted(set(r['experiment'] for r in records))):
        vals=[r['macro_rmse_mm'] for r in records if r['experiment']==eid];ax.bar(i,np.mean(vals),color=['#64748b','#0f766e','#2563eb','#b45309'][i]);ax.scatter([i-.05,i+.05],vals,c='black',s=20,zorder=4)
    ax.axhline(12.065729,color='black',ls='--',label='Fixed height-IDW (v12 same population)');ax.axhline(11.624845,color='#b45309',ls=':',label='v12 Gaussian analytic mean');ax.set_xticks(range(len(set(r['experiment'] for r in records))),labels[:len(set(r['experiment'] for r in records))]);ax.set(ylabel='Station macro RMSE (mm)',title='v13 same-input foundation comparison');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    # Source line-ending normalization after launch is verified explicitly.
    protocol=json.loads((ROOT/'protocol.json').read_text());receipt=json.loads((ROOT/'source_receipt.json').read_text());import subprocess;source=subprocess.check_output(['git','show',receipt['training_git_commit']+':'+receipt['training_source_path']]);normalized=source.replace(b'\r\n',b'\n');candidates=[hashlib.sha256(normalized).hexdigest(),hashlib.sha256(normalized.replace(b'\n',b'\r\n')).hexdigest()];assert protocol['code_sha256'] in candidates
    write_json(ROOT/'verification.json',dict(status='passed',evaluations=receipts,source_matches_up_to_line_endings=True,data_sha256=data.contract['data_sha256'],tests='tests/test_unet_foundation.py',note='Deterministic station losses; no dense pseudo-labels; validation reused for development, confirmation and original C/D unread'))
    exp=Experiment();training=[]
    # Quantify the point-to-grid readout cost before any neural correction.
    baselines=[]
    for name,kind,model in [('E1300P_point_background','mlp',ResidualMLP()),('E1300G_grid_background','unet',GridUNet(16))]:
        folder=ROOT/name;folder.mkdir(exist_ok=True);model=model.cuda().eval();metrics=exp.evaluate(model,kind,folder);baselines.append(dict(experiment=name,macro_rmse_mm=metrics['macro_rmse_mm']));del model
    write_json(ROOT/'background_comparison.json',baselines)
    publication=[]
    for folder in sorted(ROOT.glob('E130*/s[0-9]*')):
        if (folder/'example_field_mm.npy').exists():
            field=torch.as_tensor(np.load(folder/'example_field_mm.npy'),device='cuda');field=(field-data.mean[0])/data.std[0]
            pred=exp.query(field,exp.val_features.geo).cpu().numpy()*data.std[0]+data.mean[0];saved=load_pack(folder/'expanded/evaluation.npz')['samples'][0,6*24+12]
            error=float(np.max(abs(pred-saved)));assert error<.001;publication.append(dict(experiment=folder.parent.name,seed=int(folder.name[1:]),max_publication_query_error_mm=error))
    write_json(ROOT/'publication_verification.json',publication)
    hybrid_ex=HybridExperiment() if (ROOT/'E1304_point_unet/status.json').exists() else None
    for folder in sorted(ROOT.glob('E130*/s[0-9]*')):
        fit=json.loads((folder/'fit.json').read_text());kind=fit['kind'];model=(ResidualMLP() if kind=='mlp' else Hybrid() if kind=='hybrid' else GridUNet(fit['width'])).cuda();evaluator=hybrid_ex if kind=='hybrid' else exp;panel=[exp.fit[i] for i in np.random.default_rng(13001).choice(len(exp.fit),48,False)]
        scores={}
        for checkpoint in ('best','last'):
            model.load_state_dict(torch.load(folder/(checkpoint+'.pt'),weights_only=True));model.eval();sq=0.;count=0
            with torch.no_grad():
                for e in panel:
                    y=(exp.raw['train'][e['row'],12+e['hour'],:,0]-data.mean[0])/data.std[0];mask=torch.as_tensor(e['target'],device='cuda')&torch.isfinite(y);prediction=evaluator.predict(model,e,kind);sq+=float((prediction[mask]-y[mask]).square().sum());count+=int(mask.sum())
            scores[checkpoint+'_fixed_train_rmse_mm']=(sq/count)**.5*float(data.std[0])
        training.append(dict(experiment=folder.parent.name,seed=fit['seed'],panel_events=len(panel),**scores));del model
    write_json(ROOT/'fixed_training_diagnosis.json',training)
    print(json.dumps(records,indent=2))

if __name__=='__main__':run()
