"""Locked broader 2019 development replay, no coefficient or model selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,time,json
import numpy as np, torch
import spike as sp
import simple_weather_unet_pilot as pilot
from gnss_field.geometry.operators import coarsen_100km, safe_coarsening_mask


@torch.no_grad()
def publication(data,store,raw,model,wide,selection,rows,out,reference):
    xy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()];aids=store.ids('A')
    k=sp.kernels(store.xy[aids],store.stations.height.to_numpy()[aids],xy,store.grid['height'].ravel())
    t=selection['candidates'][selection['selected']]; beta=np.zeros(41,'f4');beta[t['columns']]=t['beta']
    beta=torch.tensor(beta,device='cuda');ids=np.arange(len(data.a['heights']));actual=[];smooth=[];original_smooth=[];error=0.
    first=True
    for ix in rows:
        x,base=data.batch(np.array([ix]),True);h,d=sp.features(model,x,wide);field=(base+10*d)[:,0]
        pa=(field.flatten(1)[:,data.qi[aids]]*data.qw[aids]).sum(-1)
        valid=torch.tensor(np.isfinite(raw[ix:ix+1]).all(-1)&store.query_valid[aids][None],device='cuda')
        obs=torch.tensor(np.nan_to_num(raw[ix:ix+1,:,0]),device='cuda')
        g=sp.moments(obs-pa,valid,k)/10
        f=torch.cat([h.flatten(2).transpose(1,2),torch.ones_like(g[:,:,:1]),g],-1)
        corrected=field+10*(f@beta).reshape_as(field)
        p=data.query(corrected,ids)
        error=max(error,float(np.max(abs(p.cpu().numpy()[0]-reference[ix]))))
        actual.append(p.cpu().numpy()[0]);smooth.append(data.query(coarsen_100km(corrected),ids).cpu().numpy()[0])
        original_smooth.append(data.query(coarsen_100km(field),ids).cpu().numpy()[0])
        if first:
            file=out/'published_example.npz'
            np.savez(file,hour=data.hours[ix],Z0=corrected.cpu().numpy(),original_Z0=field.cpu().numpy(),land=data.a['land'])
            restored=torch.tensor(np.load(file)['Z0'],device='cuda')
            assert float((data.query(restored,ids)-p).abs().max())<=.001
            first=False
    if error>.001:raise ValueError(f'Full-grid query mismatch: {error}')
    np.savez(out/'publication_predictions.npz',rows=rows,actual=actual,smooth=smooth,original_smooth=original_smooth)
    return error


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.monotonic()
    root=sp.ROOT/'confirmation_v02';root.mkdir(exist_ok=True)
    calendar=pilot.select_calendar()
    newhours=np.array([int(np.datetime64(f'2019-{m:02d}-08T00','h').astype('i8'))+offset
                      for m in range(1,13) for offset in range(0,72,6)])
    oldhours=calendar['development'].copy()
    assert np.min(abs(newhours[:,None]-oldhours[None]))>18
    calendar['development']=newhours
    src=sp.sources();results=sp.read_json(sp.ROOT/'results.json')
    champion=min(results,key=lambda n:results[n]['models'][results[n]['selected']]['inner']['macro_rmse'])
    oldchamp=min(results,key=lambda n:results[n]['models']['original']['inner']['macro_rmse'])
    sp.immutable_json(root/'confirmation.lock.json',dict(hours=newhours,code=sp.sha256(__file__),
        selections={k:sp.sha256(sp.ROOT/k/'selection.lock.json') for k in src},champion=champion,old_champion=oldchamp,
        interpretation='new to current U-Net evaluation; C2019 historically exposed development; not blind',
        D_read=False,no_refit=True,old_C48_hours=oldhours,minimum_separation_hours=18))
    pilot.ROOT=sp.ROOT/'confirmation/data';pilot.select_calendar=lambda:calendar
    pilot.prepare();data=pilot.PilotData()
    old=np.load(sp.PROJECT/'results/20261123_simple_weather_unet_v02/data.npz')
    assert np.array_equal(data.a['norm_mean'],old['norm_mean']) and np.array_equal(data.a['norm_std'],old['norm_std'])
    store=sp.PackedStore(pilot.DATA,purpose='context');raw=store.context_values(data.hours).astype('f4')
    cr=data.rows('development');ci=data.a['C_ids'];final={};preds={}
    for name,path in src.items():
        out=root/name;out.mkdir(exist_ok=True)
        wide=name.startswith('wide');model=(sp.ConditionedUNet() if wide else sp.SimpleUNet()).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model'])
        if (out/'features.npz').exists():
            a=np.load(out/'features.npz');base,design=a['prediction'],a['design']
        else:base,design,_=sp.extract(model,wide,data,store,raw,out)
        lock=sp.read_json(sp.ROOT/name/'selection.lock.json');t=lock['candidates'][lock['selected']]
        pred=base+10*(design[:,:,t['columns']]@np.asarray(t['beta']))
        final[name]=dict(original=sp.score(data,base,cr,ci),corrected=sp.score(data,pred,cr,ci),
                         paired_ci=sp.paired_interval(data,pred,base,cr,ci),selected=lock['selected'])
        final[name]['delta']=final[name]['corrected']['macro_rmse']-final[name]['original']['macro_rmse']
        preds[name]=pred[np.ix_(cr,ci)];preds[name+'_original']=base[np.ix_(cr,ci)]
        final[name]['publication_error_mm']=publication(data,store,raw,model,wide,lock,cr[::12],out,pred)
        print(json.dumps(dict(name=name,**final[name])),flush=True)
        del model;torch.cuda.empty_cache()
    # Same-key comparison to archived stronger geometry reference; keep both native and intersected masks.
    annual=sp.PROJECT/'results/20261122_geometry_base_retrain_v01/annual2019'
    manifest=sp.read_json(sp.PROJECT/'results/20261119_frozen_m0_audit/station_prediction_manifest.json')
    assert list(ci)==manifest['year2019_station_ids']
    archives=[dict(np.load(annual/f'2019-{m:02d}'/'scored.npz')) for m in range(1,13)]
    archive={k:np.concatenate([a[k] for a in archives]) for k in archives[0]}
    ix=np.searchsorted(archive['hours'],newhours);assert np.array_equal(archive['hours'][ix],newhours)
    y=data.a['labels'][np.ix_(cr,ci)];mask=data.a['mask'][np.ix_(cr,ci)]
    common=mask & archive['population'][ix]
    assert np.max(abs(y[common]-archive['truth'][ix][common]))<=.001
    matched={name:sp.metrics(p,y,common) for name,p in preds.items()}
    for name in ('coarse','idw','M0','old_base','new0','new1'):
        matched['historical_'+name]=sp.metrics(archive[name][ix],y,common)
    original_champ=preds[oldchamp+'_original'];chosen=preds[champion]
    result=dict(models=final,champion=champion,old_champion=oldchamp,matched_historical=matched,
                old_champion_score=sp.metrics(original_champ,y,mask),champion_score=sp.metrics(chosen,y,mask),
                hours=144,seconds=time.monotonic()-start,C_historically_exposed=True,D_used=False,server_jobs=False)
    np.savez(root/'predictions.npz',**preds,truth=y,mask=mask,common_mask=common,hours=newhours,station_ids=ci,
             **{f'historical_{n}':archive[n][ix] for n in ('coarse','idw','M0','old_base','new0','new1')})
    sp.write_json(root/'results.json',result);sp.write_json(root/'status.json',dict(stage='COMPLETE',seconds=time.monotonic()-start))
    print('CONFIRMATION_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
