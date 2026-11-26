from reproduce import root,store,ids,xy,h,b,model,device,m
import numpy as np,pandas as pd,torch
from gnss_field.data import ContextBuilder,PackedStore
from gnss_field.training.fields import predict_patches
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.geometry import surface_query,query_grid,input_geometry
from functools import lru_cache
from gnss_field.evaluation.metrics import point_metrics
aids=store.ids('A'); bids=store.ids('B'); bxy,bh=safe_query_geometry(store,bids)
train=PackedStore(store.root,purpose='train')
grid=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()]
@lru_cache(maxsize=64)
def kernel(key):
 sel=np.flatnonzero(np.frombuffer(key,dtype=bool))
 if len(sel)<2:return sel,None
 g=input_geometry(store.xy[aids[sel]],grid)
 return sel,(g['weights'],g['indices'])
out=root/'monthly';out.mkdir(exist_ok=True)
for year in [2018,2019]:
 hours=np.array(store.hours);hours=hours[pd.to_datetime(hours,unit='h').year==year]
 for month in range(1,13):
  label=f'{year}-{month:02}';dest=out/(label+'.npz')
  if dest.exists():continue
  ts=hours[pd.to_datetime(hours,unit='h').month==month]
  pred=[];support=[];innov=[];coarse=[];direct=[]
  for i in range(0,len(ts),4):
   t=ts[i:i+4];ctx=b.build_context(t)
   with torch.inference_mode():f=predict_patches(model,ctx.to(device),store,chunk=8)[0].cpu()
   if year==2018:
    pred.extend(surface_query(f,bxy,bh)[:,0].numpy())
    support.extend([b.support(u)[1][bids] for u in t]);continue
   pred.extend(surface_query(f,xy,h)[:,0].numpy())
   coarse.extend(surface_query(ctx.baseline.cpu(),xy,h)[:,0].numpy())
   vals=store.context_values(t)
   for j,u in enumerate(t):
    valid=np.isfinite(vals[j]).all(-1)
    fields=[]
    for correction in (False,True):
     sel,k=kernel((valid & (store.query_valid[aids] if correction else True)).tobytes())
     z=vals[j,sel,0].astype('f8')
     default=0 if correction else store.stats['input_mean'][0]
     if correction and len(sel):z-=query_grid(f[j,0],store.qxy[aids[sel]]).numpy()
     if k is None: field=np.full(store.shape,z[0] if len(z) else default)
     else:
      w,ii=k;raw=(w*z[ii]).sum(1);raw[w.sum(1)==0]=default;field=raw.reshape(store.shape)
     v=torch.tensor(field,dtype=torch.float32)
     if correction:v=v+f[j,0]
     fields.append(surface_query(v,xy,h).numpy())
    direct.append(fields[0]);innov.append(fields[1])
    support.append(b.support(u)[1][ids])
  y=(train.targets('B',ts) if year==2018 else store.targets('C',ts))
  mask=np.array(support)&np.isfinite(y); data=dict(hours=ts,truth=y,mask=mask,mean=np.array(pred))
  if year==2019:
   archived=np.load(m/f'full2019/5000/forecast_{label}.npz')
   np.testing.assert_array_equal(mask,archived['normal_support_S0']&archived['population_F'])
   np.testing.assert_allclose(data['mean'],archived['prediction'],rtol=0,atol=.001)
   data.update(idw=np.array(direct),innovation=np.array(innov),coarse=np.array(coarse))
  temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(dest)
  print('Completed',label,flush=True)
