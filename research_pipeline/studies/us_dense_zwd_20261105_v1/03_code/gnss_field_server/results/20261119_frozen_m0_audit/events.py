from reproduce import *
from gnss_field.training.baselines import get_baseline_predictor,_idw
from gnss_field.geometry import query_grid
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,io,zipfile
hours=np.array(store.hours); yr=pd.to_datetime(hours,unit='h').year
aids=store.ids('A');allxy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()]
def change(ts):
 v=store.context_values(ts); z=np.where(np.isfinite(v).all(-1),v[:,:,0],np.nan)
 return np.nanmedian(abs(z[3:]-z[:-3]),axis=1)
trainh=hours[yr==2018]; devh=hours[yr==2019]
threshold=np.nanquantile(change(trainh),.95);i=np.flatnonzero(change(devh)>threshold)[0]+3;t=devh[i:i+1]
ctx=b.build_context(t);mu=field(t)[0,0];valid=~ctx.padding[0].numpy()
values=store.context_values(t)[0,:,0];g=store.query_valid[aids]&valid
delta=values[g]-query_grid(mu,store.qxy[aids[g]]).numpy()
iw=_idw(store.xy[aids[g]],delta,allxy,0).reshape(store.shape)
idw=_idw(store.xy[aids[valid]],values[valid].astype(float),allxy,store.stats['input_mean'][0]).reshape(store.shape)
fields=[ctx.baseline[0,0].numpy(),idw,mu.numpy(),mu.numpy()+iw];names=['100 km background','Grid IDW','M0','M0 + innovation IDW']
region=store.grid['region'].astype(bool); xg=store.grid['x']/1000;yg=store.grid['y']/1000
extent=[xg.min()-12.5,xg.max()+12.5,yg.min()-12.5,yg.max()+12.5]
lo,hi=np.percentile(np.array(fields)[:,region],[2,98]);fig,axs=plt.subplots(1,4,figsize=(13,5),layout='constrained')
for ax,name,f in zip(axs,names,fields):
 im=ax.imshow(np.where(region,f,np.nan),origin='lower',extent=extent,vmin=lo,vmax=hi,cmap='viridis')
 ax.contour(xg,yg,ctx.maps[0,3].numpy(),levels=[.5],linewidths=.45,colors='white');ax.set_title(name);ax.set_xlabel('Lambert x (km)')
axs[0].set_ylabel('Lambert y (km)');fig.colorbar(im,ax=axs,label='Common-height Z0 (mm)',shrink=.65)
fig.suptitle(str(pd.to_datetime(t[0],unit='h'))+' | first 2019 exceedance of 2018 A-site median 3h-change P95');fig.savefig(root/'baseline_event.png',dpi=150);plt.close(fig)
# Use identical fully supported square, increasing size only while every cell is supported.
from scipy.ndimage import minimum_filter
support=ctx.maps[0,3].numpy().astype(bool)&region
for size in range(min(40,*store.shape),7,-2):
 pts=np.argwhere(minimum_filter(support.astype('u1'),size=size,mode='constant')>0)
 if len(pts):break
if len(pts):
 cy,cx=pts[len(pts)//2];sl=np.s_[cy-size//2:cy-size//2+size,cx-size//2:cx-size//2+size]
 assert support[sl].all()
 win=np.outer(np.hanning(size),np.hanning(size));freq=np.fft.fftfreq(size,25);rad=np.hypot(*np.meshgrid(freq,freq));yy,xx=np.indices((size,size));design=np.stack([np.ones_like(xx),xx,yy],axis=-1).reshape(-1,3)
 fig,ax=plt.subplots()
 for name,f in zip(names,fields):
  z=f[sl];z=z-(design@np.linalg.lstsq(design,z.ravel(),rcond=None)[0]).reshape(z.shape);power=abs(np.fft.fft2(z*win))**2/(size**2*(win**2).sum())
  bounds=[50,100,200,400,np.inf];band=[power[(rad>0)&(rad<=1/l)&(rad>1/r)].sum() for l,r in zip(bounds,bounds[1:])]
  ax.plot(range(4),band,'o-',label=name)
 ax.set_xticks(range(4),['50-100','100-200','200-400','>400']);ax.set(xlabel='Wavelength band (km)',ylabel='Band power (mm squared)',title=f'Event morphology | common {size} x {size} stencil');ax.legend();fig.savefig(root/'event_spectrum.png',dpi=150);plt.close(fig)
s=Path('dist/a800_v03/results_inventory_20261119.log').read_text(encoding='utf-8-sig')
z=zipfile.ZipFile(io.BytesIO(base64.b64decode(next(x.split('=',1)[1] for x in s.splitlines() if x.startswith('RESULT_ZIP_BASE64=')))))
def logs(key):return [json.loads(x) for x in z.read(next(n for n in z.namelist() if n.endswith(key))).splitlines()]
tr=logs('training.jsonl');de=logs('development.jsonl');fig,ax=plt.subplots(1,2,figsize=(9,4));ax[0].plot([r['step'] for r in tr],[r['loss'] for r in tr]);ax[0].set(xlabel='Update',ylabel='Training loss');ax[1].plot([r['step'] for r in de],[r['metrics']['point']['value'] for r in de],'o-');ax[1].set(xlabel='Update',ylabel='2019 subset station-macro RMSE (mm)');fig.tight_layout();fig.savefig(root/'m0_training.png',dpi=150);plt.close(fig)
print('event',t,'threshold',threshold,'spectrum size',size,flush=True)
