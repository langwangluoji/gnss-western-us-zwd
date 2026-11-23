"""Synthetic counterexamples for metric interpretation; no research observations or training."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.ndimage import gaussian_filter,uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
rng=np.random.default_rng(20261117);n=96
raw=gaussian_filter(rng.normal(size=(n,n)),5,mode='wrap')+.15*gaussian_filter(rng.normal(size=(n,n)),1.5,mode='wrap')
x=100+20*(raw-raw.mean())/raw.std();fx=np.fft.fft2(x-x.mean(),norm='ortho')
z=np.fft.fft2(rng.normal(size=x.shape),norm='ortho');fr=np.abs(fx)*z/np.maximum(abs(z),1e-30)
phase=x.mean()+np.fft.ifft2(fr,norm='ortho').real
noise=rng.normal(size=x.shape);noise-=gaussian_filter(noise,2,mode='wrap');noise*=5/noise.std()
fields={'Reference':x,'Bias +10 mm':x+10,'Blur':gaussian_filter(x,2,mode='wrap'),'Shift 4 cells':np.roll(x,4,axis=1),'Added high frequency':x+noise,'Random phase / same PSD':phase}
def ssim(a,b):
 ua=uniform_filter(a,7,mode='wrap');ub=uniform_filter(b,7,mode='wrap')
 va=uniform_filter(a*a,7,mode='wrap')-ua*ua;vb=uniform_filter(b*b,7,mode='wrap')-ub*ub;cov=uniform_filter(a*b,7,mode='wrap')-ua*ub
 c1=(.01*100)**2;c2=(.03*100)**2
 return float(np.mean((2*ua*ub+c1)*(2*cov+c2)/((ua*ua+ub*ub+c1)*(va+vb+c2))))
rows=[];spectra={};k=np.fft.fftfreq(n)*n;ky,kx=np.meshgrid(k,k,indexing='ij');radius=np.floor(np.hypot(kx,ky)).astype(int)
for name,y in fields.items():
 fy=np.fft.fft2(y-y.mean(),norm='ortho');p=abs(fy)**2;px=abs(fx)**2
 err=y-x;grad=(np.roll(err,-1,0)-err)**2+(np.roll(err,-1,1)-err)**2
 row=dict(case=name,rmse_mm=float(np.sqrt(np.mean(err**2))),bias_mm=float(err.mean()),ssim_fixed_range_100=ssim(x,y),relative_demeaned_psd_l2=float(np.linalg.norm(p-px)/np.linalg.norm(px)),relative_complex_spectrum_l2=float(np.linalg.norm(fy-fx)/np.linalg.norm(fx)),gradient_error_mm_per_cell=float(np.sqrt(grad.mean())))
 rows.append(row);spectra[name]=np.bincount(radius.ravel(),weights=p.ravel())/np.maximum(1,np.bincount(radius.ravel()))
 np.testing.assert_allclose(np.mean(abs(np.fft.fft2(err,norm='ortho'))**2),np.mean(err**2),rtol=1e-12)
assert rows[-1]['relative_demeaned_psd_l2']<1e-12 and rows[-1]['rmse_mm']>10
assert rows[1]['relative_demeaned_psd_l2']<1e-12
with (OUT/'synthetic_metric_counterexamples.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(OUT/'synthetic_metric_counterexamples.json').write_text(json.dumps(dict(seed=20261117,synthetic_only=True,periodic_boundary=True,ssim_note='Uniform 7x7 population covariance, fixed data range100, periodic boundary; not a package default or project score.',parseval_passed=True,rows=rows),indent=2),encoding='utf-8')
fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
for ax,(name,y) in zip(axes.flat,fields.items()):
 im=ax.imshow(y,vmin=40,vmax=160,cmap='viridis');ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axes.ravel().tolist(),label='Synthetic Z0 (mm)',shrink=.75);fig.suptitle('Metric counterexamples: synthetic fields, not GNSS results')
fig.savefig(OUT/'synthetic_fields.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
for name,p in spectra.items():ax.loglog(np.arange(1,40),p[1:40],label=name,ls='--' if name=='Random phase / same PSD' else '-')
ax.set(xlabel='Spatial wavenumber (cycles per domain)',ylabel='Radial mean power (mm2)',title='Bias, translation and random phase preserve demeaned power')
ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(OUT/'synthetic_spectra.png',dpi=180)
print(json.dumps(rows,indent=2))
