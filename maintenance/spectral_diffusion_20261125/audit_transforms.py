"""Local transform audit on existing allowed development residuals; no training."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENGINE = ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
FIG = ROOT/'docs/figures/spectral_diffusion_20261125'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(x):
    x = x-x.mean(axis=-1,keepdims=True)
    f = np.fft.rfftfreq(x.shape[-1],d=1.)
    power = abs(np.fft.rfft(x,axis=-1,norm='ortho'))**2
    weight = np.full(len(f),2.)
    weight[0] = 1
    if x.shape[-1]%2 == 0: weight[-1] = 1
    return f,power*weight


def basis(xy, order):
    z = 2*np.pi*(xy-xy.min(0))/(xy.max(0)-xy.min(0))
    modes = [(i,j) for i in range(-order,order+1) for j in range(-order,order+1)
             if i>0 or (i==0 and j>0)]
    columns = [np.ones(len(z))]
    for i,j in modes:
        phase = i*z[:,0]+j*z[:,1]
        columns.extend([np.sqrt(2)*np.cos(phase),np.sqrt(2)*np.sin(phase)])
    return np.stack(columns,axis=1)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    evaluation = ENGINE/'results/20261125_p14_continuous_fusion/p14_s42/evaluation.npz'
    geometry = ENGINE/'results/20261125_temporal_continuous_data/geometry.npz'
    with np.load(evaluation) as z:
        prediction = np.array(z['prediction'],dtype=np.float64)
        truth = np.array(z['truth'],dtype=np.float64)
    assert prediction.shape == (3,12,72,137) and truth.shape == (12,72,137)
    residual = truth-prediction[0]
    with np.load(geometry) as z:
        xy = np.array(z['xy'],dtype=np.float64)
    assert xy.shape == (410,2)
    monthly = []
    pooled = np.zeros(37)
    pooled_increment = np.zeros(36)
    count = 0
    fft_error = 0.
    parseval_error = 0.
    for month in range(12):
        r = residual[month].T
        r = r[np.isfinite(r).all(axis=1)]
        assert len(r)>0
        f,p = spectrum(r)
        fd,pd = spectrum(np.diff(r,axis=-1))
        monthly.append(dict(month=month+1,complete_station_windows=len(r),
            residual_low_frequency_energy_fraction=float(p[:,f<=1/12].sum()/p.sum()),
            increment_low_frequency_energy_fraction=float(pd[:,fd<=1/12].sum()/pd.sum())))
        pooled += p.sum(0); count += len(r)
        pooled_increment += pd.sum(0)
        hat = np.fft.fft(r,axis=-1,norm='ortho')
        fft_error = max(fft_error,float(abs(np.fft.ifft(hat,axis=-1,norm='ortho').real-r).max()))
        parseval_error = max(parseval_error,float(abs((abs(hat)**2).sum(-1)-(r*r).sum(-1)).max()))
    rng = np.random.default_rng(42)
    layouts = [np.sort(rng.choice(410,82,replace=False)) for _ in range(3)]
    irregular = []
    for order in (1,2,3,4,5):
        full = basis(xy,order)
        for li,ids in enumerate(layouts):
            a = full[ids]/np.sqrt(len(ids))
            sv = np.linalg.svd(a,compute_uv=False)
            rank = int((sv>sv[0]*1e-10).sum())
            gram = a.T@a
            irregular.append(dict(order=order,layout=li,observations=len(ids),
                coefficients=a.shape[1],numerical_rank=rank,nullity=a.shape[1]-rank,
                condition_number=float(sv[0]/sv[-1]) if rank==a.shape[1] else None,
                gram_identity_relative_error=float(np.linalg.norm(gram-np.eye(a.shape[1]))/np.sqrt(a.shape[1]))))
    d = cdist(xy,xy)
    neighbours = np.argsort(d,axis=1)[:,1:9]
    w = np.zeros_like(d)
    for i,ids in enumerate(neighbours): w[i,ids] = np.exp(-.5*(d[i,ids]/150)**2)
    w = np.maximum(w,w.T)
    inv_degree = 1/np.sqrt(w.sum(1))
    lap = np.eye(len(xy))-inv_degree[:,None]*w*inv_degree[None]
    lam,u = eigh(lap)
    sample = rng.normal(size=(len(xy),5))
    recovered = u@(u.T@sample)
    q_error = float(np.max(abs(u.T@u-np.eye(len(xy)))))
    rng2 = np.random.default_rng(24)
    x0 = rng2.normal(size=len(xy)); noise = rng2.normal(size=len(xy))
    alpha_bar = .4
    xk = np.sqrt(alpha_bar)*x0+np.sqrt(1-alpha_bar)*noise
    spectral = np.sqrt(alpha_bar)*(u.T@x0)+np.sqrt(1-alpha_bar)*(u.T@noise)
    equiv_error = float(np.max(abs(u.T@xk-spectral)))
    assert fft_error<1e-10 and q_error<1e-10 and equiv_error<1e-10
    report = dict(status='LOCAL_TRANSFORM_AUDIT_NOT_MODEL_TRAINING',
        generator_sha256=sha(Path(__file__)),
        data_roles='Existing development137 residuals only; training410 static coordinates; no confirmation/C/D labels',
        source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (evaluation,geometry)},
        temporal=dict(layout=0,hours_per_window=72,months=12,complete_station_windows=count,
            preprocessing='remove each station-window mean; no interpolation or zero fill; fully observed windows only',
            low_band='period >=12 h; DC removed',
            pooled_low_frequency_energy_fraction=float(pooled[f<=1/12].sum()/pooled.sum()),
            pooled_increment_low_frequency_energy_fraction=float(pooled_increment[fd<=1/12].sum()/pooled_increment.sum()),
            fft_inverse_max_error_mm=fft_error,parseval_max_absolute_energy_error=parseval_error,monthly=monthly),
        irregular_fourier=dict(scope='Actual training410 coordinates; 3 fixed random82 subsets; direct Fourier matrices, not a FINUFFT runtime benchmark',rows=irregular),
        graph=dict(scope='Fixed training410 geometry graph for algebra only; not the proposed anchor graph and no graph residual spectrum measured',
            neighbours=8,radius_km=150,normalized_laplacian=True,
            eigenvalue_min=float(lam.min()),eigenvalue_max=float(lam.max()),
            orthogonality_max_error=q_error,full_inverse_max_error=float(abs(recovered-sample).max()),
            transformed_diffusion_equivalence_max_error=equiv_error),
        new_network_trained=False)
    (HERE/'transform_audit.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    fig,axes = plt.subplots(1,3,figsize=(14,4.3))
    axes[0].plot(np.arange(1,13),[r['residual_low_frequency_energy_fraction'] for r in monthly],marker='o',label='Residual')
    axes[0].plot(np.arange(1,13),[r['increment_low_frequency_energy_fraction'] for r in monthly],marker='s',label='Hourly increment')
    axes[0].set(xlabel='2020 month',ylabel='Energy fraction, period >=12 h',ylim=(0,1),title='Complete 72 h station windows')
    axes[0].legend(fontsize=8)
    axes[1].plot(f[1:],pooled[1:]/pooled[1:].sum(),color='#416b91')
    axes[1].set(xlabel='Temporal frequency (cycles/hour)',ylabel='Fraction of centered residual energy',title='Pooled residual spectrum')
    axes[2].plot([r['coefficients'] for r in irregular if r['layout']==0],
                 [r['numerical_rank'] for r in irregular if r['layout']==0],marker='o',label='Numerical rank')
    axes[2].plot([9,25,49,81,121],[9,25,49,81,121],ls='--',color='#999999',label='Full column rank')
    axes[2].axhline(82,color='#ad7041',ls=':',label='82 observed stations')
    axes[2].set(xlabel='Real Fourier coefficients',ylabel='Rank',title='Nonuniform observation operator')
    axes[2].legend(fontsize=8)
    fig.suptitle('p14 residual transform audit — diagnostic only; no new network training',fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG/'transform_audit.png',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status=report['status'],station_windows=count,
        pooled_low_band_fraction=report['temporal']['pooled_low_frequency_energy_fraction'],
        pooled_increment_low_band_fraction=report['temporal']['pooled_increment_low_frequency_energy_fraction'],
        monthly_increment_low_band_mean=float(np.mean([r['increment_low_frequency_energy_fraction'] for r in monthly])),
        fft_inverse_error=fft_error,graph_orthogonality_error=q_error,
        large_basis_nullities=[r['nullity'] for r in irregular if r['coefficients']==121]),ensure_ascii=False))


if __name__=='__main__':
    main()
