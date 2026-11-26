"""Synthetic identifiability illustration; no GNSS data or trained model used."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    ny, nx, n = 48, 64, 128
    x = rng.uniform(1, nx - 2, n)
    y = rng.uniform(1, ny - 2, n)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    h = np.zeros((n, ny * nx), dtype=np.float64)
    for dx, dy, w in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                       (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
        h[np.arange(n), (iy+dy)*nx+ix+dx] = w
    yy, xx = np.mgrid[:ny, :nx]
    base = 100 + 15*np.sin(2*np.pi*xx/nx) + 8*np.cos(2*np.pi*yy/ny)
    smooth = (np.sin(2*np.pi*(4*xx/nx + 3*yy/ny))
              + .7*np.cos(2*np.pi*(7*xx/nx - 4*yy/ny)))
    v = smooth.ravel()
    correction = h.T @ np.linalg.lstsq(h @ h.T, h @ v, rcond=1e-12)[0]
    delta = (v - correction).reshape(ny, nx)
    delta *= 3.0 / np.sqrt(np.mean(delta**2))
    alternative = base + delta
    observed_delta = h @ delta.ravel()
    max_diff = float(np.max(np.abs(observed_delta)))
    if max_diff > 1e-9:
        raise RuntimeError('Nullspace numerical identity failed')

    def radial_psd(a):
        f = np.fft.rfft2(a-a.mean())
        power = np.abs(f)**2 / a.size**2
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.rfftfreq(nx)[None, :]
        radius = np.sqrt(ky**2+kx**2)
        bins = np.linspace(0, .5, 25)
        indices = np.digitize(radius.ravel(), bins)-1
        means = [power.ravel()[indices == j].mean() if np.any(indices == j)
                 else np.nan for j in range(len(bins)-1)]
        return (bins[:-1]+bins[1:])/2, np.asarray(means)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    vmin, vmax = min(base.min(), alternative.min()), max(base.max(), alternative.max())
    for ax, a, title in [(axes[0, 0], base, 'Synthetic field A'),
                          (axes[0, 1], alternative, 'Synthetic field B: same station values')]:
        im = ax.imshow(a, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.scatter(x, y, s=9, facecolors='none', edgecolors='white', linewidths=.6)
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[0]), label='Synthetic Z0 / mm', shrink=.8)
    im = axes[1, 0].imshow(delta, origin='lower', cmap='RdBu_r', vmin=-8, vmax=8)
    axes[1, 0].scatter(x, y, s=9, facecolors='none', edgecolors='black', linewidths=.6)
    axes[1, 0].set_title('Difference: grid RMS = 3 mm')
    fig.colorbar(im, ax=axes[1, 0], label='Difference / mm', shrink=.8)
    for a, name in [(base, 'Field A'), (alternative, 'Field B')]:
        k, p = radial_psd(a)
        axes[1, 1].semilogy(k, np.maximum(p, 1e-12), label=name)
    axes[1, 1].set(xlabel='Spatial frequency / cycles per pixel',
                   ylabel='Mean spectral power', title='Different spectra; identical measured values')
    axes[1, 1].legend()
    fig.suptitle('Synthetic illustration only: fixed sparse observations do not identify all grid modes')
    fig.savefig(root/'nullspace_demo.png', dpi=150)
    plt.close(fig)
    result = dict(kind='synthetic_linear_operator_illustration_not_model_validation',
                  seed=42, grid_shape=[ny, nx], synthetic_stations=n,
                  max_station_prediction_difference_mm=max_diff,
                  grid_difference_rms_mm=float(np.sqrt(np.mean(delta**2))),
                  mean_station_difference_mm=float(observed_delta.mean()),
                  interpretation='Neither field is designated atmospheric truth; station equality does not determine full-grid spectrum.')
    (root/'nullspace_demo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
