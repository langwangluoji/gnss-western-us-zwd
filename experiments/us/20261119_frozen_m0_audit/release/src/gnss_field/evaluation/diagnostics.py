"""Fixed-scale common-mask image diagnostics, secondary to station evidence."""
import numpy as np
from scipy.ndimage import uniform_filter


def _images(truth, prediction, mask):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    mask = np.asarray(mask)
    if truth.ndim != 2 or prediction.shape != truth.shape or mask.shape != truth.shape or mask.dtype != bool:
        raise ValueError('Expected aligned 2D images and boolean common mask')
    if not np.isfinite(truth[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError('Images must be finite on the declared common mask')
    return truth, prediction, mask


def ssim_diagnostic(truth, prediction, mask, *, data_range):
    """Uniform 7x7 population-moment SSIM, fixed training-only physical range."""
    truth, prediction, mask = _images(truth, prediction, mask)
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError('Positive fixed training data_range is required')
    if min(truth.shape) < 7:
        raise ValueError('SSIM requires at least a 7x7 common window')
    a, b = np.where(mask, truth, 0.), np.where(mask, prediction, 0.)
    average = lambda value: uniform_filter(value, size=7, mode='constant', cval=0.)
    ma, mb = average(a), average(b)
    va, vb = np.maximum(average(a*a)-ma*ma, 0.), np.maximum(average(b*b)-mb*mb, 0.)
    covariance = average(a*b)-ma*mb
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    similarity = ((2*ma*mb+c1)*(2*covariance+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
    complete = average(mask.astype(float)) >= 1-1e-12
    return dict(ssim=float(similarity[complete].mean()) if complete.any() else None,
                n_windows=int(complete.sum()), window=7, weights='uniform',
                data_range=float(data_range), moment_convention='population (divisor49)')


def _detrend(image):
    y, x = np.indices(image.shape, dtype=float)
    design = np.stack((np.ones(image.size), x.ravel(), y.ravel()), axis=1)
    trend = design @ np.linalg.lstsq(design, image.ravel(), rcond=None)[0]
    return image-trend.reshape(image.shape)


def psd_diagnostic(truth, prediction, mask=None, *, spacing_km=25., radial_edges_km=None):
    """Plane-detrended Hann energy-corrected 2D spectrum and phase error.

    Supply one complete common rectangle per call. Ensemble members and means
    are separate calls and labels in the caller's report. Spectrum power is
    diagnostic and does not establish fine-scale truth fidelity.
    """
    truth = np.asarray(truth, float)
    mask = np.ones_like(truth, bool) if mask is None else mask
    truth, prediction, mask = _images(truth, prediction, mask)
    if not mask.any() or spacing_km <= 0:
        raise ValueError('PSD requires a nonempty common rectangle and positive grid spacing')
    iy, ix = np.where(mask)
    rectangle = np.s_[iy.min():iy.max()+1, ix.min():ix.max()+1]
    if not mask[rectangle].all():
        raise ValueError('PSD common mask must form a complete rectangle')
    a, b = truth[rectangle], prediction[rectangle]
    if min(a.shape) < 3:
        raise ValueError('PSD requires a common rectangle at least 3x3')
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    energy = np.mean(window**2)
    af, bf = np.fft.fft2(_detrend(a)*window), np.fft.fft2(_detrend(b)*window)
    denominator = a.size**2*energy
    pa, pb, pe = abs(af)**2/denominator, abs(bf)**2/denominator, abs(bf-af)**2/denominator
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0], d=spacing_km),
                         np.fft.fftfreq(a.shape[1], d=spacing_km), indexing='ij')
    frequency = np.hypot(fy, fx)
    wavelength = np.full_like(frequency, np.inf)
    np.divide(1., frequency, out=wavelength, where=frequency > 0)
    nonzero = frequency > 0
    bands = {}
    for low, high in [(50, 100), (100, 200), (200, 400), (400, np.inf)]:
        selection = (wavelength >= low) & (wavelength < high) & nonzero
        label = f'{low}-{high}' if np.isfinite(high) else '>400'
        bands[label] = dict(truth_power=float(pa[selection].sum()),
                            prediction_power=float(pb[selection].sum()),
                            squared_error_power=float(pe[selection].sum()), n_modes=int(selection.sum()))
    if radial_edges_km is None:
        radial_edges_km = np.linspace(0., np.sqrt(2)/(2*spacing_km), 33)
    edges = np.asarray(radial_edges_km, float)
    if edges.ndim != 1 or len(edges) < 2 or (np.diff(edges) <= 0).any():
        raise ValueError('Radial frequency edges must be strictly increasing')
    radial = []
    for low, high in zip(edges[:-1], edges[1:]):
        selection = (frequency >= low) & (frequency < high) & nonzero
        radial.append(dict(frequency_low_cycles_km=float(low), frequency_high_cycles_km=float(high),
                           truth_power=float(pa[selection].sum()), prediction_power=float(pb[selection].sum()),
                           squared_error_power=float(pe[selection].sum()), n_modes=int(selection.sum())))
    return dict(truth_power=float(pa.sum()), prediction_power=float(pb.sum()),
                squared_error_power=float(pe.sum()), phase_rmse=float(np.sqrt(np.mean((b-a)**2))),
                bands=bands, radial=radial, spacing_km=float(spacing_km), shape=list(a.shape),
                detrend='least_squares_plane', window='symmetric_Hann_2D',
                window_energy=float(energy), normalization='Parseval mean square / mean(window_squared)')
