"""Hydrostatic water-column operators, independent of xarray and ML frameworks.

Pressure is Pa, temperature K, specific humidity kg/kg and heights metres on ONE
consistent vertical datum. k2'=22.1 K/hPa, k3=373900 K^2/hPa. We use pressure
quadrature under hydrostatic balance, constant g and a moist ideal gas:
 PWV = integral(q dp)/g
 ZWD = 1e-6 * Rv/g * integral(q * (k2'_Pa + k3_Pa/T) dp).
PWV in kg/m^2 is numerically equal to liquid water depth in mm.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G0 = 9.80665
RD = 287.05
RV = 461.5
EPSILON = RD / RV
K2_PA = 0.221
K3_PA = 3739.0


@dataclass(frozen=True)
class ColumnIntegral:
    zwd_m: float
    pwv_kg_m2: float
    zwd_top_m: float = 0.0
    pwv_top_kg_m2: float = 0.0
    bottom_pressure_pa: float = float("nan")
    underground_levels_excluded: int = 0
    extrapolated_below_surface_m: float = 0.0


def vapor_pressure_pa(q, pressure_pa):
    q, p = np.broadcast_arrays(np.asarray(q, float), np.asarray(pressure_pa, float))
    return q * p / (EPSILON + (1 - EPSILON) * q)


def specific_humidity_from_vapor_pressure(e_pa, pressure_pa):
    e, p = np.broadcast_arrays(np.asarray(e_pa, float), np.asarray(pressure_pa, float))
    if np.any((e < 0) | (e >= p)):
        raise ValueError("Vapour pressure must lie between zero and total pressure")
    return EPSILON * e / (p - (1 - EPSILON) * e)


def specific_humidity_from_dewpoint(dewpoint_k, pressure_pa):
    """Bolton/Magnus water-phase fit, same convention as legacy preprocessing."""
    td_c = np.asarray(dewpoint_k, float) - 273.15
    e_pa = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    return specific_humidity_from_vapor_pressure(e_pa, pressure_pa)


def wet_refractivity(pressure, temperature_k, q, pressure_unit="Pa"):
    p = np.asarray(pressure, float)
    if pressure_unit == "hPa":
        p = 100 * p
    elif pressure_unit != "Pa":
        raise ValueError("pressure_unit must be Pa or hPa")
    t = np.asarray(temperature_k, float)
    e = vapor_pressure_pa(q, p)
    return K2_PA * e / t + K3_PA * e / t**2


def virtual_temperature(temperature_k, q):
    return np.asarray(temperature_k, float) * (1 + (1 / EPSILON - 1) * np.asarray(q, float))


def _validate_thermodynamics(p, t, q):
    if not (np.isfinite(p).all() and np.isfinite(t).all() and np.isfinite(q).all()):
        raise ValueError("Nonfinite pressure/temperature/humidity")
    if np.any(p <= 0) or np.any(t <= 0) or np.any((q < 0) | (q >= 1)):
        raise ValueError("Invalid physical pressure, temperature or specific humidity")


def integrate_pressure_column(pressure_pa, temperature_k, q, height_m, *,
                              surface_pressure_pa, surface_height_m,
                              surface_temperature_k, surface_q,
                              lower_height_m=None, max_below_surface_m=0.0,
                              include_top_residual=True, gravity=G0):
    """Integrate sampled pressure levels, with an explicit surface endpoint.

Below-ground pressure levels are excluded before their thermodynamics are used.
Surface T2m/q2m may supply the endpoint; this is a near-surface approximation.
For a lower boundary above ground, pressure is interpolated log-linearly in
height and T/q linearly in height. Below-ground extension is disabled by default;
when explicitly permitted it holds surface T/q constant hydrostatically.
The optional top remainder holds the top-level T/q constant from p_top to zero;
its separately returned value exposes this small, unobserved contribution.
"""
    p, t, hum, z = [np.asarray(a, float).ravel() for a in
                    (pressure_pa, temperature_k, q, height_m)]
    if not (p.shape == t.shape == hum.shape == z.shape):
        raise ValueError("Profile arrays must have the same shape")
    if not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError("Missing pressure/height must not be treated as underground levels")
    ps, zs, ts, qs = map(float, (surface_pressure_pa, surface_height_m,
                               surface_temperature_k, surface_q))
    _validate_thermodynamics(np.array([ps]), np.array([ts]), np.array([qs]))
    if not np.isfinite(zs) or not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("Invalid surface height or gravity")
    # Both pressure and geometric criteria are required: interpolated pressure
    # levels below actual terrain cannot be used as measured atmospheric states.
    above = (p < ps) & (z > zs)
    excluded = int((~above).sum())
    if above.sum() < 2:
        raise ValueError("Fewer than two atmospheric pressure levels above surface")
    p, t, hum, z = p[above], t[above], hum[above], z[above]
    _validate_thermodynamics(p, t, hum)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite atmospheric height")
    order = np.argsort(z)
    p, t, hum, z = p[order], t[order], hum[order], z[order]
    if np.any(np.diff(z) <= 0) or np.any(np.diff(p) >= 0):
        raise ValueError("Height must increase and pressure decrease monotonically")
    p = np.r_[ps, p]
    t = np.r_[ts, t]
    hum = np.r_[qs, hum]
    z = np.r_[zs, z]
    lower = zs if lower_height_m is None else float(lower_height_m)
    if not np.isfinite(lower) or lower >= z[-1]:
        raise ValueError("Lower boundary must be finite and below highest sampled level")
    below = max(zs - lower, 0.0)
    if below > max_below_surface_m + 1e-9:
        raise ValueError("Lower boundary exceeds permitted below-surface extrapolation")
    if below:
        pb = ps * np.exp(gravity * below / (RD * virtual_temperature(ts, qs)))
        p, t, hum, z = (np.r_[pb, p], np.r_[ts, t], np.r_[qs, hum], np.r_[lower, z])
    elif lower > zs:
        j = int(np.searchsorted(z, lower, side="right"))
        fraction = (lower - z[j - 1]) / (z[j] - z[j - 1])
        pb = np.exp((1 - fraction) * np.log(p[j - 1]) + fraction * np.log(p[j]))
        tb = (1 - fraction) * t[j - 1] + fraction * t[j]
        qb = (1 - fraction) * hum[j - 1] + fraction * hum[j]
        p, t, hum = np.r_[pb, p[j:]], np.r_[tb, t[j:]], np.r_[qb, hum[j:]]
    dp = -np.diff(p)
    integrand = hum * (K2_PA + K3_PA / t)
    pwv = np.sum(0.5 * (hum[:-1] + hum[1:]) * dp) / gravity
    zwd = 1e-6 * RV * np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dp) / gravity
    top_pwv = hum[-1] * p[-1] / gravity if include_top_residual else 0.0
    top_zwd = 1e-6 * RV * integrand[-1] * p[-1] / gravity if include_top_residual else 0.0
    return ColumnIntegral(float(zwd + top_zwd), float(pwv + top_pwv),
                          float(top_zwd), float(top_pwv), float(p[0]), excluded, below)


def hybrid_half_level_pressures(a_pa, b, surface_pressure_pa):
    """p_half = a + b*ps; a must already be in Pa (read product metadata)."""
    a, b = np.asarray(a_pa, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Hybrid coefficients must be equal-length 1-D arrays")
    ps = np.asarray(surface_pressure_pa, float)
    return a.reshape((-1,) + (1,) * ps.ndim) + b.reshape((-1,) + (1,) * ps.ndim) * ps


def integrate_hybrid_column(half_pressure_pa, temperature_k, q, *,
                            surface_height_m, lower_height_m=None,
                            max_below_surface_m=0.0, include_top_residual=True,
                            gravity=G0):
    """Finite-volume mixed-layer integral, compatible with ERA5/CERRA adapters.

Coefficients and T/q are supplied in matching top-to-bottom or bottom-to-top
order. T/q are held constant within each native layer. Hypsometric layer bounds
permit a height lower limit without importing another model's profile. A zero
pressure top is allowed: its finite wet mass is integrated, while its formal
height is infinite. No guessed full-level pressure coefficients are required.
"""
    edges, t, hum = [np.asarray(a, float).ravel() for a in (half_pressure_pa, temperature_k, q)]
    if len(edges) != len(t) + 1 or t.shape != hum.shape:
        raise ValueError("N native layers require N+1 pressure interfaces")
    if np.all(np.diff(edges) < 0):
        edges, t, hum = edges[::-1], t[::-1], hum[::-1]
    if not np.isfinite(edges).all() or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Interface pressure must be monotonic and nonnegative")
    _validate_thermodynamics(edges[1:], t, hum)
    zs = float(surface_height_m)
    lower = zs if lower_height_m is None else float(lower_height_m)
    if not np.isfinite([zs, lower, gravity]).all() or gravity <= 0:
        raise ValueError("Invalid height/gravity")
    below = max(zs - lower, 0.0)
    if below > max_below_surface_m + 1e-9:
        raise ValueError("Lower boundary exceeds permitted below-surface extrapolation")
    z_bottom = zs
    effective_dp = np.zeros(len(t))
    bottom_pressure = float("nan")
    for j in range(len(t) - 1, -1, -1):
        p_top, p_bottom = edges[j], edges[j + 1]
        scale = RD * float(virtual_temperature(t[j], hum[j])) / gravity
        z_top = np.inf if p_top == 0 else z_bottom + scale * np.log(p_bottom / p_top)
        if lower <= z_bottom:
            effective_dp[j] = p_bottom - p_top
            if np.isnan(bottom_pressure):
                bottom_pressure = p_bottom
        elif lower < z_top:
            p_cut = p_bottom * np.exp(-(lower - z_bottom) / scale)
            effective_dp[j] = p_cut - p_top
            if np.isnan(bottom_pressure):
                bottom_pressure = p_cut
        z_bottom = z_top
    if below:
        scale = RD * float(virtual_temperature(t[-1], hum[-1])) / gravity
        p_cut = edges[-1] * np.exp(below / scale)
        effective_dp[-1] += p_cut - edges[-1]
        bottom_pressure = p_cut
    if not np.isfinite(bottom_pressure):
        raise ValueError("Lower boundary is above model top")
    integrand = hum * (K2_PA + K3_PA / t)
    pwv = float(np.sum(hum * effective_dp) / gravity)
    zwd = float(1e-6 * RV * np.sum(integrand * effective_dp) / gravity)
    top_pwv = float(hum[0] * edges[0] / gravity) if include_top_residual else 0.0
    top_zwd = float(1e-6 * RV * integrand[0] * edges[0] / gravity) if include_top_residual else 0.0
    return ColumnIntegral(zwd + top_zwd, pwv + top_pwv, top_zwd, top_pwv,
                          float(bottom_pressure), 0, below)
