#!/usr/bin/env python3
"""Real ERA5 pressure-profile diagnostic, never a primary-model input.

Uses a four-date sample to inspect integration, TCWV closure and station-height
sensitivity. No fitted correction, model selection or teacher-quality claim is
made. The inherited station-height datum and GNSS matching times remain explicit
limitations. Default below-ground extension is limited to 200 m and flagged.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tropogan.physics import G0, integrate_pressure_column, specific_humidity_from_dewpoint


def load_gtx(path):
    with Path(path).open("rb") as stream:
        lat0, lon0, dlat, dlon, nlat, nlon = struct.unpack(">4d2i", stream.read(40))
        values = np.frombuffer(stream.read(), dtype=">f4").reshape(nlat, nlon).astype(float)
    lat_axis = lat0 + dlat * np.arange(nlat)
    lon_axis = lon0 + dlon * np.arange(nlon)
    def separation(lat, lon):
        return sum(wy * wx * values[iy, ix] for iy, wy in brackets(lat_axis, lat)
                   for ix, wx in brackets(lon_axis, lon))
    return separation, {"path": str(path), "sha256": digest(path),
                        "lat0": lat0, "lon0": lon0, "dlat": dlat, "dlon": dlon,
                        "shape": [nlat, nlon], "undulation_units": "m",
                        "source": "https://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx"}


def ecef_height_wgs84(xyz):
    x, y, z = xyz
    a, f = 6378137., 1 / 298.257223563
    e2 = f * (2 - f)
    radius = np.hypot(x, y)
    phi = np.arctan2(z, radius * (1 - e2))
    for _ in range(12):
        n = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        height = radius / np.cos(phi) - n
        phi = np.arctan2(z, radius * (1 - e2 * n / (n + height)))
    return float(height)


def audit_station_heights(station, archive):
    """Confirm height datum numerically against the same weekly TRO ECEF block.

The ambiguous GEOID_HGT column is not interpreted as undulation. Instead APP_HGT
and inherited H5 height must agree with a WGS84 ECEF-derived ellipsoidal height.
This audits this sample's station epochs only, not every historical H5 row.
"""
    candidates = []
    for path in archive.rglob("*.gz"):
        match = re.search(r"_(\d{4})(\d{3})0000_", path.name)
        if match:
            start = pd.Timestamp(int(match[1]), 1, 1) + pd.Timedelta(days=int(match[2]) - 1)
            candidates.append((start, path))
    cache, records = {}, []
    for row in station.itertuples(index=False):
        time = pd.Timestamp(row.datetime)
        matching = [(start, path) for start, path in candidates if start <= time < start + pd.Timedelta(days=7)]
        rec = {"station": row.station, "datetime": str(time), "h5_height_m": row.height_m,
               "ellipsoidal_height_confirmed": False}
        if not matching:
            rec["failure"] = "No corresponding weekly raw TRO file"
            records.append(rec)
            continue
        _, path = max(matching, key=lambda item: item[0])
        if path not in cache:
            section, site, coordinates, headers = "", {}, {}, []
            with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
                for line in stream:
                    if line.startswith("+"):
                        section = line.strip()
                    elif line.startswith("-"):
                        section = ""
                    elif line.startswith("*") and section in ["+SITE/ID", "+SITE/COORDINATES"]:
                        headers.append(line.strip())
                    elif line.startswith(" ") and line.strip():
                        fields = line.split()
                        if section == "+SITE/ID":
                            site[fields[0][:4]] = float(fields[-2])
                        elif section == "+SITE/COORDINATES":
                            coordinates[fields[0][:4]] = list(map(float, fields[6:9]))
            cache[path] = {"site": site, "coordinates": coordinates, "headers": headers,
                           "sha256": digest(path)}
        source = cache[path]
        rec.update({"source_path": str(path), "source_sha256": source["sha256"]})
        if row.station in source["site"] and row.station in source["coordinates"]:
            approx = source["site"][row.station]
            derived = ecef_height_wgs84(source["coordinates"][row.station])
            rec.update({"site_id_app_hgt_m": approx, "ecef_wgs84_ellipsoidal_height_m": derived,
                        "app_hgt_minus_ecef_m": approx - derived,
                        "h5_minus_app_hgt_m": row.height_m - approx,
                        "ellipsoidal_height_confirmed": bool(abs(approx - derived) < 2 and abs(row.height_m - approx) < .01)})
        else:
            rec["failure"] = "Station missing in SITE/ID or SITE/COORDINATES"
        records.append(rec)
    lookup = {(rec["station"], pd.Timestamp(rec["datetime"])): rec["ellipsoidal_height_confirmed"] for rec in records}
    return lookup, {"records": records, "confirmed_rows": sum(rec["ellipsoidal_height_confirmed"] for rec in records),
                    "raw_file_headers": {str(path): value["headers"] for path, value in cache.items()},
                    "criterion": "abs(APP_HGT - WGS84_ECEF_height)<2m and abs(H5_height-APP_HGT)<0.01m",
                    "scope": "Only the matched four-date, twelve-station sample is audited."}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scores(pred, reference):
    pred, reference = np.asarray(pred, float), np.asarray(reference, float)
    good = np.isfinite(pred) & np.isfinite(reference)
    if not good.any():
        return {"n": 0, "invalid": int(len(pred))}
    error = pred[good] - reference[good]
    return {"n": int(good.sum()), "invalid": int((~good).sum()),
            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()),
            "p95_absolute": float(np.quantile(np.abs(error), .95)),
            "reference_mean": float(reference[good].mean())}


def write_grid(path, times, lat, lon, fields, units, attributes):
    import h5netcdf
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5netcdf.File(path, "w") as target:
        target.dimensions = {"valid_time": len(times), "latitude": len(lat), "longitude": len(lon)}
        tv = target.create_variable("valid_time", ("valid_time",), dtype="i8")
        tv[:] = np.asarray(times).astype("datetime64[s]").astype("int64")
        tv.attrs.update({"units": "seconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"})
        for name, values, unit in [("latitude", lat, "degrees_north"), ("longitude", lon, "degrees_east")]:
            v = target.create_variable(name, (name,), dtype="f8")
            v[:] = values
            v.attrs["units"] = unit
        for name, values in fields.items():
            v = target.create_variable(name, ("valid_time", "latitude", "longitude"), dtype="f8")
            v[:] = values
            v.attrs["units"] = units.get(name, "1")
            v.attrs["data_role"] = "diagnostic_only_not_primary_model_input"
        target.attrs.update(attributes)


def brackets(axis, value):
    order = np.argsort(axis)
    ascending = np.asarray(axis)[order]
    if value < ascending[0] or value > ascending[-1]:
        raise ValueError("Station lies outside profile grid")
    j = min(max(int(np.searchsorted(ascending, value, side="right")), 1), len(axis) - 1)
    fraction = (value - ascending[j - 1]) / (ascending[j] - ascending[j - 1])
    return [(int(order[j - 1]), 1 - fraction), (int(order[j]), fraction)]


def main():
    import xarray as xr
    parser = argparse.ArgumentParser(description=__doc__)
    stem = "2023_0101_0401_0701_1001_6h_45_50N_5_10E.nc"
    parser.add_argument("--pressure-levels", type=Path, default=ROOT / "data/external" / ("era5_pl_" + stem))
    parser.add_argument("--surface", type=Path, default=ROOT / "data/external" / ("era5_sfc_" + stem))
    parser.add_argument("--station-data", type=Path, default=ROOT / "data/processed/station_pilot.parquet")
    parser.add_argument("--max-below-surface-m", type=float, default=200)
    parser.add_argument("--output-stem", default="era5_profile_pilot_2023_4days")
    parser.add_argument("--geoid-grid", type=Path, default=ROOT / "data/external/egm96_15.gtx")
    parser.add_argument("--tro-archive", type=Path, default=ROOT.parent.parent /
        "联合信号分析与深度学习的高精度对流层建模/Data/model/Model4/_project_archive/2026-08-20_pre_patent_reorganization/gnss_source_and_cache/ted_data/euref_troposphere_products")
    args = parser.parse_args()
    # h5netcdf supports Unicode Windows paths; netCDF4's local native build does
    # not reliably open these paths. Avoid lazy arrays in this tiny diagnostic.
    with xr.open_dataset(args.pressure_levels, engine="h5netcdf") as pl, \
         xr.open_dataset(args.surface, engine="h5netcdf") as surface:
        times, lat, lon = pl.valid_time.values, pl.latitude.values, pl.longitude.values
        for name, values in [("valid_time", times), ("latitude", lat), ("longitude", lon)]:
            if not np.array_equal(surface[name].values, values):
                raise ValueError("Pressure/surface coordinates must match exactly")
        p = pl.pressure_level.values.astype(float) * 100
        t, q, z = (pl[name].values.astype(float) for name in ["t", "q", "z"])
        z /= G0
        sp, t2m, d2m, tcwv = (surface[name].values.astype(float)
                              for name in ["sp", "t2m", "d2m", "tcwv"])
        ground = surface.surface_geopotential.values.astype(float) / G0
        q2m = specific_humidity_from_dewpoint(d2m, sp)
        source_units = {"pressure_levels": {name: pl[name].attrs.get("units") for name in ["t", "q", "z", "pressure_level"]},
                        "surface": {name: surface[name].attrs.get("units") for name in surface.data_vars}}
    shape = sp.shape
    fields = {name: np.full(shape, np.nan) for name in
              ["zwd_m", "pwv_kg_m2", "top_zwd_m", "top_pwv_kg_m2", "excluded_underground_levels"]}
    errors = []

    def column(it, iy, ix, lower=None):
        return integrate_pressure_column(p, t[it, :, iy, ix], q[it, :, iy, ix], z[it, :, iy, ix],
            surface_pressure_pa=sp[it, iy, ix], surface_height_m=ground[iy, ix],
            surface_temperature_k=t2m[it, iy, ix], surface_q=q2m[it, iy, ix],
            lower_height_m=lower, max_below_surface_m=args.max_below_surface_m,
            include_top_residual=True)

    for it in range(len(times)):
        for iy in range(len(lat)):
            for ix in range(len(lon)):
                try:
                    result = column(it, iy, ix)
                    fields["zwd_m"][it, iy, ix] = result.zwd_m
                    fields["pwv_kg_m2"][it, iy, ix] = result.pwv_kg_m2
                    fields["top_zwd_m"][it, iy, ix] = result.zwd_top_m
                    fields["top_pwv_kg_m2"][it, iy, ix] = result.pwv_top_kg_m2
                    fields["excluded_underground_levels"][it, iy, ix] = result.underground_levels_excluded
                except ValueError as exc:
                    errors.append({"time": str(times[it]), "iy": iy, "ix": ix, "error": str(exc)})
    fields["native_tcwv_kg_m2"] = tcwv
    fields["pwv_minus_tcwv_kg_m2"] = fields["pwv_kg_m2"] - tcwv
    out_grid = ROOT / "data/processed" / (args.output_stem + ".nc")
    attrs = {
        "data_role": "diagnostic_only_not_primary_model_input_or_residual_base",
        "scientific_scope": "four-date pressure-level integration diagnostic; not GAN training or teacher validation",
        "height_definition": "surface and pressure-level geopotential divided by g0; station EGM96 scheme separately audits source ellipsoidal heights",
        "integration": "hydrostatic pressure trapezoids with T2m/q2m surface endpoint; underground levels excluded",
        "top_remainder": "constant top-level q/T from 1 hPa to zero pressure, separately reported",
    }
    write_grid(out_grid, times, lat, lon, fields,
               {"zwd_m": "m", "pwv_kg_m2": "kg m-2", "top_zwd_m": "m",
                "top_pwv_kg_m2": "kg m-2", "native_tcwv_kg_m2": "kg m-2",
                "pwv_minus_tcwv_kg_m2": "kg m-2"}, attrs)
    station = pd.read_parquet(args.station_data)
    station = station[station.datetime.isin(pd.to_datetime(times)) &
                      station.lat.between(float(lat.min()), float(lat.max())) &
                      station.lon.between(float(lon.min()), float(lon.max()))].copy()
    height_confirmed, height_audit = audit_station_heights(station, args.tro_archive)
    geoid, geoid_metadata = load_gtx(args.geoid_grid) if args.geoid_grid.exists() else (None, None)
    time_lookup = {pd.Timestamp(value): i for i, value in enumerate(times)}
    comparisons = []
    for row in station.itertuples(index=False):
        it = time_lookup[pd.Timestamp(row.datetime)]
        weights = [(iy, ix, wy * wx) for iy, wy in brackets(lat, row.lat)
                   for ix, wx in brackets(lon, row.lon) if wy * wx > 1e-12]
        native_zwd = sum(w * fields["zwd_m"][it, iy, ix] for iy, ix, w in weights)
        mean_ground = sum(w * ground[iy, ix] for iy, ix, w in weights)
        # +/-30 m are sensitivity scenarios, NOT an assumed geoid correction.
        schemes = [("legacy_numeric", 0.0), ("sensitivity_minus30m", -30.0), ("sensitivity_plus30m", 30.0)]
        confirmed = height_confirmed[(row.station, pd.Timestamp(row.datetime))]
        if geoid is not None:
            undulation = geoid(row.lat, row.lon)
            schemes.append(("egm96_source_confirmed" if confirmed else "egm96_unconfirmed_sensitivity", -undulation))
        for scheme, offset in schemes:
            record = {"station": row.station, "datetime": row.datetime,
                      "lat": row.lat, "lon": row.lon, "height_m": row.height_m,
                      "height_sensitivity_offset_m": offset,
                      "height_scheme": scheme,
                      "inherited_minus_model_height_m": row.height_m - mean_ground,
                      "zwd_reference_m": row.zwd_m, "ztd_gnss_m": row.ztd_m,
                      "zwd_native_surface_interpolation_m": native_zwd,
                      "zwd_at_station_m": np.nan, "ztd_profile_pressure_m": np.nan,
                      "pressure_profile_minus_legacy_hpa": np.nan,
                      "max_below_surface_extrapolation_m": np.nan,
                      "valid": False, "failure": "",
                      "height_datum_status": ("source_ellipsoidal_height_confirmed_then_EGM96_h_minus_N_approx_geopotential_height"
                                              if scheme == "egm96_source_confirmed" else
                                              "inherited_numeric_height_or_sensitivity_only"),
                      "data_role": "diagnostic_only_not_primary_model_input"}
            try:
                pieces = [(w, column(it, iy, ix, row.height_m + offset)) for iy, ix, w in weights]
                value = sum(w * result.zwd_m for w, result in pieces)
                pressure = sum(w * result.bottom_pressure_pa for w, result in pieces) / 100
                zhd = .0022768 * pressure / (1 - .00266 * np.cos(2 * np.deg2rad(row.lat)) -
                                             .00028 * (row.height_m + offset) / 1000)
                record.update({"zwd_at_station_m": value, "ztd_profile_pressure_m": value + zhd,
                               "pressure_profile_minus_legacy_hpa": pressure - row.pressure_station_hpa,
                               "max_below_surface_extrapolation_m": max(r.extrapolated_below_surface_m for _, r in pieces),
                               "valid": True})
            except ValueError as exc:
                record["failure"] = str(exc)
            comparisons.append(record)
    comparison = pd.DataFrame(comparisons)
    out_station = ROOT / "data/processed" / (args.output_stem + "_station_comparison.csv")
    comparison.to_csv(out_station, index=False)
    common_keys = comparison.groupby(["station", "datetime"]).valid.all()
    common_keys = common_keys[common_keys].index
    station_stats = {}
    for scheme, group in comparison.groupby("height_scheme"):
        valid = group.valid
        common = group.set_index(["station", "datetime"]).index.isin(common_keys)
        station_stats[scheme] = {
            "zwd_mm": scores(group.zwd_at_station_m.to_numpy() * 1000, group.zwd_reference_m.to_numpy() * 1000),
            "ztd_mm_using_profile_adjusted_pressure": scores(group.ztd_profile_pressure_m.to_numpy() * 1000, group.ztd_gnss_m.to_numpy() * 1000),
            "valid_stations": int(group.loc[valid, "station"].nunique()),
            "common_sample_zwd_mm": scores(group.loc[common, "zwd_at_station_m"].to_numpy() * 1000,
                                            group.loc[common, "zwd_reference_m"].to_numpy() * 1000),
            "common_sample_ztd_mm_using_profile_adjusted_pressure": scores(
                group.loc[common, "ztd_profile_pressure_m"].to_numpy() * 1000,
                group.loc[common, "ztd_gnss_m"].to_numpy() * 1000),
            "valid_rows_using_any_below_surface_extrapolation": int((valid & group.max_below_surface_extrapolation_m.gt(0)).sum()),
            "zwd_no_below_surface_extrapolation_mm": scores(
                group.loc[valid & group.max_below_surface_extrapolation_m.eq(0), "zwd_at_station_m"].to_numpy() * 1000,
                group.loc[valid & group.max_below_surface_extrapolation_m.eq(0), "zwd_reference_m"].to_numpy() * 1000),
        }
    base = comparison[comparison.height_scheme.eq("legacy_numeric")]
    by_time = {str(time): scores(fields["pwv_kg_m2"][i].ravel(), tcwv[i].ravel())
               for i, time in enumerate(times)}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "physics_module_sha256": digest(ROOT / "src/tropogan/physics.py"),
        "sources": [{"path": str(path), "sha256": digest(path)} for path in
                    [args.pressure_levels, args.surface, args.station_data]],
        "source_units": source_units, "shape": {"time": len(times), "pressure_levels": len(p),
            "latitude": len(lat), "longitude": len(lon)},
        "data_role": attrs["data_role"],
        "pressure_levels_hpa": (p / 100).tolist(),
        "tcwv_closure_kg_m2": scores(fields["pwv_kg_m2"].ravel(), tcwv.ravel()),
        "tcwv_closure_by_time_kg_m2": by_time,
        "grid_valid": int(np.isfinite(fields["zwd_m"]).sum()), "grid_failed": len(errors),
        "grid_errors": errors[:20],
        "grid_zwd_range_mm": [float(np.nanmin(fields["zwd_m"]) * 1000), float(np.nanmax(fields["zwd_m"]) * 1000)],
        "maximum_top_residual_zwd_mm": float(np.nanmax(fields["top_zwd_m"]) * 1000),
        "maximum_top_residual_pwv_kg_m2": float(np.nanmax(fields["top_pwv_kg_m2"])),
        "max_below_surface_extrapolation_permitted_m": args.max_below_surface_m,
        "matched_station_rows": len(station), "matched_station_count": int(station.station.nunique()),
        "native_surface_station_zwd_mm_not_height_matched": scores(base.zwd_native_surface_interpolation_m.to_numpy() * 1000,
                                                                   base.zwd_reference_m.to_numpy() * 1000),
        "native_surface_station_zwd_mm_on_adjusted_valid_sample": scores(
            base.loc[base.valid, "zwd_native_surface_interpolation_m"].to_numpy() * 1000,
            base.loc[base.valid, "zwd_reference_m"].to_numpy() * 1000),
        "station_height_sensitivity": station_stats,
        "raw_tro_height_audit": height_audit,
        "geoid_grid": geoid_metadata,
        "recommended_height_scheme_for_this_diagnostic": "egm96_source_confirmed" if geoid is not None and all(height_confirmed.values()) else "sensitivity_only",
        "known_limitations": [
            "This is four individual dates, not four complete months; no general teacher reliability or skill conclusion follows.",
            "TCWV closure is an internal consistency diagnostic using shared ERA5 inputs, not independent validation.",
            "ERA5 pressure-level interpolation omits native-model vertical detail; surface T2m/q2m approximate the endpoint.",
            "Station comparisons use inherited H5 six-hour matched epochs, not original GNSS half-hour epochs.",
            "Only source-confirmed EGM96 scenarios convert ellipsoidal height h to approximate orthometric height h-N; orthometric/geopotential and local-gravity differences remain a smaller approximation. Other offsets are sensitivity tests.",
            "Legacy GNSS-minus-ERA5-ZHD target values are deliberately unchanged; their original pressure correction mixed station ellipsoidal height and ERA5 geopotential height. Corrected physical ZTD and legacy numerical ZWD-reference scores must not be conflated.",
            "Below-surface extrapolation holds T/q constant and is restricted and reported; invalid rows are retained in the comparison CSV.",
            "TCWV/PWV/IWV and profile-derived ZWD fields are prohibited primary-model features and residual bases by the user.",
            "No CERRA atmospheric information participates in ERA5 integration or its station-height corrections.",
        ],
        "outputs": [{"path": str(path), "sha256": digest(path)} for path in [out_grid, out_station]],
    }
    report_path = ROOT / "reports" / (args.output_stem + "_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"grid_valid": report["grid_valid"], "grid_failed": len(errors),
                      "tcwv_closure_kg_m2": report["tcwv_closure_kg_m2"],
                      "station_count": report["matched_station_count"],
                      "station_height_sensitivity": station_stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
