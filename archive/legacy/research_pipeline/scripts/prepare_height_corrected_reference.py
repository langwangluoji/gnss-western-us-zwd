#!/usr/bin/env python3
"""Versioned GNSS-ZTD minus ERA5-ZHD reference with explicit height datums.

Never overwrites the legacy parquet or changes any split. A datum is accepted
only where a source TRO SITE/ID height agrees with a WGS84 ECEF-derived height
and with the inherited station metadata. EGM96 converts h to H=h-N. An explicit
normal-gravity approximation converts H to geopotential height for ERA5 pressure
adjustment. Near-surface vapour pressure supplies virtual temperature exactly as
in the upstream script; no column water or vertical humidity enters ML features.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from integrate_profile_pilot import load_gtx

G0, RD = 9.80665, 287.05
EARTH_RADIUS_M = 6371008.8
FEATURES = ["lat", "lon", "height_orthometric_m", "pressure_station_hpa",
            "pressure_era5_hpa", "temperature_c", "wvp_hpa", "orography_m",
            "height_diff_m", "doy_sin", "doy_cos", "hour_sin", "hour_cos"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ecef_geodetic(xyz):
    x, y, z = xyz
    a, flattening = 6378137., 1 / 298.257223563
    e2 = flattening * (2 - flattening)
    radius = np.hypot(x, y)
    phi = np.arctan2(z, radius * (1 - e2))
    for _ in range(12):
        n = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        h = radius / np.cos(phi) - n
        phi = np.arctan2(z, radius * (1 - e2 * n / (n + h)))
    return float(np.rad2deg(phi)), float(np.rad2deg(np.arctan2(y, x))), float(h)


def parse_tro_height_evidence(path):
    section, heights, coordinates, headers = "", {}, {}, []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            if line.startswith("+"):
                section = line.strip()
            elif line.startswith("-"):
                if section == "+SITE/COORDINATES":
                    break
                section = ""
            elif line.startswith("*") and section in ["+SITE/ID", "+SITE/COORDINATES"]:
                headers.append(line.strip())
            elif line.startswith(" ") and line.strip():
                fields = line.split()
                if section == "+SITE/ID":
                    heights[fields[0][:4]] = {"station_id": fields[0], "app_hgt_m": float(fields[-2])}
                elif section == "+SITE/COORDINATES":
                    coordinates[fields[0][:4]] = list(map(float, fields[6:9]))
    evidence = {}
    for station, values in heights.items():
        if station in coordinates:
            lat, lon, h = ecef_geodetic(coordinates[station])
            evidence[station] = {**values, "ecef_lat": lat, "ecef_lon": lon,
                                 "ecef_ellipsoidal_height_m": h,
                                 "app_minus_ecef_m": values["app_hgt_m"] - h,
                                 "ellipsoidal_column_confirmed": abs(values["app_hgt_m"] - h) < 2.0}
    return evidence, {"path": str(path), "sha256": digest(path), "headers": headers,
                       "stations_with_both_blocks": len(evidence)}


def collect_height_evidence(frame, archive):
    files = []
    for path in archive.rglob("*.gz"):
        match = re.search(r"_(\d{4})(\d{3})0000_", path.name)
        if match:
            start = pd.Timestamp(int(match[1]), 1, 1) + pd.Timedelta(days=int(match[2]) - 1)
            files.append((start, path))
    files.sort(key=lambda item: item[0])
    wanted = set(frame.station)
    evidence = {station: [] for station in wanted}
    manifests = []
    consumed = set()

    def consume(path, reason):
        source, manifest = parse_tro_height_evidence(path)
        manifest["reason"] = reason
        manifest["regional_stations_present"] = sorted(wanted & set(source))
        manifests.append(manifest)
        consumed.add(path)
        for station in wanted & set(source):
            evidence[station].append({**source[station], "source_path": str(path),
                                      "source_sha256": manifest["sha256"]})

    # Fixed primary audit weeks: do not choose dates by model accuracy.
    for marker in ["2023001", "2024007"]:
        matching = [path for _, path in files if "_" + marker + "0000_" in path.name]
        if matching:
            consume(matching[0], "primary_2023_or_2024_datum_audit")
    primary_covered = {station for station, values in evidence.items()
                       if any(value["ellipsoidal_column_confirmed"] for value in values)}
    # Earlier site closures need earlier coordinate evidence. Only inspect the
    # first archived week of each earlier year until all station codes are covered.
    for year in [2019, 2020, 2021, 2022]:
        if all(any(v["ellipsoidal_column_confirmed"] for v in values) for values in evidence.values()):
            break
        eligible = [path for start, path in files if start.year == year and path not in consumed]
        if eligible:
            consume(eligible[0], "supplement_for_stations_absent_from_primary_weeks")
    # Newly introduced stations may postdate the fixed 2024 audit week. Inspect
    # their first represented observation week, never infer the datum by proximity.
    uncovered = [station for station, values in evidence.items()
                 if not any(v["ellipsoidal_column_confirmed"] for v in values)]
    for station in uncovered:
        first_epoch = frame.loc[frame.station.eq(station), "datetime"].min()
        matching = [(start, path) for start, path in files
                    if start <= first_epoch < start + pd.Timedelta(days=7) and path not in consumed]
        if matching:
            consume(max(matching, key=lambda item: item[0])[1],
                    "first_observed_week_for_previously_uncovered_station_" + station)
    return evidence, {"source_files": manifests,
                      "primary_weeks_station_coverage": len(primary_covered),
                      "primary_weeks_uncovered_stations": sorted(wanted - primary_covered),
                      "source_column_evidence": evidence}


def normal_gravity(latitude_deg):
    sin2 = np.sin(np.deg2rad(latitude_deg))**2
    return 9.7803253359 * (1 + .00193185265241 * sin2) / np.sqrt(1 - .00669437999013 * sin2)


def orthometric_to_geopotential_approx(height_orthometric_m, latitude_deg):
    """Integrate normal surface gravity with a spherical inverse-square decrease.

This resolves the gross ellipsoid/geoid offset. It is not a measured geopotential
number: EGM96 resolution, local gravity anomalies and tide conventions remain.
"""
    height = np.asarray(height_orthometric_m, float)
    return normal_gravity(np.asarray(latitude_deg, float)) / G0 * EARTH_RADIUS_M * height / (EARTH_RADIUS_M + height)


def saastamoinen_zhd(pressure_hpa, lat, orthometric_height_m):
    return .0022768 * pressure_hpa / (1 - .00266 * np.cos(2 * np.deg2rad(lat)) -
                                    .00028 * orthometric_height_m / 1000)


def numeric_summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "minimum": float(x.min()), "maximum": float(x.max()),
            "p95_absolute": float(np.quantile(np.abs(x), .95))}


def error_metrics(prediction, target):
    error = np.asarray(prediction, float) - np.asarray(target, float)
    good = np.isfinite(error)
    if not good.any():
        return {"n": 0}
    error = error[good]
    return {"n": int(good.sum()), "rmse_mm": float(np.sqrt(np.mean(error**2))),
            "bias_mm": float(error.mean()), "mae_mm": float(np.abs(error).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/station_pilot.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/station_reference_v2.parquet")
    parser.add_argument("--geoid-grid", type=Path, default=ROOT / "data/external/egm96_15.gtx")
    parser.add_argument("--tro-archive", type=Path, default=ROOT.parent.parent /
        "联合信号分析与深度学习的高精度对流层建模/Data/model/Model4/_project_archive/2026-08-20_pre_patent_reorganization/gnss_source_and_cache/ted_data/euref_troposphere_products")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("A new versioned output path is required")
    source = pd.read_parquet(args.input)
    original_digest = digest(args.input)
    frame = source.copy()
    evidence, datum_audit = collect_height_evidence(frame, args.tro_archive)
    geoid, geoid_metadata = load_gtx(args.geoid_grid)
    metadata = frame[["station", "lat", "lon", "height_m"]].drop_duplicates().reset_index(drop=True)
    proof_rows = []
    for row in metadata.itertuples(index=False):
        proofs = evidence[row.station]
        possible = [proof for proof in proofs if proof["ellipsoidal_column_confirmed"] and
                    abs(proof["app_hgt_m"] - row.height_m) < 2 and
                    abs(proof["ecef_lat"] - row.lat) < .01 and abs(proof["ecef_lon"] - row.lon) < .01]
        chosen = min(possible, key=lambda proof: abs(proof["app_hgt_m"] - row.height_m)) if possible else None
        proof_rows.append({"station": row.station, "lat": row.lat, "lon": row.lon,
                           "height_m": row.height_m, "height_datum_confirmed": bool(chosen),
                           "height_evidence_source": chosen["source_path"] if chosen else "",
                           "height_source_difference_m": row.height_m - chosen["app_hgt_m"] if chosen else None,
                           "geoid_undulation_m": geoid(row.lat, row.lon),
                           "height_datum_status": "source_ecef_confirms_ellipsoidal_height" if chosen else "datum_provenance_unknown"})
    metadata = pd.DataFrame(proof_rows)
    frame = frame.drop(columns=["height_datum_status"], errors="ignore").merge(
        metadata, on=["station", "lat", "lon", "height_m"], how="left", validate="many_to_one", sort=False)
    for original, legacy in [("zwd_m", "zwd_legacy_m"), ("zhd_m", "zhd_legacy_m"),
                             ("pressure_station_hpa", "pressure_station_legacy_hpa"),
                             ("height_diff_m", "height_diff_legacy_m"),
                             ("qc_model_ready", "qc_model_ready_legacy")]:
        frame[legacy] = frame[original]
    frame["height_ellipsoidal_m"] = frame.height_m
    confirmed = frame.height_datum_confirmed.astype(bool)
    frame["height_orthometric_m"] = (frame.height_ellipsoidal_m - frame.geoid_undulation_m).where(confirmed)
    frame["height_geopotential_m"] = orthometric_to_geopotential_approx(frame.height_orthometric_m, frame.lat)
    frame["height_diff_m"] = frame.height_geopotential_m - frame.orography_m
    # Preserve the exact upstream T_v convention; this is near-surface humidity,
    # not a prohibited vertical specific-humidity feature.
    e_pa = frame.wvp_hpa.to_numpy(float) * 100
    sp_pa = frame.pressure_era5_hpa.to_numpy(float) * 100
    q_surface = .622 * e_pa / (sp_pa - .378 * e_pa)
    tv = (frame.temperature_c.to_numpy(float) + 273.15) * (1 + .61 * q_surface)
    legacy_p = sp_pa * np.exp(-G0 * frame.height_diff_legacy_m.to_numpy() / (RD * tv)) / 100
    legacy_zhd = saastamoinen_zhd(legacy_p, frame.lat.to_numpy(), frame.height_ellipsoidal_m.to_numpy())
    pressure_reconstruction = float(np.max(np.abs(legacy_p - frame.pressure_station_legacy_hpa)))
    zhd_reconstruction = float(np.max(np.abs(legacy_zhd - frame.zhd_legacy_m)))
    if pressure_reconstruction > 1e-8 or zhd_reconstruction > 1e-10:
        raise ValueError("Reconstruction does not match the original pressure/ZHD formula")
    frame["pressure_station_hpa"] = sp_pa * np.exp(-G0 * frame.height_diff_m.to_numpy() / (RD * tv)) / 100
    frame["zhd_m"] = saastamoinen_zhd(frame.pressure_station_hpa, frame.lat, frame.height_orthometric_m)
    frame["zwd_m"] = frame.ztd_m - frame.zhd_m
    frame["zwd_reference_change_mm"] = (frame.zwd_m - frame.zwd_legacy_m) * 1000
    frame["pressure_reference_change_hpa"] = frame.pressure_station_hpa - frame.pressure_station_legacy_hpa
    frame["reference_version"] = "v2_egm96_confirmed_ellipsoid_to_geopotential_pressure"
    frame["target_source"] = "gnss_ztd_minus_era5_zhd_geoid_datum_corrected_v2"
    frame["qc_model_ready"] = confirmed & np.isfinite(frame[FEATURES + ["zwd_m", "ztd_m", "zhd_m"]]).all(axis=1)
    frame["qc_zwd_negative"] = frame.zwd_m.lt(0)
    # Strong invariants: preserve order, GNSS observations and every split exactly.
    for name in ["station", "datetime", "group_id", "split", "space_split", "time_split", "lat", "lon", "ztd_m"]:
        if not np.array_equal(frame[name].to_numpy(), source[name].to_numpy()):
            raise AssertionError(f"Unintended change to {name}")
    if not np.array_equal(frame.zwd_legacy_m.to_numpy(), source.zwd_m.to_numpy()):
        raise AssertionError("Legacy reference changed")
    if not np.array_equal(frame.zhd_legacy_m.to_numpy(), source.zhd_m.to_numpy()):
        raise AssertionError("Legacy ZHD changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, engine="pyarrow", compression="zstd", index=False)
    if digest(args.input) != original_digest:
        raise AssertionError("The immutable legacy input was modified")
    by_station = frame.groupby("station").agg(
        rows=("station", "size"), ready_rows=("qc_model_ready", "sum"),
        confirmed_rows=("height_datum_confirmed", "sum"),
        geoid_undulation_m=("geoid_undulation_m", "median"),
        zwd_change_mean_mm=("zwd_reference_change_mm", "mean"),
        zwd_change_min_mm=("zwd_reference_change_mm", "min"),
        zwd_change_max_mm=("zwd_reference_change_mm", "max"),
        pressure_change_mean_hpa=("pressure_reference_change_hpa", "mean"),
    )
    by_station_path = ROOT / "reports/station_reference_v2_station_changes.csv"
    by_station.to_csv(by_station_path)
    physical_path = ROOT / "data/processed/era5_profile_pilot_2023_4days_station_comparison.csv"
    physical_changes = {}
    if physical_path.exists():
        diagnostic = pd.read_csv(physical_path, parse_dates=["datetime"])
        diagnostic = diagnostic[diagnostic.height_scheme.eq("egm96_source_confirmed") & diagnostic.valid]
        diagnostic = diagnostic.merge(frame[["station", "datetime", "zwd_m", "zwd_legacy_m"]],
                                      on=["station", "datetime"], how="inner", validate="one_to_one")
        physical_changes = {
            "scope": "Unfitted ERA5 profile diagnostic on exactly the same EGM96-adjusted sample, not a trained model or broad validation.",
            "zwd_error_against_legacy_reference": error_metrics(diagnostic.zwd_at_station_m * 1000, diagnostic.zwd_legacy_m * 1000),
            "zwd_error_against_v2_reference": error_metrics(diagnostic.zwd_at_station_m * 1000, diagnostic.zwd_m * 1000),
            "ztd_error_against_gnss_using_profile_pressure": error_metrics(diagnostic.ztd_profile_pressure_m * 1000, diagnostic.ztd_gnss_m * 1000),
            "source_path": str(physical_path), "source_sha256": digest(physical_path),
        }
    upstream = ROOT.parent.parent / "联合信号分析与深度学习的高精度对流层建模/Data/model/Model4/Model4_FFT_SH_Patent/data_preparation/build_model_ready_dataset.py"
    report = {
        "schema_version": "station-reference-v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__)), "script_sha256": digest(Path(__file__)),
        "geoid_loader_script_sha256": digest(ROOT / "scripts/integrate_profile_pilot.py"),
        "legacy_input": {"path": str(args.input), "sha256": original_digest, "untouched": True},
        "upstream_reference_builder": {"path": str(upstream), "sha256": digest(upstream)},
        "output": {"path": str(args.output), "sha256": digest(args.output), "rows": len(frame),
                   "stations": int(frame.station.nunique()), "groups": int(frame.group_id.nunique()),
                   "unique_station_coordinate_height_records": len(metadata), "columns": list(frame.columns)},
        "feature_policy_version": "v2_surface_only_no_column_or_vertical_water",
        "feature_columns": FEATURES, "target_column": "zwd_m",
        "feature_exclusions": ["height_m", "height_ellipsoidal_m", "tcwv_kg_m2", "ztd_m", "zhd_m", "zwd_m",
                               "legacy targets and pressures", "PWV/IWV/TCWV and any ZWD background/proxy", "vertical humidity profiles", "station identifiers or target history"],
        "diagnostic_only_columns": ["tcwv_kg_m2", "geoid_undulation_m", "zwd_reference_change_mm", "pressure_reference_change_hpa"],
        "datum_audit": datum_audit,
        "metadata_datum_evidence": json.loads(metadata.to_json(orient="records")),
        "confirmed_stations": int(frame.loc[confirmed, "station"].nunique()),
        "unknown_stations": sorted(frame.loc[~confirmed, "station"].unique().tolist()),
        "confirmed_rows": int(confirmed.sum()), "unknown_rows": int((~confirmed).sum()),
        "ready_rows": int(frame.qc_model_ready.sum()),
        "geoid_grid": geoid_metadata,
        "formulas": {
            "orthometric_height_m": "H=h_ellipsoidal-N_EGM96(latitude,longitude)",
            "normal_gravity": "gamma=9.7803253359*(1+0.00193185265241*sin(lat)^2)/sqrt(1-0.00669437999013*sin(lat)^2)",
            "geopotential_height_approx_m": "Hphi=(gamma/9.80665)*6371008.8*H/(6371008.8+H)",
            "surface_specific_humidity": "q2=.622*e_pa/(sp_pa-.378*e_pa); e_pa=100*wvp_hpa; inherited upstream convention",
            "virtual_temperature_k": "Tv=(temperature_c+273.15)*(1+.61*q2); inherited upstream convention",
            "station_pressure_hpa": "p_station=(sp_pa/100)*exp[-9.80665*(Hphi-orography_m)/(287.05*Tv)]",
            "zhd_m": ".0022768*p_station/[1-.00266*cos(2*lat)-.00028*H/1000]",
            "zwd_m": "GNSS_ztd_m - zhd_m",
        },
        "legacy_formula_reconstruction": {"pressure_max_abs_difference_hpa": pressure_reconstruction,
                                          "zhd_max_abs_difference_m": zhd_reconstruction},
        "change_summary": {"zwd_change_mm": numeric_summary(frame.zwd_reference_change_mm),
                           "pressure_change_hpa": numeric_summary(frame.pressure_reference_change_hpa),
                           "geoid_undulation_m": numeric_summary(metadata.geoid_undulation_m),
                           "geopotential_minus_orthometric_m": numeric_summary(frame.height_geopotential_m - frame.height_orthometric_m)},
        "physical_diagnostic_error_changes": physical_changes,
        "negative_v2_zwd_rows_retained": int(frame.qc_zwd_negative.sum()),
        "split_summary": json.loads(frame.groupby("split").agg(rows=("station", "size"),
            ready_rows=("qc_model_ready", "sum"), stations=("station", "nunique")).to_json(orient="index")),
        "known_limitations": [
            "Datum evidence establishes the coordinate convention for sampled source weeks and matching station metadata, not a full daily monument/tide-system history.",
            "EGM96 15-minute resolution and normal-gravity conversion are approximations; local geoid/gravity errors remain.",
            "Pressure still comes from ERA5 and a constant near-surface virtual-temperature barometric adjustment; this remains a mixed reference, not pure GNSS wet delay.",
            "Original H5 temporal matching and unknown upstream assimilation independence remain unchanged.",
            "Negative derived ZWD values are retained and flagged; no target-based clipping or silent filtering is performed.",
            "The four-date profile comparison does not establish general accuracy and is not a new model performance result.",
        ],
    }
    audit_path = ROOT / "reports/station_reference_v2_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "stations": report["output"]["stations"],
                      "confirmed_stations": report["confirmed_stations"], "unknown_stations": report["unknown_stations"],
                      "ready_rows": report["ready_rows"], "change_summary": report["change_summary"],
                      "negative_zwd_rows_retained": report["negative_v2_zwd_rows_retained"],
                      "physical_diagnostic_error_changes": physical_changes}, indent=2), flush=True)


if __name__ == "__main__":
    main()
