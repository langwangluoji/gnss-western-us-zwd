"""Parse CAS-AIR GPAC-Repro PWV files using their actual solution columns."""
import calendar
import gzip
import hashlib
import io
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer

PIPELINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPELINE / "scripts"))
from parse_ngl_reference import sinex_epoch

EXPECTED_HEADER = ["SITE", "EPOCH", "TROTOT", "STDDEV", "PWV", "TROWET", "TRODRY", "ERAZTD", "ERAPWV", "TEMP", "PRESS", "TM"]
FIELDS = ["ztd_mm", "ztd_sigma_mm", "pwv_mm", "zwd_mm", "zhd_mm", "era5_ztd_mm", "era5_pwv_mm", "temperature_k", "pressure_hpa", "tm_k"]
ECEF = Transformer.from_crs(4978, 4979, always_xy=True)


def parse_header(lines):
    metadata = {}
    section = ""
    for raw in lines:
        line = raw.strip()
        parts = line.split()
        if not parts:
            continue
        if line.startswith("%=TRO"):
            metadata.update(station=parts[-1].upper(), product_header=line,
                            header_first_epoch=parts[5], header_last_epoch=parts[6])
            metadata["header_start"] = str(sinex_epoch(parts[5]))
            metadata["header_end"] = str(sinex_epoch(parts[6]))
        elif line.startswith("+"):
            section = line[1:]
        elif line.startswith("-"):
            section = ""
        elif line.startswith("*"):
            if section == "TROP/SOLUTION" and "TROTOT" in line:
                actual = [value.strip("_") for value in line[1:].split()]
                if actual != EXPECTED_HEADER:
                    raise ValueError("Unrecognized actual solution header: " + line)
                metadata["solution_header"] = actual
                break
        elif section == "FILE/REFERENCE" and parts[0] in {"INPUT", "SOFTWARE", "OUTPUT"}:
            metadata[parts[0].lower()] = " ".join(parts[1:])
        elif section == "TROP/DESCRIPTION":
            if line.startswith("SAMPLING TROP"):
                metadata["sampling_seconds"] = int(parts[-1])
            if line.startswith("TIME SYSTEM"):
                metadata["time_system_header"] = parts[-1]
        elif section == "SITE/ID":
            values = parts[-7:]
            lon = abs(float(values[0])) + float(values[1]) / 60 + float(values[2]) / 3600
            if values[0].startswith("-"):
                lon = -lon
            lat = abs(float(values[3])) + float(values[4]) / 60 + float(values[5]) / 3600
            if values[3].startswith("-"):
                lat = -lat
            height = np.nan if set(values[6]) == {"*"} else float(values[6])
            metadata.update(approx_lon=(lon + 180) % 360 - 180, approx_lat=lat, approx_height_m=height)
        elif section == "TROP/STA_COORDINATES":
            x, y, z = map(float, parts[4:7])
            lon, lat, height = ECEF.transform(x, y, z)
            metadata.update(x_ecef_m=x, y_ecef_m=y, z_ecef_m=z, lon=lon, lat=lat,
                            ellipsoidal_height_m=height, coordinate_frame=parts[7])
    for key in ("station", "lon", "lat", "solution_header", "sampling_seconds"):
        if key not in metadata:
            raise ValueError("Missing source metadata: " + key)
    if metadata["sampling_seconds"] != 3600:
        raise ValueError("Unexpected product sampling interval")
    metadata.setdefault("time_system_header", "UNSPECIFIED")
    metadata["approx_coordinate_warning"] = bool(abs(metadata["lat"] - metadata.get("approx_lat", metadata["lat"])) > .01
                                                or abs((metadata["lon"] - metadata.get("approx_lon", metadata["lon"]) + 180) % 360 - 180) > .01)
    return metadata


def read_header(path):
    prefix = []
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for _, line in zip(range(250), stream):
            prefix.append(line)
            if line.startswith("*") and "TROTOT" in line and "TROWET" in line:
                break
    metadata = parse_header(prefix)
    if metadata["station"] != Path(path).name.split(".")[0].upper():
        raise ValueError("Filename and header station disagree")
    metadata.update(source_path=str(Path(path).resolve()), source_bytes=Path(path).stat().st_size,
                    global_header_only=True, gzip_crc_verified=False)
    return metadata


def epoch_hours(tokens):
    parts = tokens.str.extract(r"^(\d{2}|\d{4}):(\d{3}):(\d{5})$")
    if parts.isna().any().any():
        raise ValueError("Malformed product epoch")
    years = parts[0].astype(int).to_numpy()
    years = np.where(years < 80, years + 2000, np.where(years < 100, years + 1900, years))
    days = parts[1].astype(int).to_numpy()
    seconds = parts[2].astype(int).to_numpy()
    last_days = np.array([366 if calendar.isleap(int(year)) else 365 for year in years])
    if np.any((days < 1) | (days > last_days) | (seconds < 0) | (seconds >= 86400)):
        raise ValueError("Invalid product day or seconds-of-day")
    year_bases = {year: int(np.datetime64(f"{year:04d}-01-01T00", "h").astype(np.int64)) for year in np.unique(years)}
    hours = np.array([year_bases[year] for year in years], dtype=np.int64) + (days - 1) * 24 + seconds // 3600
    return hours, seconds % 3600 == 0


def read_station(path):
    path = Path(path)
    before = path.stat()
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        text = stream.read()  # Read through EOF to validate gzip CRC and size.
    metadata = parse_header(text.splitlines()[:250])
    metadata["solution_block_terminated"] = "-TROP/SOLUTION" in text
    metadata["file_end_marker_present"] = bool(re.search(r"%=?ENDTRO\s*$", text))
    if metadata["station"] != path.name.split(".")[0].upper():
        raise ValueError("Filename and header station disagree")
    section = text.split("+TROP/SOLUTION", 1)[1].split("-TROP/SOLUTION", 1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        frame = pd.read_csv(io.StringIO(section), sep=r"\s+", comment="*", header=None,
                            names=["station", "sinex_epoch", *FIELDS], index_col=False,
                            dtype={"station": str, "sinex_epoch": str, **{name: np.float64 for name in FIELDS}},
                            na_values=["-nan", "NAN", "NaN"], on_bad_lines="error")
    if frame.empty or not frame.station.eq(metadata["station"]).all():
        raise ValueError("Unexpected station rows or empty solution")
    hours, on_hour = epoch_hours(frame.sinex_epoch)
    frame["hour_id"] = hours
    limits = {"ztd_mm": (800, 3500), "ztd_sigma_mm": (0, 100), "pwv_mm": (0, 120),
              "zwd_mm": (0, 700), "zhd_mm": (600, 3000), "temperature_k": (180, 340),
              "pressure_hpa": (400, 1100), "tm_k": (180, 340)}
    valid = on_hour.copy()
    for name, (low, high) in limits.items():
        valid &= np.isfinite(frame[name].to_numpy()) & frame[name].between(low, high).to_numpy()
    frame["joint_valid"] = valid
    duplicate = frame.hour_id.duplicated(keep=False)
    conflicts = frame.loc[duplicate].groupby("hour_id")[FIELDS].nunique(dropna=False).max(axis=1)
    conflict_hours = conflicts[conflicts > 1].index.to_numpy()
    frame.loc[frame.hour_id.isin(conflict_hours), "joint_valid"] = False
    metadata.update(raw_records=len(frame), duplicate_records=int(frame.hour_id.duplicated().sum()),
                    conflicting_duplicate_hours=len(conflict_hours), off_hour_records=int((~on_hour).sum()),
                    out_of_order_records=int((np.diff(hours) < 0).sum()),
                    nonfinite_numeric_records=int((~np.isfinite(frame[FIELDS].to_numpy()).all(axis=1)).sum()))
    frame = frame.drop_duplicates("hour_id", keep="first").sort_values("hour_id").reset_index(drop=True)
    frame["epoch_product"] = frame.hour_id.to_numpy().astype("datetime64[h]").astype("datetime64[ns]")
    frame["ztd_closure_residual_mm"] = frame.ztd_mm - frame.zwd_mm - frame.zhd_mm
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reading")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata.update(source_path=str(path.resolve()), source_bytes=after.st_size, source_sha256=digest,
                    gzip_crc_verified=True, global_header_only=False, records=len(frame), valid_records=int(frame.joint_valid.sum()),
                    actual_first_epoch=str(frame.epoch_product.iloc[0]), actual_last_epoch=str(frame.epoch_product.iloc[-1]),
                    header_bounds_match_actual=(str(frame.epoch_product.iloc[0]) == metadata["header_start"] and str(frame.epoch_product.iloc[-1]) == metadata["header_end"]),
                    closure_abs_max_mm=float(frame.ztd_closure_residual_mm.abs().max()),
                    closure_abs_gt_1mm_records=int(frame.ztd_closure_residual_mm.abs().gt(1).sum()),
                    zero_wet_records=int(frame.zwd_mm.eq(0).sum()))
    return frame, metadata


def window_metrics(hours, start, stop):
    hours = np.asarray(hours)
    left, right = np.searchsorted(hours, [start, stop])
    selected = hours[left:right]
    gaps = np.diff(np.r_[start - 1, selected, stop]) - 1
    expected = int(stop - start)
    return {"expected_hours": expected, "available_hours": len(selected), "missing_hours": expected - len(selected),
            "complete_fraction": len(selected) / expected, "max_missing_run_hours": int(gaps.max())}
