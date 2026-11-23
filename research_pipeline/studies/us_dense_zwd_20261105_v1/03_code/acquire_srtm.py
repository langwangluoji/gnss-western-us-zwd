"""Download official CGIAR-CSI SRTM v4.1 and verify the US support terrain."""
from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.merge import merge

STUDY = Path(__file__).resolve().parents[1]
RAW = STUDY / "02_data/raw/srtm"
OUTPUT = STUDY / "02_data/processed/terrain"
MANIFEST = STUDY / "01_inventory/srtm_manifest.json"
STATUS = STUDY / "07_operations/srtm_status.json"
BOUNDS = (-123.0, 31.0, -113.0, 39.0)  # west, south, east, north
BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
USGS_URL = "https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1-arc"
PRODUCT = "CGIAR-CSI / CIAT hole-filled SRTM version 4.1, 3 arc-seconds"
TILES = [(column, row) for column in (12, 13, 14) for row in (5, 6)]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def status(phase: str, **details) -> None:
    save(STATUS, {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **details})


def download(url: str, target: Path) -> None:
    if target.exists():
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Existing archive failed CRC: {target}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected download Content-Range")
                expected = int(response.headers.get("Content-Length", 0))
                received = 0
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                if expected and received != expected:
                    raise ValueError("Incomplete HTTP download")
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Downloaded archive failed CRC")
            partial.replace(target)
            return
        except (requests.RequestException, zipfile.BadZipFile, ValueError) as error:
            corrupt = isinstance(error, (zipfile.BadZipFile, ValueError))
            invalid_range = isinstance(error, requests.HTTPError) and error.response.status_code == 416
            if corrupt or invalid_range:
                partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def tile_metadata(path: Path, column: int, row: int, working: Path) -> tuple[dict, Path]:
    expected = (-180 + (column - 1) * 5, 60 - row * 5,
                -180 + column * 5, 60 - (row - 1) * 5)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
        members = archive.infolist()
        tif = next(info for info in members if info.filename.lower().endswith(".tif"))
        readme = next(info for info in members if info.filename.lower() == "readme.txt")
        readme_text = archive.read(readme).decode("utf-8", errors="replace")
        assert "VERSION 4.1" in readme_text
        local_tif = working / Path(tif.filename).name
        # Fixed member basename prevents archive path traversal.
        with archive.open(tif) as source, local_tif.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        (RAW / "product_readme.txt").write_text(readme_text, encoding="utf-8")
        inventory = [{"name": info.filename, "bytes": info.file_size,
                      "crc32": f"{info.CRC:08x}"} for info in members]
    with rasterio.open(local_tif) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.width == dataset.height == 6000
        assert np.allclose(tuple(dataset.bounds), expected, atol=1e-8), dataset.bounds
        assert np.allclose(dataset.res, [1 / 1200, 1 / 1200], atol=1e-12)
        metadata = {"tile": path.stem, "url": f"{BASE_URL}/{path.name}",
                    "path": str(path), "bytes": path.stat().st_size,
                    "sha256": digest(path), "zip_crc_verified": True,
                    "zip_members": inventory, "raster_bounds_wsen": list(dataset.bounds),
                    "crs": dataset.crs.to_string(), "width": dataset.width,
                    "height": dataset.height, "resolution_degrees": list(dataset.res),
                    "dtype": dataset.dtypes[0], "nodata": dataset.nodata,
                    "uncompressed_tif_sha256": digest(local_tif)}
    return metadata, local_tif


def coverage_qa(path: Path) -> dict:
    transform = Transformer.from_crs(
        "EPSG:4326", "+proj=aeqd +lat_0=35 +lon_0=-118 +datum=WGS84 +units=m", always_xy=True)
    valid_count = total_count = core_count = core_valid = 0
    minimum, maximum = math.inf, -math.inf
    with rasterio.open(path) as dataset:
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window, masked=True)
            valid = ~np.ma.getmaskarray(values)
            valid_count += int(valid.sum())
            total_count += int(valid.size)
            if valid.any():
                minimum = min(minimum, int(values.data[valid].min()))
                maximum = max(maximum, int(values.data[valid].max()))
            rows = np.arange(window.row_off, window.row_off + window.height)
            columns = np.arange(window.col_off, window.col_off + window.width)
            longitude = dataset.transform.c + (columns + .5) * dataset.transform.a
            latitude = dataset.transform.f + (rows + .5) * dataset.transform.e
            xx, yy = transform.transform(*np.meshgrid(longitude, latitude))
            inside = (np.abs(xx) <= 300000) & (np.abs(yy) <= 300000)
            core_count += int(inside.sum())
            core_valid += int((inside & valid).sum())
        assert tuple(dataset.bounds) == BOUNDS
        assert (dataset.width, dataset.height) == (12000, 9600)
        assert valid_count > 0 and core_valid > 0
        return {"status": "verified", "bounds_wsen": list(dataset.bounds),
                "tile_footprints_cover_full_support_rectangle": True,
                "width": dataset.width, "height": dataset.height,
                "pixel_count": total_count, "valid_elevation_pixels": valid_count,
                "nodata_pixels": total_count - valid_count,
                "valid_fraction": valid_count / total_count,
                "core_aeqd_pixels": core_count, "core_aeqd_valid_pixels": core_valid,
                "core_aeqd_valid_fraction": core_valid / core_count,
                "valid_elevation_min_m": minimum, "valid_elevation_max_m": maximum,
                "nodata": dataset.nodata,
                "nodata_interpretation": "Product ocean mask and any remaining invalid terrain cells are preserved; no coastline-based classification or ocean filling performed.",
                "vertical_reference": "EGM96 orthometric height, meters",
                "ellipsoid_conversion": "h_WGS84 = H_EGM96 + N_EGM96; no height conversion applied to this raster"}


def main() -> None:
    for directory in (RAW, OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
    working = RAW / "working"
    working.mkdir(exist_ok=True)
    assert working.resolve().is_relative_to(RAW.resolve())
    manifest = {"product": PRODUCT, "product_source": "https://srtm.csi.cgiar.org/",
                "vertical_datum_source": USGS_URL, "support_bounds_wsen": list(BOUNDS),
                "horizontal_crs": "EPSG:4326", "vertical_reference": "EGM96 orthometric height, meters",
                "nominal_resolution": "3 arc-seconds, approximately 90 m at equator",
                "processing": "CIAT hole filling with interpolation and auxiliary DEM data, as documented in product readme",
                "citation": "Jarvis A., H.I. Reuter, A. Nelson, E. Guevara, 2008, Hole-filled seamless SRTM data V4, CIAT.",
                "nasa_30m_access_check": {"status": "Earthdata authentication required", "http_status": 302,
                    "redirect_host": "urs.earthdata.nasa.gov", "metadata_source": "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=SRTMGL1&version=003"},
                "tiles": []}
    rasters = []
    for column, row in TILES:
        filename = f"srtm_{column:02d}_{row:02d}.zip"
        status("downloading", tile=filename, tiles_verified=len(rasters), tiles_expected=len(TILES))
        download(f"{BASE_URL}/{filename}", RAW / filename)
        metadata, raster = tile_metadata(RAW / filename, column, row, working)
        manifest["tiles"].append(metadata)
        rasters.append(raster)
        save(MANIFEST, manifest)
        print(f"Verified {filename}: {metadata['bytes']} bytes, bbox={metadata['raster_bounds_wsen']}", flush=True)
    status("building_clipped_dem", tiles_verified=len(rasters))
    output = OUTPUT / "srtm_cgiar_v41_90m_support_123W_113W_31N_39N.tif"
    temporary = output.with_suffix(".tmp.tif")
    merge(rasters, bounds=BOUNDS, res=1 / 1200, nodata=-32768, dtype="int16",
          mem_limit=64, dst_path=str(temporary),
          dst_kwds={"compress": "DEFLATE", "predictor": 2, "tiled": True,
                    "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"})
    with rasterio.open(temporary, "r+") as dataset:
        dataset.update_tags(PRODUCT=PRODUCT, VERTICAL_DATUM="EGM96", VERTICAL_UNITS="metre",
                            AREA_OR_POINT="Area", SOURCE="https://srtm.csi.cgiar.org/",
                            HEIGHT_TYPE="orthometric", RESAMPLING="none; original 3 arc-second grid")
    status("verifying_clipped_dem", tiles_verified=len(rasters))
    qa = coverage_qa(temporary)
    temporary.replace(output)
    qa_path = OUTPUT / "srtm_coverage_qa.json"
    save(qa_path, qa)
    manifest["output"] = {"path": str(output), "bytes": output.stat().st_size,
                          "sha256": digest(output), "coverage_qa": str(qa_path),
                          "resolution_degrees": [1 / 1200, 1 / 1200], "bounds_wsen": list(BOUNDS)}
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(MANIFEST, manifest)
    for path in rasters:
        assert path.resolve().is_relative_to(working.resolve())
        path.unlink()
    working.rmdir()
    status("complete", manifest=str(MANIFEST), terrain=str(output), coverage_qa=str(qa_path),
           tiles_verified=len(rasters), raw_zip_bytes=sum(tile["bytes"] for tile in manifest["tiles"]),
           terrain_bytes=output.stat().st_size, valid_fraction=qa["valid_fraction"],
           core_valid_fraction=qa["core_aeqd_valid_fraction"])
    print(json.dumps({"terrain": str(output), "qa": qa}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise
