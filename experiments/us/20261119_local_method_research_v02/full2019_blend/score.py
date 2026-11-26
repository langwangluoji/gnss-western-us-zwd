"""Reproduce the B-locked constant blend on the full C2019 scoring population."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from gnss_field.evaluation.bootstrap import _blocks
from gnss_field.evaluation.metrics import increment_metrics, pair_increment_arrays, point_metrics

AUDIT = PROJECT / "results/20261119_frozen_m0_audit"
M0_PATH = AUDIT / "release/seed_42/mean_base/full2019/5000"
CONTROL_PATH = AUDIT / "monthly"
IDW_REFERENCE = PROJECT / "results/local_baselines_2019_20261118/A_IDW"
WEIGHT_LOCK = OUTPUT.parent.parent / "20261119_local_method_research_v01/B_only_parameters.lock.json"
STATIONS = PROJECT / "../../02_data/processed/gnss_field_server_v1/stations.csv"
REPLICATES = 2000
SEED = 20261117
WEIGHT = 0.26112698785540106


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bootstrap(block_sums, block_counts, *, crossed=False):
    rng = np.random.default_rng(SEED)
    block_count, columns = block_counts.shape
    differences = []
    attempts = 0
    while len(differences) < REPLICATES:
        indices = rng.integers(0, block_count, size=block_count)
        groups = rng.integers(0, columns, size=columns) if crossed else slice(None)
        counts = block_counts[indices][:, groups].sum(0)
        attempts += 1
        if (counts == 0).any():
            continue
        values = [np.sqrt(sums[indices][:, groups].sum(0) / counts).mean()
                  for sums in block_sums]
        differences.append([values[2] - values[0], values[2] - values[1]])
    draws = np.asarray(differences)
    intervals = dict(zip(("blend_minus_M0", "blend_minus_A_IDW"),
                         np.quantile(draws, (.025, .975), axis=0).T.tolist()))
    return dict(ci95_mm=intervals, seed=SEED, block_days=7, n_blocks=block_count,
                replicates=REPLICATES, attempts=attempts, crossed_dense_groups=crossed)


def main():
    lock = json.loads(WEIGHT_LOCK.read_text(encoding="utf-8"))
    assert lock["blend"]["weights"] == [WEIGHT]
    monthly = []
    inputs = [WEIGHT_LOCK, STATIONS]
    max_m0_error, max_idw_error = 0., 0.
    for month in range(1, 13):
        label = f"2019-{month:02d}"
        m0_file = M0_PATH / f"forecast_{label}.npz"
        control_file = CONTROL_PATH / f"{label}.npz"
        idw_file = IDW_REFERENCE / f"forecast_{label}.npz"
        inputs.extend((m0_file, control_file, idw_file))
        with np.load(m0_file) as m, np.load(control_file) as a, np.load(idw_file) as ref:
            np.testing.assert_array_equal(m["hours"], a["hours"])
            np.testing.assert_array_equal(m["truth"], a["truth"])
            f0 = m["population_F"]
            s0 = f0 & m["normal_support_S0"]
            np.testing.assert_array_equal(s0, a["mask"])
            sample_idx = np.searchsorted(m["hours"], ref["hours"])
            np.testing.assert_array_equal(m["hours"][sample_idx], ref["hours"])
            np.testing.assert_array_equal(m["station_ids"], ref["station_ids"])
            max_m0_error = max(max_m0_error, float(np.max(abs(m["prediction"] - a["mean"]))))
            max_idw_error = max(max_idw_error, float(np.max(abs(a["idw"][sample_idx] - ref["prediction"]))))
            monthly.append(dict(hours=m["hours"], truth=m["truth"], population_F=f0,
                                population_S0=s0, station_ids=m["station_ids"],
                                M0=m["prediction"].astype("f8"), A_IDW=a["idw"].astype("f8")))
    assert max_m0_error <= .001 and max_idw_error <= .001
    ids = monthly[0]["station_ids"]
    for row in monthly:
        np.testing.assert_array_equal(ids, row["station_ids"])
    merged = {k: np.concatenate([r[k] for r in monthly]) for k in monthly[0] if k != "station_ids"}
    hours, truth, pop = merged["hours"], merged["truth"], merged["population_S0"]
    assert len(hours) == 8760 and np.all(np.diff(hours) == 1)
    assert np.count_nonzero(pop.any(0)) == 76 and int(pop.sum()) == 642258
    metadata = pd.read_csv(STATIONS).iloc[ids].reset_index(drop=True)
    assert (metadata["role"] == "development").all()
    m0, idw = merged["M0"], merged["A_IDW"]
    blend = m0 + WEIGHT * (idw - m0)
    predictions = dict(M0=m0, A_IDW=idw, blend=blend)
    blocks = _blocks(hours, 7)
    active = pop.any(0)
    station_counts = np.array([pop[i].sum(0)[active] for i in blocks])
    point_sums = [np.array([np.where(pop[i], (p[i] - truth[i]) ** 2, 0).sum(0)[active]
                            for i in blocks]) for p in predictions.values()]
    pairs = [pair_increment_arrays(truth, p, pop, metadata) for p in predictions.values()]
    in_range = pairs[0]["distance_km"] <= 50
    paired_pop = pairs[0]["population"] & in_range[None]
    pair_groups = pairs[0]["group_ids"]
    group_masks = [pair_groups == g for g in dict.fromkeys(pair_groups[paired_pop.any(0)])]
    group_counts = np.array([[paired_pop[i][:, g].sum() for g in group_masks] for i in blocks])
    pair_sums = [np.array([[np.where(paired_pop[i][:, g], (p["prediction"][i][:, g]
                 - p["truth"][i][:, g]) ** 2, 0).sum() for g in group_masks] for i in blocks]) for p in pairs]
    metrics = {k: point_metrics(truth, p, pop) for k, p in predictions.items()}
    increments = {k: increment_metrics(truth, p, pop, metadata)["0-50"] for k, p in predictions.items()}
    point_ci = bootstrap(point_sums, station_counts)
    point_ci["difference_mm"] = {f"blend_minus_{k}": metrics["blend"]["station_macro_rmse_mm"]
                                - metrics[k]["station_macro_rmse_mm"] for k in ("M0", "A_IDW")}
    increment_ci = bootstrap(pair_sums, group_counts, crossed=True)
    increment_ci["difference_mm"] = {f"blend_minus_{k}": increments["blend"]["group_macro_rmse_mm"]
                                    - increments[k]["group_macro_rmse_mm"] for k in ("M0", "A_IDW")}
    payload = dict(weight=WEIGHT, fit_scope="locked B2018 864-hour exploratory pool",
                   evaluation_scope="C2019 full-year development",
                   formula="M0 + weight * (A_IDW - M0)", score_population="S0 intersected with F0",
                   hours=len(hours), stations=int(active.sum()), station_hours=int(pop.sum()),
                   full_f0_stations=int(merged["population_F"].any(0).sum()),
                   full_f0_station_hours=int(merged["population_F"].sum()),
                   maximum_M0_reconstruction_error_mm=max_m0_error,
                   maximum_A_IDW_check_error_mm=max_idw_error, metrics=metrics,
                   increments_0_50km=increments, point_intervals=point_ci,
                   increment_intervals=increment_ci,
                   inputs_sha256={p.relative_to(PROJECT).as_posix() if p.is_relative_to(PROJECT)
                                  else str(p.resolve()): sha256(p) for p in inputs})
    np.savez_compressed(OUTPUT / "forecast_2019.npz", station_ids=ids, blend=blend, **merged)
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(metrics=metrics, point_intervals=point_ci,
                          increments_0_50km=increments, increment_intervals=increment_ci), indent=2))


if __name__ == "__main__":
    main()

