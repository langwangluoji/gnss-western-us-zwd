"""Read-only verification and mean-swap diagnostic for the saved C2019 EDM members."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent
PILOT = OUTPUT.parent
PROJECT = OUTPUT.parents[2]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "tools")]

from gnss_field.artifacts import sha256
from gnss_field.evaluation.metrics import point_metrics
from local_probability_experiment import joint_series

RESULTS = PROJECT / "results"
PROBE = RESULTS / "20261119_local_condition_probe_v01"
METHOD = RESULTS / "20261119_local_method_research_v02"
SEED = 20261117
REPLICATES = 2000
METRICS = ("ES", "VS")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_heun_gain():
    """Independent Heun recurrence for d(x)/d(sigma) = sigma*x/(1+sigma**2)."""
    ramp = np.linspace(0, 1, 18)
    schedule = (80 ** (1 / 7) + ramp * (.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
    schedule = np.r_[schedule, 0.]
    current = 80.
    for i, (sigma, following) in enumerate(zip(schedule[:-1], schedule[1:])):
        step = following - sigma
        derivative = sigma * current / (1 + sigma ** 2)
        predictor = current + step * derivative
        if i == len(schedule) - 2:
            current = predictor
        else:
            corrected = following * predictor / (1 + following ** 2)
            current += .5 * step * (derivative + corrected)
    return float(current)


def crossed_intervals(hours, model, reference):
    np.testing.assert_array_equal(np.isfinite(model), np.isfinite(reference))
    active = np.isfinite(model[..., 0]).any(0)
    delta = (model - reference)[:, active]
    result = {"difference_mm": dict(zip(METRICS, np.nanmean(delta, axis=0).mean(0).tolist()))}
    for days in (7, 14):
        block = (hours - hours[0]) // (24 * days)
        indices = np.unique(block)
        sums = np.array([np.nansum(delta[block == i], axis=0) for i in indices])
        counts = np.array([np.isfinite(delta[block == i, :, 0]).sum(0) for i in indices])
        group_count = delta.shape[1]
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(REPLICATES):
            t = rng.integers(0, len(indices), len(indices))
            g = rng.integers(0, group_count, group_count)
            denominator = counts[t][:, g].sum(0)
            assert (denominator > 0).all()
            draws.append((sums[t][:, g].sum(0) / denominator[:, None]).mean(0))
        result[str(days)] = dict(intervals_mm=dict(zip(METRICS, np.quantile(draws, [.025, .975], axis=0).T.tolist())),
                                 n_time_blocks=len(indices), n_groups=group_count)
    return result


def check_training_inputs(lock, selection, dependencies):
    b = dict(np.load(PROBE / "conditions_2018.npz"))
    train = np.isin(b["hours"], lock["training_hours"])
    development = np.isin(b["hours"], lock["development_hours"])
    assert (int(train.sum()), int(development.sum())) == (576, 288)
    assert not (set(b["hours"][train] // 168) & set(b["hours"][development] // 168))
    with np.load(METHOD / "observable_subspace.npz") as v:
        np.testing.assert_array_equal(v["B_station_ids"], b["station_ids"])
        np.testing.assert_array_equal(v["availability"], b["valid"].mean(0))
        vectors = v["right_vectors"][:64]
        np.testing.assert_allclose(vectors @ vectors.T, np.eye(64), rtol=0., atol=1e-12)
    with np.load(PILOT / "condition_pca.npz") as pca:
        x = b["features"][train, :-16].astype("f8")
        np.testing.assert_array_equal(x.mean(0), pca["center"])
        np.testing.assert_array_equal(x.std(0).clip(1e-5), pca["scale"])
        centered = np.where(pca["active"], (x - pca["center"]) / pca["scale"], 0.)
        retained = centered @ pca["projection"]
        np.testing.assert_allclose(retained.T @ retained, np.diag(pca["singular_values"][:32] ** 2),
                                   rtol=1e-9, atol=1e-8)
    recorded = [
        (0, read_json(PILOT / "scientific/initial_metrics.json")["joint"]["value"])
    ]
    for line in (PILOT / "scientific/development.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not item["startup"]:
            recorded.append((item["step"], item["metrics"]["joint"]["value"]))
    assert min(recorded, key=lambda row: (row[1], row[0])) == (selection["step"], selection["value"])
    assert sha256(Path(selection["checkpoint"])) == selection["sha256"]
    checks = {
        "pca": PILOT / "condition_pca.npz",
        "projection": METHOD / "observable_subspace.npz",
        "covariance": RESULTS / "20261119_local_fused_probability_v02/B_only_parameters.lock.json",
        "amplitude": RESULTS / "20261119_local_conditional_probability_v01/B_only_parameters.lock.json",
        "source": PILOT / "code_snapshot/local_subspace_data.py",
        "model_source": PILOT / "code_snapshot/subspace.py",
        "runner_source": PILOT / "code_snapshot/local_subspace_pilot.py",
    }
    for key, path in checks.items():
        assert sha256(path) == dependencies[key], key
    return dict(training_hours=int(train.sum()), development_hours=int(development.sum()),
                minimum_gap_hours=int(np.abs(b["hours"][train, None] - b["hours"][None, development]).min()),
                # The base mean, covariance/scale and operator were frozen before training this head.
                scope="EDM head and condition PCA only. Frozen B2018 background reused B development weeks.",
                selection_values=[dict(step=step, value=value) for step, value in recorded]), list(checks.values())


def review_members(forecast, metadata, members):
    h, y, p = forecast["hours"], forecast["truth"], forecast["population"]
    analytic = forecast["analytic_mean"].astype(float)
    edm = forecast["samples"][:, :members].astype(float)
    gauss = forecast["reference"][:, :members].astype(float)
    e_mean, g_mean = edm.mean(1), gauss.mean(1)
    e_centered = edm - e_mean[:, None]
    g_centered = gauss - g_mean[:, None]
    variants = {
        "EDM": edm,
        "Gaussian": gauss,
        "EDM_shape_Gaussian_mean": e_centered + g_mean[:, None],
        "Gaussian_shape_EDM_mean": g_centered + e_mean[:, None],
        "EDM_shape_locked_mean": e_centered + analytic[:, None],
        "Gaussian_shape_locked_mean": g_centered + analytic[:, None],
    }
    results, series = {}, {}
    for name, values in variants.items():
        scores, names = joint_series(y, values, p, metadata)
        series[name] = scores
        valid_groups = np.isfinite(scores[..., 0]).any(0)
        results[name] = dict(zip(METRICS, np.nanmean(scores[:, valid_groups], axis=0).mean(0).tolist()))
        results[name]["station_macro_rmse_mm"] = point_metrics(y, values.mean(1), p)["station_macro_rmse_mm"]
    contrasts = {
        "original_EDM_minus_Gaussian": ("EDM", "Gaussian"),
        "same_locked_mean": ("EDM_shape_locked_mean", "Gaussian_shape_locked_mean"),
        "same_Gaussian_empirical_mean": ("EDM_shape_Gaussian_mean", "Gaussian"),
        "same_EDM_empirical_mean": ("EDM", "Gaussian_shape_EDM_mean"),
    }
    scores = series["EDM"]
    valid_groups = np.isfinite(scores[..., 0]).any(0)
    named_groups = np.asarray(names)[valid_groups]
    counts = np.array([p[:, metadata.dense_group == g].sum(1) for g in named_groups]).T
    np.testing.assert_array_equal(counts >= 4, np.isfinite(scores[:, valid_groups, 0]))
    drift = e_mean - g_mean
    return dict(products=results, crossed={key: crossed_intervals(h, series[a], series[b])
                                         for key, (a, b) in contrasts.items()},
                mean_drift_rms_mm=float(np.sqrt(np.mean(drift[p] ** 2))),
                mean_drift_bias_mm=float(drift[p].mean()),
                centered_member_difference_rms_mm=float(np.sqrt(np.mean(
                    ((e_centered - g_centered).transpose(0, 2, 1)[p]) ** 2))),
                station_member_std_difference_rms_mm=float(np.sqrt(np.mean(
                    (edm.std(1)[p] - gauss.std(1)[p]) ** 2))),
                group_ids=named_groups.tolist(), group_hours=int((counts >= 4).sum()),
                joint_station_hours=int(counts[counts >= 4].sum()))


def main():
    lock = read_json(PILOT / "pilot.lock.json")
    selection = read_json(PILOT / "selection.lock.json")
    preflight = read_json(PILOT / "preflight.json")
    contract = read_json(PILOT / "scientific/contract.json")
    training, paths = check_training_inputs(lock, selection, contract["spec"]["dependencies"])
    direct = scalar_heun_gain()
    assert abs(direct - preflight["gaussian_heun_gain"]) < 1e-12
    forecast = dict(np.load(PILOT / "C2019_forecast.npz"))
    with np.load(PROBE / "conditions_2019.npz") as c:
        for key, expected in [("hours", "hours"), ("station_ids", "station_ids"),
                              ("truth", "truth"), ("population", "valid")]:
            np.testing.assert_array_equal(forecast[key], c[expected])
    station_path = PROJECT / "../../02_data/processed/gnss_field_server_v1/stations.csv"
    metadata = pd.read_csv(station_path).iloc[forecast["station_ids"]].reset_index(drop=True)
    assert (metadata.role == "development").all()
    pop = forecast["population"]
    assert len(pop) == 288 and int(pop.sum()) == 20909 and int(pop.any(0).sum()) == 76
    paths.extend([station_path, PROBE / "conditions_2018.npz", PROBE / "conditions_2019.npz",
                  PILOT / "pilot.lock.json", PILOT / "selection.lock.json", PILOT / "preflight.json",
                  PILOT / "scientific/contract.json", PILOT / "scientific/initial_metrics.json",
                  PILOT / "scientific/development.jsonl", Path(selection["checkpoint"]),
                  PILOT / "C2019_forecast.npz", PILOT / "C2019_results.json"])
    report = dict(
        scope="Read-only replay of the saved pilot outputs. C2019 is development, D2020 was not read.",
        engineering_preflight=preflight,
        gain=dict(independent=direct, locked=preflight["gaussian_heun_gain"],
                  absolute_error=abs(direct - preflight["gaussian_heun_gain"])),
        training=training, evaluation={"hours": int(len(pop)), "stations_S0": int(pop.any(0).sum()),
            "station_hours_S0": int(pop.sum()),
            "locked_point_rmse_mm": point_metrics(forecast["truth"], forecast["analytic_mean"], pop)["station_macro_rmse_mm"]},
        results={str(k): review_members(forecast, metadata, k) for k in (64, 128)},
        bootstrap={"replicates": REPLICATES, "seed": SEED,
                   "scope": "Paired calendar blocks and fixed 50 km dense groups, group macro"},
        diagnostic="Empirical mean alignment uses the same finite ensemble and couples its members. It is a decomposition diagnostic, not an iid forecast or a separately fitted model.",
        inputs_sha256={p.relative_to(PROJECT).as_posix(): sha256(p) for p in paths},
    )
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gain": report["gain"], "population": report["evaluation"],
                      "K128": report["results"]["128"]["crossed"]}, indent=2))


if __name__ == "__main__":
    main()

