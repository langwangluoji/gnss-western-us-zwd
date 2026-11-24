"""Contract-locked E01-E05 experiment launcher.

This first implementation validates the five mechanism switches and delegates
training to the existing p15 formal runner until each mechanism's model branch
is enabled. It never changes the p14 evaluation contract.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .optimization_campaign import Campaign

MECHANISMS = {
    "E01": {"terrain": True, "edges": False, "history": False, "delta_loss": False, "residual": False},
    "E02": {"terrain": True, "edges": True, "history": False, "delta_loss": False, "residual": False},
    "E03": {"terrain": True, "edges": True, "history": True, "delta_loss": False, "residual": False},
    "E04": {"terrain": True, "edges": True, "history": True, "delta_loss": True, "residual": False},
    "E05": {"terrain": True, "edges": True, "history": True, "delta_loss": True, "residual": True},
}

def run(experiment: str, sampling: str, seconds: float, out: Path) -> dict:
    if experiment not in MECHANISMS: raise ValueError(experiment)
    if sampling not in ("random", "hole100"): raise ValueError(sampling)
    c = Campaign(sampling, "fixed3000")
    switches = MECHANISMS[experiment]
    contract = {"experiment": experiment, "sampling": sampling, "seconds": seconds,
                "seeds": [42, 17], "station_contract": [410, 137], "layouts": 3,
                "hours": 12*24, "protocol": "p14-expanded-v1", "switches": switches,
                "forbidden": ["PWV", "IWV", "TCWV", "C", "D"],
                "fit_events": len(c.fit_events), "monitor_events": len(c.monitor)}
    out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(contract, sort_keys=True).encode(); contract["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract

def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment", choices=sorted(MECHANISMS), required=True)
    p.add_argument("--sampling", choices=["random", "hole100"], required=True); p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--out", type=Path, required=True); a=p.parse_args(); print(json.dumps(run(a.experiment,a.sampling,a.seconds,a.out), indent=2))
if __name__ == "__main__": main()
