"""Run the whole pipeline on synthetic CPU data: no Docker or Prometheus needed.

    python src/demo_offline.py

Writes everything to data/demo/ (model, explanations.jsonl, evaluation.json,
explanation_report.html) so the real data/ folder is untouched.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data" / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import features as F  # noqa: E402
from detect import compact  # noqa: E402
from explain import IncidentTracker  # noqa: E402
from report import build_report  # noqa: E402
from train import fit  # noqa: E402
from utils import is_anomaly_induced, now_str  # noqa: E402

CORES = 4


def synth_stream(n, rng, events=(), t0=0.0):
    """Synthetic raw snapshots: a noisy idle baseline plus the given induced events."""
    rows, load = [], 0.3
    for i in range(n):
        epoch = t0 + i
        cpu = max(0.02, 0.35 + 0.08 * np.sin(i / 90) + rng.normal(0, 0.06))
        for ev in events:
            s, e = ev["start"], ev["end"]
            if s <= epoch < e:
                if ev["type"] == "fluctuations":
                    period = ev["high"] + ev["low"]
                    if (epoch - s) % period < ev["high"]:
                        cpu = CORES * ev["load_fraction"] * rng.uniform(0.85, 0.98)
                else:
                    cpu = CORES * ev["load_fraction"] * rng.uniform(0.85, 0.98)
        load += (cpu - load) / 60
        rows.append({"timestamp": now_str(epoch), "epoch": epoch, "cpu_total": cpu,
                     "cpu_user": cpu * 0.72, "cpu_system": cpu * 0.25,
                     "cpu_iowait": abs(rng.normal(0.01, 0.005)), "load1": load})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(7)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in (config.EXPLANATIONS_PATH, config.SCHEDULE_PATH):
        if p.exists():
            p.unlink()

    print("== Training on 900 s of synthetic normal load")
    train_raw = synth_stream(900, rng, t0=time.time() - 3000)
    explainer = fit(train_raw)

    t0 = time.time() - 1500
    plan = [("spike", 90, 15, {"load_fraction": 1.0}),
            ("sustained", 260, 75, {"load_fraction": 0.6}),
            ("fluctuations", 470, 64, {"load_fraction": 1.0, "cycles": 4, "high": 8, "low": 8}),
            ("spike", 680, 12, {"load_fraction": 1.0}),
            ("sustained", 840, 60, {"load_fraction": 0.6})]
    events = []
    for kind, offset, dur, extra in plan:
        s = t0 + offset
        events.append({"type": kind, "start": s, "end": s + dur, "start_str": now_str(s), "end_str": now_str(s + dur), **extra})
    with open(config.SCHEDULE_PATH, "w") as f:
        json.dump(events, f, indent=2)

    print("\n== Detecting on 1000 s of synthetic load with 5 induced anomalies")
    stream = synth_stream(1000, rng, events, t0=t0)
    builder = F.FeatureBuilder()
    tracker = IncidentTracker(explainer)
    with open(config.EXPLANATIONS_PATH, "w") as log:
        for rec in stream.to_dict("records"):
            feats = builder.update(rec)
            if feats is None:
                continue
            exp = explainer.explain(feats, timestamp=rec["timestamp"])
            exp["epoch"] = rec["epoch"]
            tracker.update(exp, rec["epoch"], feats["cpu_total"])
            exp["induced"] = is_anomaly_induced(rec["epoch"], events)
            log.write(json.dumps(exp if exp["is_anomaly"] else compact(exp)) + "\n")

    print("\n== Example explanations (last flagged sample of each incident)")
    last = {}
    with open(config.EXPLANATIONS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["is_anomaly"] and r.get("induced"):
                last[r["incident_id"]] = r
    shown = set()
    for r in last.values():
        if r["induced"] not in shown:
            shown.add(r["induced"])
            print(f"\n[induced {r['induced']}] {r['summary']}")

    print("\n== Evaluation")
    build_report()


if __name__ == "__main__":
    main()
