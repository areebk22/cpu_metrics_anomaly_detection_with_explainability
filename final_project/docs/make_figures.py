"""Generate the README figures from a run's output (defaults to the offline demo).

    pip install matplotlib
    python src/demo_offline.py
    python docs/make_figures.py            # or: python docs/make_figures.py data
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/demo")
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

TYPE_COLORS = {"spike": "#d9480f", "sustained": "#7048e8", "fluctuations": "#0c8599"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

records = [json.loads(l) for l in open(RUN / "explanations.jsonl") if l.strip()]
schedule = json.load(open(RUN / "anomaly_schedule.json"))
glob = json.load(open(RUN / "global_explanation.json"))
t0 = records[0]["epoch"]
t = np.array([r["epoch"] - t0 for r in records])


def value(r, name):
    if "values" in r:
        return r["values"][name]
    return next(f["value"] for f in r["features"] if f["feature"] == name)


cpu = np.array([value(r, "cpu_total") for r in records])
score = np.array([r["decision_score"] for r in records])
flag = np.array([r["is_anomaly"] for r in records])


def shade(ax):
    for ev in schedule:
        ax.axvspan(ev["start"] - t0, ev["end"] - t0, color=TYPE_COLORS[ev["type"]], alpha=0.15, lw=0)


# 1. Timeline: CPU, flags, induced windows, decision score
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
shade(a1)
a1.plot(t, cpu, color="#0969da", lw=1, label="total busy cores")
a1.scatter(t[flag], cpu[flag], s=6, color="#cf222e", zorder=3, label="flagged as anomaly")
handles, _ = a1.get_legend_handles_labels()
handles += [Patch(color=c, alpha=0.3, label=f"induced {k}") for k, c in TYPE_COLORS.items()]
a1.set_ylabel("CPU (cores)")
a1.set_ylim(top=cpu.max() * 1.08)
a1.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, frameon=False)
fig.suptitle("Detection timeline: induced anomalies (shaded) vs flagged samples (red)", x=0.01, ha="left")
shade(a2)
a2.plot(t, score, color="#57606a", lw=1)
a2.axhline(0, color="#cf222e", lw=1, ls="--")
a2.fill_between(t, score, 0, where=score < 0, color="#cf222e", alpha=0.3)
a2.set_ylabel("decision score")
a2.set_xlabel("seconds since detection started")
a2.text(t[-1], score.min(), "below dashed line = anomaly", ha="right", va="bottom", color="#cf222e", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "timeline.png")
plt.close(fig)

# 2. Global feature importance
imp = glob["feature_importance"][::-1]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh([i["description"] for i in imp], [i["share"] for i in imp], color="#0969da")
for y, i in enumerate(imp):
    ax.text(i["share"] + 0.003, y, f"{i['share']:.0%}", va="center", fontsize=8)
ax.set_xlabel(f"share of mean |{glob['attribution_method']}| over training data")
ax.set_title("Global feature importance", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "global_importance.png")
plt.close(fig)

# 3. Local explanation: contributions for the strongest sample of one incident per type
full = [r for r in records if r["is_anomaly"] and "features" in r and r.get("induced")]
by_incident = {}
for r in full:
    by_incident.setdefault((r["induced"], r["incident_id"]), []).append(r)
picked = {}
for (k, _), group in by_incident.items():
    if k not in picked:
        picked[k] = group[len(group) // 2]  # midpoint: the incident's settled shape
kinds = [k for k in ("spike", "sustained", "fluctuations") if k in picked]
fig, axes = plt.subplots(1, len(kinds), figsize=(11, 3.8), sharey=True)
axes = np.atleast_1d(axes)
order = [f["feature"] for f in glob["feature_importance"]][::-1]
for ax, k in zip(axes, kinds):
    r = picked[k]
    final = [x for x in records if x.get("incident_id") == r.get("incident_id") and x["is_anomaly"]][-1]
    c = {f["feature"]: f["contribution"] for f in r["features"]}
    vals = [c[f] for f in order]
    ax.barh(order, vals, color=["#cf222e" if v > 0 else "#1a7f37" for v in vals])
    ax.axvline(0, color="#57606a", lw=0.8)
    ax.set_title(f"induced {k}: explained as '{final['pattern']}'", loc="left", color=TYPE_COLORS[k], fontsize=9)
    ax.set_xlabel("contribution (red = towards anomaly)")
    ax.grid(axis="y", visible=False)
fig.suptitle("Why each incident was flagged: SHAP contributions at the midpoint of the incident", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "local_contributions.png")
plt.close(fig)

# 4. Isolation depth: anomalies are isolated in fewer splits
depth = np.array([r["path_length"] for r in records])
fig, ax = plt.subplots(figsize=(7, 3))
bins = np.linspace(depth.min(), depth.max(), 40)
ax.hist(depth[~flag], bins=bins, color="#0969da", alpha=0.7, label="classed normal")
ax.hist(depth[flag], bins=bins, color="#cf222e", alpha=0.7, label="classed anomaly")
ax.axvline(glob["typical_path_length"], color="#57606a", ls="--", lw=1)
ax.text(glob["typical_path_length"], ax.get_ylim()[1] * 0.9, " typical depth for normal data", fontsize=7)
ax.set_xlabel("average isolation depth (splits)")
ax.set_ylabel("samples")
ax.set_title("Isolation depth: anomalies are separated from the data in far fewer splits", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "isolation_depth.png")
plt.close(fig)

print(f"Figures written to {OUT}")
