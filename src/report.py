"""Render a self-contained HTML report that shows the model's reasoning for each incident."""
import html
import json

import config
from evaluate import evaluate, load_explanations
from utils import load_schedule

TYPE_COLORS = {"spike": "#d9480f", "sustained": "#7048e8", "fluctuations": "#0c8599"}

CSS = """
:root{--bg:#fafafa;--card:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--pos:#cf222e;--neg:#1a7f37;--accent:#0969da;--shade:.14}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--pos:#ff7b72;--neg:#56d364;--accent:#58a6ff;--shade:.22}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpi b{display:block;font-size:22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{height:10px;border-radius:2px;display:inline-block;vertical-align:middle}.wrap{overflow-x:auto}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.summary{font-size:15px;margin:8px 0 12px}.out{color:var(--pos);font-weight:600}svg text{fill:var(--muted);font-size:11px}
"""


def _fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def _tag(text, color):
    return f'<span class="tag" style="background:{color}">{html.escape(str(text))}</span>'


def _incidents(records):
    groups = {}
    for r in records:
        if r["is_anomaly"]:
            groups.setdefault(r.get("incident_id") or f"t{r['epoch']}", []).append(r)
    return list(groups.values())


def _timeline(records, schedule):
    if len(records) < 2:
        return "<p class='muted'>Not enough samples for a timeline.</p>"
    w, h, pad = 1000, 220, 36
    t0, t1 = records[0]["epoch"], records[-1]["epoch"]
    vals = [r["values"]["cpu_total"] if "values" in r else
            next(f["value"] for f in r["features"] if f["feature"] == "cpu_total") for r in records]
    vmax = max(vals) * 1.1 or 1
    x = lambda t: pad + (t - t0) / ((t1 - t0) or 1) * (w - 2 * pad)
    y = lambda v: h - pad - v / vmax * (h - 2 * pad)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="CPU timeline">']
    for ev in schedule:
        if ev["end"] < t0 or ev["start"] > t1:
            continue
        x0, x1 = x(max(ev["start"], t0)), x(min(ev["end"], t1))
        parts.append(f'<rect x="{x0:.1f}" y="{pad}" width="{max(x1 - x0, 1):.1f}" height="{h - 2 * pad}" '
                     f'fill="{TYPE_COLORS.get(ev["type"], "#888")}" style="fill-opacity:var(--shade)">'
                     f'<title>induced {ev["type"]}</title></rect>')
    pts = " ".join(f"{x(r['epoch']):.1f},{y(v):.1f}" for r, v in zip(records, vals))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>')
    for r, v in zip(records, vals):
        if r["is_anomaly"]:
            parts.append(f'<circle cx="{x(r["epoch"]):.1f}" cy="{y(v):.1f}" r="2.5" fill="var(--pos)">'
                         f'<title>{html.escape(r["timestamp"])}: {html.escape(r["pattern"])}</title></circle>')
    parts.append(f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="var(--line)"/>')
    parts.append(f'<text x="{pad}" y="{pad - 8}">{vmax:.2f} cores</text><text x="{pad}" y="{h - pad + 16}">'
                 f'{html.escape(records[0]["timestamp"])}</text><text x="{w - pad}" y="{h - pad + 16}" '
                 f'text-anchor="end">{html.escape(records[-1]["timestamp"])}</text></svg>')
    legend = " ".join(_tag(f"induced {k}", c) for k, c in TYPE_COLORS.items())
    return "".join(parts) + f'<p class="muted">Blue: total busy cores. Red dots: flagged samples. {legend}</p>'


def _contribution_rows(exp):
    feats = sorted(exp["features"], key=lambda f: f["contribution"], reverse=True)
    scale = max(abs(f["contribution"]) for f in feats) or 1
    rows = []
    for f in feats:
        width = abs(f["contribution"]) / scale * 120
        color = "var(--pos)" if f["contribution"] > 0 else "var(--neg)"
        status = f'<span class="out">{f["status"]}</span>' if f["status"] != "within normal range" else f["status"]
        rows.append(
            f"<tr><td>{html.escape(f['description'])}<br><span class='muted'>{f['feature']}</span></td>"
            f"<td class='num'>{f['value']:.3f}</td><td class='num'>{f['normal_low']:.3f} – {f['normal_high']:.3f}</td>"
            f"<td class='num'>{f['z_score']:+.1f}σ</td><td>{status}</td>"
            f"<td class='num'>{f['contribution']:+.4f}</td><td><span class='bar' style='width:{width:.0f}px;background:{color}'></span></td></tr>")
    return ("<div class='wrap'><table><tr><th>Feature</th><th>Value</th><th>Normal range (p1–p99)</th><th>z</th>"
            "<th>Status</th><th>Contribution</th><th>Red: towards anomaly · green: towards normal</th></tr>"
            + "".join(rows) + "</table></div>")


def _incident_card(i, group, schedule):
    # Feature evidence comes from the most anomalous sample; the pattern and
    # narrative come from the last one, where the incident label is final.
    worst = min(group, key=lambda r: r["decision_score"])
    final = group[-1]
    worst = {**worst, "pattern": final["pattern"], "pattern_reason": final["pattern_reason"]}
    induced = next((r["induced"] for r in group if r.get("induced")), None)
    induced_tag = _tag(f"induced: {induced}", TYPE_COLORS.get(induced, "#888")) if induced else _tag("not induced", "#6e7781")
    cfs = "".join(
        f"<li>{html.escape(c['description'])}: {c['current']:.3f} → "
        f"{'normal at ' + format(c['normal_if'], '.3f') if c['normal_if'] is not None else 'no single-feature fix'}</li>"
        for c in worst.get("counterfactuals", []))
    return f"""
<div class="card">
  <div><b>Incident {i}</b> · {html.escape(group[0]['timestamp'])} to {html.escape(group[-1]['timestamp'])}
   · {len(group)} flagged samples · explained as {_tag(worst['pattern'], TYPE_COLORS.get(worst['pattern'], '#57606a'))} {induced_tag}</div>
  <p class="summary">{html.escape(final['summary'])}</p>
  <p class="muted">Above: explanation at the end of the incident ({html.escape(final['timestamp'])}). Table below: the most anomalous sample ({html.escape(worst['timestamp'])}):
   decision score {worst['decision_score']:+.4f}, confidence {_pct(worst['confidence'])},
   isolated in {worst['path_length']:.1f} splits vs {worst['typical_path_length']:.1f} typical,
   attribution via {html.escape(worst['attribution_method'])}.</p>
  <p><b>Pattern rule:</b> {html.escape(worst['pattern_reason'])}</p>
  {('<p><b>Counterfactuals</b> (one feature changed, others fixed):</p><ul>' + cfs + '</ul>') if cfs else ''}
  {_contribution_rows(worst)}
</div>"""


def build_report(path=None):
    records = load_explanations()
    schedule = load_schedule()
    try:
        with open(config.GLOBAL_REPORT_PATH) as f:
            glob = json.load(f)
    except FileNotFoundError:
        glob = None
    ev = evaluate(records, schedule, save=True) if records else None
    incidents = _incidents(records)

    kpis = ""
    if ev:
        items = [("Samples scored", ev["samples"]), ("Flagged", ev["flagged"]),
                 ("Events detected", f"{ev['events_detected']}/{ev['events']}"),
                 ("Sample precision", _pct(ev["sample_precision"])),
                 ("Pattern accuracy", _pct(ev["pattern_accuracy"])),
                 ("Mean latency", f"{_fmt(ev['mean_latency_s'], 1)} s")]
        kpis = "<div class='kpis'>" + "".join(f"<div class='kpi'><span class='muted'>{k}</span><b>{v}</b></div>" for k, v in items) + "</div>"

    global_html = "<p class='muted'>No global explanation found. Train the model first.</p>"
    if glob:
        top = max(i["share"] for i in glob["feature_importance"]) or 1
        imp = "".join(
            f"<tr><td>{html.escape(i['description'])}<br><span class='muted'>{i['feature']}</span></td>"
            f"<td class='num'>{i['share']:.1%}</td><td><span class='bar' style='width:{i['share'] / top * 200:.0f}px;"
            f"background:var(--accent)'></span></td></tr>" for i in glob["feature_importance"])
        ranges = "".join(
            f"<tr><td>{html.escape(r['description'])}</td><td class='num'>{r['low']:.3f}</td>"
            f"<td class='num'>{r['typical']:.3f}</td><td class='num'>{r['high']:.3f}</td><td>{r['unit']}</td></tr>"
            for r in glob["normal_ranges"])
        global_html = f"""
<p class="muted">Trained {html.escape(glob.get('trained_at', ''))} on {glob['n_train']} samples, {glob['n_estimators']} trees,
 contamination {glob['contamination']}. Normal samples are isolated in about {glob['typical_path_length']:.1f} splits.</p>
<div class="card"><b>Global feature importance</b> <span class="muted">(mean |{html.escape(glob['attribution_method'])}| over training data)</span>
<div class="wrap"><table><tr><th>Feature</th><th>Share</th><th></th></tr>{imp}</table></div></div>
<div class="card"><b>What the model learned as normal</b>
<div class="wrap"><table><tr><th>Feature</th><th>p1</th><th>Typical (p50)</th><th>p99</th><th>Unit</th></tr>{ranges}</table></div></div>"""

    cards = "".join(_incident_card(i + 1, g, schedule) for i, g in enumerate(incidents[:50]))
    more = f"<p class='muted'>Showing the first 50 of {len(incidents)} incidents.</p>" if len(incidents) > 50 else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CPU Anomaly Explanations</title><style>{CSS}</style></head>
<body><main>
<h1>CPU anomaly explanations</h1>
<p class="muted">Why the IsolationForest flagged each incident: feature contributions, distance from normal, isolation depth,
 counterfactuals and the pattern rule that fired.</p>
{kpis}
<h2>Timeline</h2><div class="card">{_timeline(records, schedule)}</div>
<h2>Model overview</h2>{global_html}
<h2>Incidents ({len(incidents)})</h2>{cards or "<p class='muted'>No anomalies flagged.</p>"}{more}
</main></body></html>"""
    out = path or config.HTML_REPORT_PATH
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Report written to {out}")
    return out


if __name__ == "__main__":
    build_report()
