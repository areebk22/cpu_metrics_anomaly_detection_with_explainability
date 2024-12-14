"""Explainability for the IsolationForest CPU anomaly detector.

For every scored sample the explainer produces evidence from several angles:

  1. Verdict and confidence: decision score, distance from the decision
     boundary, and how unusual the sample is compared with the training data.
  2. Isolation depth: IsolationForest flags points that random splits isolate
     quickly. The average path length is the model's own reasoning, so it is
     reported against the typical depth for normal training samples.
  3. Feature contributions: SHAP values from TreeExplainer, oriented so that
     a positive number pushes the sample towards "anomaly". If shap is not
     installed, an occlusion fallback (replace a feature with its median and
     measure the score change) is used instead.
  4. Baseline comparison: each feature against the normal range seen in training.
  5. Counterfactual: the value a driver would need for the sample to be
     classed as normal, with the other features unchanged.
  6. Pattern: a rule-based label (spike / sustained / fluctuations / drop),
     together with the rule that fired.
  7. Narrative: all of the above as plain language.
"""
import math
import time

import numpy as np

import config
import features as F

try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SHAP = False

_EULER = 0.5772156649


def _avg_path_length(n):
    """Expected path length of an unsuccessful BST search, c(n) in the IsolationForest paper."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + _EULER) - 2.0 * (n - 1) / n


def _fmt(value, unit=""):
    if not unit and float(value).is_integer():
        s = str(int(value))
    elif abs(value) >= 100:
        s = f"{value:,.0f}"
    elif abs(value) >= 10:
        s = f"{value:.1f}"
    else:
        s = f"{value:.2f}"
    return f"{s} {unit}".strip()


def _signed(value, unit=""):
    return ("+" if value > 0 else "") + _fmt(value, unit)


class AnomalyExplainer:
    def __init__(self, model, scaler, baseline, shap_sign=1.0):
        self.model = model
        self.scaler = scaler
        self.baseline = baseline
        self.shap_sign = shap_sign
        self.features = baseline["features"]
        self._shap = None
        self._c_n = _avg_path_length(int(getattr(model, "max_samples_", 256)))

    # ------------------------------------------------------------------ building
    @classmethod
    def fit(cls, model, scaler, X_raw):
        """Learn the baseline statistics the explanations compare against."""
        X_raw = np.asarray(X_raw, dtype=float)
        X_scaled = scaler.transform(X_raw)
        scores = model.score_samples(X_scaled)
        decisions = model.decision_function(X_scaled)
        normal = decisions >= 0

        per_feature = {}
        for i, f in enumerate(F.FEATURES):
            col = X_raw[:, i]
            per_feature[f] = {
                "mean": float(col.mean()),
                "std": float(col.std()) or 1e-9,
                "min": float(col.min()),
                "max": float(col.max()),
                **{f"p{q:02d}": float(np.percentile(col, q)) for q in (1, 5, 50, 95, 99)},
            }
        abs_delta = np.abs(X_raw[:, F.FEATURES.index("cpu_delta")])

        tmp = cls(model, scaler, {"features": list(F.FEATURES)})
        depths = tmp._path_length(scores)
        baseline = {
            "features": list(F.FEATURES),
            "per_feature": per_feature,
            "abs_delta_p99": float(np.percentile(abs_delta, 99)),
            "train_scores_sorted": np.sort(scores).tolist(),
            "median_score": float(np.median(scores)),
            "offset": float(model.offset_),
            "median_decision_normal": float(np.median(decisions[normal])) if normal.any() else 0.1,
            "median_depth_normal": float(np.median(depths[normal])) if normal.any() else float(np.median(depths)),
            "n_train": int(len(X_raw)),
        }
        exp = cls(model, scaler, baseline)
        exp.shap_sign = exp._calibrate_shap_sign(X_scaled)
        return exp

    @classmethod
    def from_bundle(cls, bundle):
        return cls(bundle["model"], bundle["scaler"], bundle["baseline"], bundle.get("shap_sign", 1.0))

    def to_bundle(self):
        return {"model": self.model, "scaler": self.scaler, "baseline": self.baseline, "shap_sign": self.shap_sign}

    # ------------------------------------------------------------------ primitives
    def _path_length(self, scores):
        # score_samples = -2 ** (-E[h(x)] / c(n))  =>  E[h(x)] = -log2(-score) * c(n)
        scores = np.clip(-np.asarray(scores, dtype=float), 1e-12, None)
        return -np.log2(scores) * self._c_n

    def _shap_explainer(self):
        if self._shap is None and _HAS_SHAP:
            import shap
            self._shap = shap.TreeExplainer(self.model)
        return self._shap

    def _calibrate_shap_sign(self, X_scaled):
        """Orient SHAP output so positive = towards anomaly, whatever convention shap uses for IsolationForest."""
        explainer = self._shap_explainer()
        if explainer is None:
            return 1.0
        sample = X_scaled[np.linspace(0, len(X_scaled) - 1, min(200, len(X_scaled))).astype(int)]
        explained = np.asarray(explainer.shap_values(sample)).sum(axis=1)
        corr = np.corrcoef(explained, self.model.score_samples(sample))[0, 1]
        # shap output that rises with score_samples rises with "normality", so flip it.
        return -1.0 if (np.isnan(corr) or corr > 0) else 1.0

    def contributions(self, X_scaled):
        """Per-feature push towards anomaly for each row (positive = more anomalous)."""
        explainer = self._shap_explainer()
        if explainer is not None:
            return self.shap_sign * np.asarray(explainer.shap_values(X_scaled)), "shap"
        # Occlusion fallback: how much more normal does the row get when a feature is set to its median?
        base = self.model.score_samples(X_scaled)
        medians = self.scaler.transform(
            np.array([[self.baseline["per_feature"][f]["p50"] for f in self.features]]))[0]
        out = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            occluded = X_scaled.copy()
            occluded[:, i] = medians[i]
            out[:, i] = self.model.score_samples(occluded) - base
        return out, "occlusion"

    def _decision(self, X_raw):
        return self.model.decision_function(self.scaler.transform(X_raw))

    # ------------------------------------------------------------------ evidence
    def _feature_evidence(self, x):
        rows = []
        for i, f in enumerate(self.features):
            st = self.baseline["per_feature"][f]
            v = float(x[i])
            if v > st["p99"]:
                status = "above normal range"
            elif v < st["p01"]:
                status = "below normal range"
            else:
                status = "within normal range"
            rows.append({
                "feature": f,
                "description": F.describe(f),
                "unit": F.unit(f),
                "value": v,
                "typical": st["p50"],
                "normal_low": st["p01"],
                "normal_high": st["p99"],
                "z_score": (v - st["mean"]) / st["std"],
                "status": status,
            })
        return rows

    def _counterfactual(self, x, feature_idx, target):
        """Smallest move of one feature towards `target` that flips the verdict to normal (others fixed)."""
        steps = np.linspace(0.0, 1.0, 41)
        grid = np.repeat(x[None, :], len(steps), axis=0)
        grid[:, feature_idx] = x[feature_idx] + steps * (target - x[feature_idx])
        ok = np.where(self._decision(grid) >= 0)[0]
        return None if len(ok) == 0 else float(grid[ok[0], feature_idx])

    def _joint_counterfactual(self, x):
        """How far the whole sample must move back towards typical values before it is classed normal."""
        typical = np.array([self.baseline["per_feature"][f]["p50"] for f in self.features])
        steps = np.linspace(0.0, 1.0, 41)
        grid = x[None, :] + steps[:, None] * (typical - x)[None, :]
        ok = np.where(self._decision(grid) >= 0)[0]
        if len(ok) == 0:
            return None
        point = grid[ok[0]]
        return {"fraction": float(steps[ok[0]]),
                "values": {f: float(point[i]) for i, f in enumerate(self.features)}}

    def _classify_pattern(self, feats):
        pf = self.baseline["per_feature"]
        cpu, mean, std = feats["cpu_total"], feats["cpu_roll_mean"], feats["cpu_roll_std"]
        delta, crossings = feats["cpu_delta"], feats["cpu_crossings"]
        jump_limit = max(self.baseline["abs_delta_p99"], 0.05)

        if crossings >= max(4, pf["cpu_crossings"]["p99"] + 2) and std > pf["cpu_roll_std"]["p99"]:
            return "fluctuations", (f"CPU crossed its rolling average {int(crossings)} times "
                                    f"(normal max {pf['cpu_crossings']['p99']:.0f}) and variability "
                                    f"{_fmt(std, 'cores')} exceeds the normal {_fmt(pf['cpu_roll_std']['p99'], 'cores')}")
        if delta > jump_limit and cpu - mean > std:
            return "spike", (f"CPU jumped {_signed(delta, 'cores')} in {config.DELTA_LAG} samples "
                             f"(normal max change {_fmt(jump_limit, 'cores')}) and is well above its recent average "
                             f"{_fmt(mean, 'cores')}")
        if mean > pf["cpu_roll_mean"]["p99"]:
            return "sustained", (f"average CPU over the window is {_fmt(mean, 'cores')}, above the normal "
                                 f"maximum {_fmt(pf['cpu_roll_mean']['p99'], 'cores')}, with little change "
                                 f"({_signed(delta, 'cores')})")
        if delta < -jump_limit or cpu < pf["cpu_total"]["p01"]:
            return "drop", (f"CPU fell {_signed(delta, 'cores')} to {_fmt(cpu, 'cores')} "
                            f"(normal low {_fmt(pf['cpu_total']['p01'], 'cores')})")
        return "point", "no single pattern rule fired; the combination of feature values is unusual"

    # ------------------------------------------------------------------ public API
    def explain(self, feats, timestamp=None, top_k=3):
        x = np.array([feats[f] for f in self.features], dtype=float)
        X_scaled = self.scaler.transform(x[None, :])
        score = float(self.model.score_samples(X_scaled)[0])
        decision = float(self.model.decision_function(X_scaled)[0])
        is_anomaly = decision < 0

        train_scores = self.baseline["train_scores_sorted"]
        # Share of training samples that are more normal (higher score) than this one.
        rarity = 1.0 - float(np.searchsorted(train_scores, score, side="right")) / len(train_scores)
        spread = max(self.baseline["median_decision_normal"], 1e-6)
        confidence = min(1.0, 0.5 + 0.5 * abs(decision) / spread)

        depth = float(self._path_length([score])[0])
        typical_depth = self.baseline["median_depth_normal"]

        contrib, method = self.contributions(X_scaled)
        contrib = contrib[0]
        positive = np.clip(contrib, 0, None)
        total_pos = positive.sum() or 1.0

        evidence = self._feature_evidence(x)
        for i, row in enumerate(evidence):
            row["contribution"] = float(contrib[i])
            row["share"] = float(positive[i] / total_pos)
        ranked = sorted(evidence, key=lambda r: r["contribution"], reverse=True)
        drivers = [r for r in ranked if r["contribution"] > 0][:top_k]

        groups = []
        for g, members in F.FEATURE_GROUPS.items():
            share = sum(r["share"] for r in evidence if r["feature"] in members)
            groups.append({"group": g, "description": F.GROUP_INFO[g], "share": float(share)})
        groups.sort(key=lambda d: d["share"], reverse=True)

        counterfactuals, joint = [], None
        if is_anomaly:
            for r in drivers:
                idx = self.features.index(r["feature"])
                cf = self._counterfactual(x, idx, r["typical"])
                counterfactuals.append({"feature": r["feature"], "description": r["description"],
                                        "unit": r["unit"], "current": r["value"], "normal_if": cf})
            joint = self._joint_counterfactual(x)
            pattern, pattern_reason = self._classify_pattern(feats)
        else:
            pattern, pattern_reason = "none", "sample is within the learned normal region"

        result = {
            "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(),
            "is_anomaly": bool(is_anomaly),
            "decision_score": decision,
            "raw_score": score,
            "confidence": confidence,
            "rarity": rarity,
            "path_length": depth,
            "typical_path_length": typical_depth,
            "attribution_method": method,
            "pattern": pattern,
            "pattern_reason": pattern_reason,
            "sample_pattern": pattern,
            "top_feature": drivers[0]["feature"] if drivers else None,
            "features": evidence,
            "evidence_groups": groups,
            "drivers": [r["feature"] for r in drivers],
            "counterfactuals": counterfactuals,
            "joint_counterfactual": joint,
        }
        result["summary"] = self.narrative(result)
        return result

    def narrative(self, r):
        lines = []
        verdict = "ANOMALY" if r["is_anomaly"] else "Normal"
        lines.append(f"{verdict} (confidence {r['confidence']:.0%}) at {r['timestamp']}: "
                     f"more unusual than {r['rarity']:.1%} of training samples.")
        lines.append(f"Isolation depth {r['path_length']:.1f} splits vs a typical {r['typical_path_length']:.1f} "
                     f"for normal data" + (" (isolated unusually fast)." if r["is_anomaly"] else "."))

        by_name = {f["feature"]: f for f in r["features"]}
        if r["is_anomaly"] and r["drivers"]:
            parts = []
            for name in r["drivers"]:
                f = by_name[name]
                parts.append(f"{f['description']} = {_fmt(f['value'], f['unit'])} "
                             f"(normal {_fmt(f['normal_low'])} to {_fmt(f['normal_high'], f['unit'])}, "
                             f"{f['share']:.0%} of the evidence)")
            lines.append("Main drivers: " + "; ".join(parts) + ".")
            g = [x for x in r["evidence_groups"] if x["share"] >= 0.05]
            lines.append("Evidence by kind: " + ", ".join(f"{x['description']} {x['share']:.0%}" for x in g) + ".")
            lines.append(f"Pattern: {r['pattern']}. {r['pattern_reason'][0].upper()}{r['pattern_reason'][1:]}.")
            cfs = [c for c in r["counterfactuals"] if c["normal_if"] is not None]
            joint = r.get("joint_counterfactual")
            if cfs:
                c = cfs[0]
                lines.append(f"Would be classed normal if {c['description']} were "
                             f"{_fmt(c['normal_if'], c['unit'])} instead of {_fmt(c['current'], c['unit'])} "
                             f"(other features unchanged).")
            elif joint:
                cpu_now = by_name["cpu_total"]["value"]
                lines.append(f"No single feature change makes this normal because several features are abnormal "
                             f"together; the whole sample would need to move {joint['fraction']:.0%} of the way back "
                             f"to typical (total CPU about {_fmt(joint['values']['cpu_total'], 'cores')} instead of "
                             f"{_fmt(cpu_now, 'cores')}).")
        elif not r["is_anomaly"]:
            outside = [f for f in r["features"] if f["status"] != "within normal range"]
            if outside:
                names = ", ".join(f"{f['description']} ({f['status']})" for f in outside)
                lines.append(f"Some features are outside their usual range ({names}), "
                             f"but not enough to cross the decision boundary.")
            else:
                lines.append("All features are within the ranges seen during training.")
        return " ".join(lines)

    def global_explanation(self, X_raw, max_rows=500):
        """Training-time summary: which features matter overall and what "normal" looks like."""
        X_raw = np.asarray(X_raw, dtype=float)
        idx = np.linspace(0, len(X_raw) - 1, min(max_rows, len(X_raw))).astype(int)
        X_scaled = self.scaler.transform(X_raw[idx])
        contrib, method = self.contributions(X_scaled)
        decisions = self.model.decision_function(X_scaled)
        mean_abs = np.abs(contrib).mean(axis=0)
        anomalous = decisions < 0
        anom_mean = contrib[anomalous].mean(axis=0) if anomalous.any() else np.zeros(len(self.features))
        total = mean_abs.sum() or 1.0

        importance = sorted(
            ({"feature": f, "description": F.describe(f),
              "mean_abs_contribution": float(mean_abs[i]), "share": float(mean_abs[i] / total),
              "mean_contribution_on_training_anomalies": float(anom_mean[i])}
             for i, f in enumerate(self.features)),
            key=lambda d: d["mean_abs_contribution"], reverse=True)

        normal_ranges = [{"feature": f, "description": F.describe(f), "unit": F.unit(f),
                          "low": st["p01"], "typical": st["p50"], "high": st["p99"],
                          "mean": st["mean"], "std": st["std"]}
                         for f, st in self.baseline["per_feature"].items()]
        return {
            "attribution_method": method,
            "n_train": self.baseline["n_train"],
            "contamination": float(getattr(self.model, "contamination", 0) or 0),
            "n_estimators": int(self.model.n_estimators),
            "typical_path_length": self.baseline["median_depth_normal"],
            "training_anomalies_in_sample": int(anomalous.sum()),
            "feature_importance": importance,
            "normal_ranges": normal_ranges,
        }


class IncidentTracker:
    """Groups consecutive anomalous samples into incidents and labels the incident by its overall shape.

    A single sample cannot tell a short spike from the first seconds of a long
    plateau, or from the first swing of a fluctuation. The tracker looks at
    every sample since the incident opened and labels it by how long CPU stayed
    high and how often it switched between high and low. The label is updated
    as the incident develops.
    """

    def __init__(self, explainer, close_after=15.0, spike_max=25.0):
        self.explainer = explainer
        self.close_after = close_after
        self.spike_max = spike_max
        self.next_id = 1
        self.current = None

    def _classify(self, sample_pattern, sample_reason):
        pf = self.explainer.baseline["per_feature"]["cpu_total"]
        t = np.array([s[0] for s in self.current["samples"]])
        cpu = np.array([s[1] for s in self.current["samples"]])
        peak = float(cpu.max())
        if peak <= pf["p99"]:
            if cpu.min() < pf["p01"]:
                return "drop", f"CPU fell to {_fmt(cpu.min(), 'cores')}, below the normal low {_fmt(pf['p01'], 'cores')}"
            return sample_pattern, sample_reason

        mid = pf["p50"] + 0.5 * (peak - pf["p50"])
        above = cpu > mid
        switches = int(np.sum(above[1:] != above[:-1]))
        dt = float(np.median(np.diff(t))) if len(t) > 1 else config.SAMPLE_INTERVAL
        high_time = float(above.sum() * dt)
        high_cycles = int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))

        if high_cycles >= 2 and switches >= 3:
            return "fluctuations", (f"over this incident CPU went above {_fmt(mid, 'cores')} {high_cycles} separate "
                                    f"times and dropped back in between ({switches} switches), peaking at "
                                    f"{_fmt(peak, 'cores')}")
        if high_time <= self.spike_max:
            ongoing = " so far; it becomes 'sustained' if it stays high" if above[-1] else ""
            return "spike", (f"CPU has been above {_fmt(mid, 'cores')} for only {high_time:.0f}s{ongoing} "
                             f"(spike limit {self.spike_max:.0f}s), peaking at {_fmt(peak, 'cores')}")
        return "sustained", (f"CPU stayed above {_fmt(mid, 'cores')} for {high_time:.0f}s without dropping back "
                             f"(longer than the {self.spike_max:.0f}s spike limit), peaking at {_fmt(peak, 'cores')}")

    def update(self, exp, epoch, cpu_total):
        exp["incident_id"] = None
        if self.current is not None and epoch - self.current["last_anomaly"] > self.close_after:
            self.current = None
        if exp["is_anomaly"] and self.current is None:
            self.current = {"id": self.next_id, "start": epoch, "last_anomaly": epoch, "samples": []}
            self.next_id += 1
        if self.current is None:
            return exp

        self.current["samples"].append((epoch, cpu_total))
        exp["incident_id"] = self.current["id"]
        exp["incident_start"] = self.current["start"]
        if exp["is_anomaly"]:
            self.current["last_anomaly"] = epoch
            pattern, reason = self._classify(exp["sample_pattern"], exp["pattern_reason"])
            exp["pattern"], exp["pattern_reason"] = pattern, reason
            exp["summary"] = self.explainer.narrative(exp)
        return exp
