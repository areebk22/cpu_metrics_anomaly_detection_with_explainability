"""Live detection loop: score every sample, explain it, publish metrics, and log the reasoning."""
import json
import time

import config
import features as F
from explain import IncidentTracker
from train import load_explainer
from utils import get_cpu_snapshot, is_anomaly_induced, load_schedule, now_str, push_metrics

PATTERNS = ["spike", "sustained", "fluctuations", "drop", "point", "none"]


def explanation_metrics(exp):
    m = [
        ("cpu_anomaly_detection_score", "IsolationForest decision score (negative = anomaly)", {}, exp["decision_score"]),
        ("cpu_anomaly_detection_is_anomaly", "1 if the sample is classed as an anomaly", {}, int(exp["is_anomaly"])),
        ("cpu_anomaly_confidence", "Confidence in the verdict, 0.5 at the decision boundary", {}, exp["confidence"]),
        ("cpu_anomaly_rarity", "Share of training samples that look more normal than this one", {}, exp["rarity"]),
        ("cpu_anomaly_path_length", "Average isolation depth across the trees", {}, exp["path_length"]),
        ("cpu_anomaly_typical_path_length", "Median isolation depth of normal training samples", {}, exp["typical_path_length"]),
    ]
    for f in exp["features"]:
        lbl = {"feature": f["feature"]}
        m.append(("cpu_anomaly_feature_value", "Current feature value", lbl, f["value"]))
        m.append(("cpu_anomaly_feature_contribution", "Push towards anomaly (positive) or normal (negative)", lbl, f["contribution"]))
        m.append(("cpu_anomaly_feature_zscore", "Feature z-score against the training baseline", lbl, f["z_score"]))
    for p in PATTERNS:
        m.append(("cpu_anomaly_pattern", "1 for the pattern of the current sample", {"pattern": p}, int(exp["pattern"] == p)))
    return m


def compact(exp):
    """Smaller log record for normal samples."""
    keep = ["timestamp", "epoch", "is_anomaly", "decision_score", "confidence", "rarity",
            "path_length", "pattern", "top_feature", "incident_id", "induced", "summary"]
    rec = {k: exp.get(k) for k in keep}
    rec["values"] = {f["feature"]: round(f["value"], 4) for f in exp["features"]}
    rec["contributions"] = {f["feature"]: round(f["contribution"], 5) for f in exp["features"]}
    return rec


def start_detection(explainer=None, max_samples=None, push=True):
    explainer = explainer or load_explainer()
    if explainer is None:
        print("Model not found. Train it first.")
        return

    builder = F.FeatureBuilder()
    tracker = IncidentTracker(explainer)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Detection running. Explanations are written to {config.EXPLANATIONS_PATH}")
    n = 0
    with open(config.EXPLANATIONS_PATH, "a") as log:
        while max_samples is None or n < max_samples:
            epoch = time.time()
            snap = get_cpu_snapshot()
            feats = builder.update(snap) if snap is not None else None
            if feats is not None:
                exp = explainer.explain(feats, timestamp=now_str(epoch))
                exp["epoch"] = epoch
                tracker.update(exp, epoch, feats["cpu_total"])
                exp["induced"] = is_anomaly_induced(epoch, load_schedule())
                if push:
                    push_metrics(explanation_metrics(exp))
                log.write(json.dumps(exp if exp["is_anomaly"] else compact(exp)) + "\n")
                log.flush()
                if exp["is_anomaly"]:
                    print(exp["summary"], flush=True)
                n += 1
            time.sleep(max(0.0, config.SAMPLE_INTERVAL - (time.time() - epoch)))


if __name__ == "__main__":
    start_detection()
