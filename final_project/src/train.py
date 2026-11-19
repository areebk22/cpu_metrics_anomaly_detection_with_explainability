"""Collect normal CPU behaviour, train the IsolationForest, and save it with its explainer baseline."""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config
import features as F
from explain import AnomalyExplainer
from utils import get_cpu_snapshot, is_anomaly_induced, load_schedule, now_str


def collect_live(duration, interval=None):
    """Sample raw CPU snapshots from Prometheus for `duration` seconds."""
    interval = interval or config.SAMPLE_INTERVAL
    print(f"Collecting {duration}s of live CPU data (one sample every {interval}s)...")
    rows, end = [], time.time() + duration
    while time.time() < end:
        epoch = time.time()
        snap = get_cpu_snapshot()
        if snap is not None:
            snap["timestamp"] = now_str(epoch)
            snap["epoch"] = epoch
            snap["induced_anomaly"] = is_anomaly_induced(epoch, load_schedule()) or ""
            rows.append(snap)
        time.sleep(interval)
    return pd.DataFrame(rows)


def load_historical(path):
    df = pd.read_csv(path)
    if "cpu_total" not in df.columns and "cpu_usage" in df.columns:
        print("Legacy CSV (cpu_usage only): per-mode CPU and load are unavailable and set to 0.")
        df = df.rename(columns={"cpu_usage": "cpu_total"})
    for col in F.RAW_FIELDS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def fit(raw_df, contamination=None, save=True):
    """Train on a DataFrame of raw snapshots. Returns the fitted AnomalyExplainer."""
    contamination = contamination if contamination is not None else config.CONTAMINATION
    feat_df = F.build_feature_frame(raw_df)
    if len(feat_df) < 50:
        raise ValueError(f"Only {len(feat_df)} usable samples after warm-up; collect more training data.")
    X = feat_df[F.FEATURES].to_numpy(dtype=float)
    if np.allclose(X[:, 0], X[0, 0]):
        print("WARNING: CPU is constant in the training data. Is Prometheus scraping node_exporter?")

    scaler = StandardScaler().fit(X)
    model = IsolationForest(contamination=contamination, n_estimators=200, max_samples="auto", random_state=42)
    model.fit(scaler.transform(X))

    explainer = AnomalyExplainer.fit(model, scaler, X)
    global_report = explainer.global_explanation(X)
    global_report["trained_at"] = now_str()

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        bundle = explainer.to_bundle()
        bundle["version"] = config.MODEL_FORMAT_VERSION
        bundle["features"] = list(F.FEATURES)
        joblib.dump(bundle, config.MODEL_PATH)
        with open(config.GLOBAL_REPORT_PATH, "w") as f:
            json.dump(global_report, f, indent=2)
        print(f"Model saved to {config.MODEL_PATH}")
        print(f"Global explanation saved to {config.GLOBAL_REPORT_PATH}")

    print("Global feature importance:")
    for item in global_report["feature_importance"]:
        print(f"  {item['feature']:<14} {item['share']:6.1%}  {item['description']}")
    return explainer


def load_explainer():
    """Load the saved model as an AnomalyExplainer, or None if it is missing or in the old (model, scaler) format."""
    if not os.path.exists(config.MODEL_PATH):
        return None
    bundle = joblib.load(config.MODEL_PATH)
    if not isinstance(bundle, dict) or bundle.get("version") != config.MODEL_FORMAT_VERSION \
            or bundle.get("features") != list(F.FEATURES):
        print("Saved model is from an older version without explainability metadata.")
        return None
    return AnomalyExplainer.from_bundle(bundle)


def train_model(duration=None, data_path=None, force=False):
    if not force and load_explainer() is not None:
        print(f"Model already exists at {config.MODEL_PATH}. Skipping training.")
        return load_explainer()

    if data_path and os.path.exists(data_path):
        print(f"Training on historical data from {data_path}")
        raw = load_historical(data_path)
    else:
        raw = collect_live(duration if duration is not None else config.TRAIN_DURATION)
        raw.to_csv(config.TRAINING_DATA_PATH, index=False)
        print(f"Raw training data saved to {config.TRAINING_DATA_PATH}")
    return fit(raw)


if __name__ == "__main__":
    train_model(force=True)
