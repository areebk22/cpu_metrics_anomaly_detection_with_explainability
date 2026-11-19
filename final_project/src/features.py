"""Turns raw CPU snapshots into the features the model sees.

The same FeatureBuilder is used for training and for live detection, so both
see identical features. A single raw CPU number can only say "the value is
unusual"; the rolling features let the explanation say *how* it is unusual
(a sudden jump, a long plateau, or rapid flapping).
"""
from collections import deque

import numpy as np
import pandas as pd

import config

RAW_FIELDS = ["cpu_total", "cpu_user", "cpu_system", "cpu_iowait", "load1"]

FEATURES = [
    "cpu_total",
    "cpu_user",
    "cpu_system",
    "cpu_iowait",
    "load1",
    "cpu_roll_mean",
    "cpu_roll_std",
    "cpu_delta",
    "cpu_crossings",
]

# Human-readable descriptions used in explanations.
FEATURE_INFO = {
    "cpu_total": ("total CPU busy", "cores"),
    "cpu_user": ("user-mode CPU", "cores"),
    "cpu_system": ("kernel-mode CPU", "cores"),
    "cpu_iowait": ("CPU waiting on I/O", "cores"),
    "load1": ("1-minute load average", ""),
    "cpu_roll_mean": (f"average CPU over the last {config.ROLL_WINDOW} samples", "cores"),
    "cpu_roll_std": (f"CPU variability over the last {config.ROLL_WINDOW} samples", "cores"),
    "cpu_delta": (f"CPU change over the last {config.DELTA_LAG} samples", "cores"),
    "cpu_crossings": ("times CPU crossed its rolling average", ""),
}


# Features grouped by the kind of evidence they carry, so explanations can say
# "mostly a level problem" vs "mostly a variability problem".
FEATURE_GROUPS = {
    "level": ["cpu_total", "cpu_user", "cpu_system", "cpu_roll_mean", "load1"],
    "change": ["cpu_delta"],
    "variability": ["cpu_roll_std", "cpu_crossings"],
    "io": ["cpu_iowait"],
}
GROUP_INFO = {
    "level": "how high CPU is",
    "change": "how fast CPU changed",
    "variability": "how much CPU is swinging",
    "io": "I/O wait",
}


def describe(feature):
    return FEATURE_INFO.get(feature, (feature, ""))[0]


def unit(feature):
    return FEATURE_INFO.get(feature, (feature, ""))[1]


def _count_crossings(values, mean, std):
    """Count sign changes of (value - mean), ignoring wiggles inside a dead band."""
    band = max(0.05, 0.5 * std)
    signs = []
    for v in values:
        d = v - mean
        if d > band:
            signs.append(1)
        elif d < -band:
            signs.append(-1)
    return int(sum(1 for a, b in zip(signs, signs[1:]) if a != b))


class FeatureBuilder:
    """Keeps a rolling window of raw snapshots and emits one feature row per snapshot."""

    def __init__(self, window=None, lag=None):
        self.window = window or config.ROLL_WINDOW
        self.lag = lag or config.DELTA_LAG
        self.history = deque(maxlen=self.window)

    @property
    def ready(self):
        return len(self.history) >= self.window

    def update(self, snapshot):
        """Add a raw snapshot (dict with RAW_FIELDS). Returns a feature dict, or None while warming up."""
        snap = {k: float(snapshot.get(k, 0.0) or 0.0) for k in RAW_FIELDS}
        self.history.append(snap)
        if not self.ready:
            return None

        cpu = np.array([s["cpu_total"] for s in self.history])
        mean = float(cpu.mean())
        std = float(cpu.std())
        feats = dict(snap)
        feats["cpu_roll_mean"] = mean
        feats["cpu_roll_std"] = std
        feats["cpu_delta"] = float(cpu[-1] - cpu[-1 - self.lag])
        feats["cpu_crossings"] = float(_count_crossings(cpu, mean, std))
        return feats


def build_feature_frame(raw_df):
    """Run a DataFrame of raw snapshots through a FeatureBuilder (rows during warm-up are dropped)."""
    builder = FeatureBuilder()
    rows = []
    for rec in raw_df.to_dict("records"):
        feats = builder.update(rec)
        if feats is not None:
            feats["timestamp"] = rec.get("timestamp")
            rows.append(feats)
    return pd.DataFrame(rows, columns=["timestamp"] + FEATURES)


def to_vector(feats):
    return np.array([[feats[f] for f in FEATURES]], dtype=float)
