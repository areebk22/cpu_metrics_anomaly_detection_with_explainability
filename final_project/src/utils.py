"""Prometheus I/O, Pushgateway publishing, and the shared anomaly schedule."""
import json
import os
import random
import time

import requests

import config

_session = requests.Session()


# --------------------------------------------------------------------------- Prometheus
def _query(expr):
    resp = _session.get(f"{config.PROMETHEUS_URL}/api/v1/query", params={"query": expr}, timeout=5)
    resp.raise_for_status()
    return resp.json()["data"]["result"]


def get_cpu_count():
    try:
        result = _query('count(node_cpu_seconds_total{mode="idle"})')
        return int(float(result[0]["value"][1])) if result else None
    except Exception as e:
        print(f"Could not read CPU count from Prometheus: {e}")
        return None


def get_cpu_snapshot():
    """One raw sample: busy cores in total and per mode, plus load1. Returns None if Prometheus is unreachable."""
    w = config.RATE_WINDOW
    try:
        total = _query(f'sum(rate(node_cpu_seconds_total{{mode!="idle"}}[{w}]))')
        modes = _query(f'sum by (mode) (rate(node_cpu_seconds_total{{mode=~"user|system|iowait"}}[{w}]))')
        load = _query("node_load1")
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return None
    if not total:
        return None

    snap = {"cpu_total": float(total[0]["value"][1]), "cpu_user": 0.0, "cpu_system": 0.0, "cpu_iowait": 0.0}
    for series in modes:
        snap[f"cpu_{series['metric']['mode']}"] = float(series["value"][1])
    snap["load1"] = float(load[0]["value"][1]) if load else 0.0
    return snap


def get_real_time_cpu_usage():
    """Backwards-compatible helper: total busy cores only."""
    snap = get_cpu_snapshot()
    return snap["cpu_total"] if snap else 0.0


# --------------------------------------------------------------------------- Pushgateway
def _escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def push_metrics(metrics, job="anomaly_detection", instance="instance1"):
    """Replace this job's metric group on the Pushgateway.

    `metrics` is a list of (name, help, labels_dict, value). The job and
    instance labels come from the URL, so they are not repeated in the body.
    """
    lines, seen = [], set()
    for name, help_text, labels, value in metrics:
        if name not in seen:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            seen.add(name)
        label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in (labels or {}).items())
        lines.append(f"{name}{{{label_str}}} {float(value)}" if label_str else f"{name} {float(value)}")
    body = "\n".join(lines) + "\n"
    try:
        resp = _session.put(f"{config.PUSHGATEWAY_URL}/metrics/job/{job}/instance/{instance}", data=body, timeout=5)
        if resp.status_code not in (200, 202):
            print(f"Pushgateway rejected metrics ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"Failed to push metrics: {e}")


def send_to_prometheus(anomaly_score, is_anomaly):
    """Backwards-compatible helper used by the original detector."""
    push_metrics([
        ("cpu_anomaly_detection_score", "Anomaly detection score for CPU usage", {}, anomaly_score),
        ("cpu_anomaly_detection_is_anomaly", "Anomaly detection flag for CPU usage", {}, is_anomaly),
    ])


# --------------------------------------------------------------------------- anomaly schedule
def schedule_anomalies(count=None):
    """Random plan of anomalies to induce. Each entry is a dict describing one anomaly."""
    count = count or config.ANOMALY_COUNT
    plan = []
    for _ in range(count):
        kind = random.choice(["sustained", "spike", "fluctuations"])
        if kind == "sustained":
            plan.append({"type": "sustained", "duration": random.randint(45, 90), "load_fraction": 0.6})
        elif kind == "spike":
            plan.append({"type": "spike", "duration": random.randint(10, 20), "load_fraction": 1.0})
        else:
            plan.append({"type": "fluctuations", "cycles": random.randint(3, 4),
                         "high": random.randint(6, 10), "low": random.randint(6, 10), "load_fraction": 1.0})
    return plan


def load_schedule(path=None):
    path = path or config.SCHEDULE_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def record_anomaly(event, path=None):
    """Append one induced anomaly (with real start/end epoch times) to the shared schedule file."""
    path = path or config.SCHEDULE_PATH
    events = load_schedule(path)
    events.append(event)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(events, f, indent=2)
    os.replace(tmp, path)


def is_anomaly_induced(epoch, schedule=None, tolerance=0):
    """Return the anomaly type active at `epoch` (seconds), or None."""
    schedule = load_schedule() if schedule is None else schedule
    for ev in schedule:
        if ev["start"] <= epoch <= ev["end"] + tolerance:
            return ev["type"]
    return None


def now_str(epoch=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch if epoch is not None else time.time()))
