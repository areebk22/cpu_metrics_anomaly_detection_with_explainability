"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path


def _env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

MODEL_PATH = DATA_DIR / "trained_model.pkl"
TRAINING_DATA_PATH = DATA_DIR / "realtime_data.csv"
GLOBAL_REPORT_PATH = DATA_DIR / "global_explanation.json"
EXPLANATIONS_PATH = DATA_DIR / "explanations.jsonl"
SCHEDULE_PATH = DATA_DIR / "anomaly_schedule.json"
EVALUATION_PATH = DATA_DIR / "evaluation.json"
HTML_REPORT_PATH = DATA_DIR / "explanation_report.html"

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "http://pushgateway:9091")

# Rate window for the Prometheus CPU queries. Must cover at least two scrapes
# (prometheus.yml scrapes node_exporter every 5s).
RATE_WINDOW = os.getenv("RATE_WINDOW", "20s")

TRAIN_DURATION = int(os.getenv("TRAIN_DURATION", 300))   # seconds of live data to train on
SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", 1.0))  # seconds between samples
CONTAMINATION = float(os.getenv("CONTAMINATION", 0.01))

HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH") or None
HISTORICAL_TRAINING = _env_bool("HISTORICAL_TRAINING")
# IsolationForest cannot be updated in place, so "incremental" means: retrain even if a model exists.
FORCE_RETRAIN = _env_bool("INCREMENTAL_TRAINING") or _env_bool("FORCE_RETRAIN")

# Rolling-window feature settings (in samples).
ROLL_WINDOW = int(os.getenv("ROLL_WINDOW", 30))
DELTA_LAG = int(os.getenv("DELTA_LAG", 5))

# Anomaly induction.
ANOMALY_COUNT = int(os.getenv("ANOMALY_COUNT", 6))
ANOMALY_GAP = int(os.getenv("ANOMALY_GAP", 60))           # seconds of normal load between anomalies
INDUCTION_WARMUP = int(os.getenv("INDUCTION_WARMUP", 60))  # seconds to wait after the model is ready

# A detection counts as matching an induced anomaly if it lands within this
# many seconds after the anomaly ends (rolling features lag behind the load).
EVAL_TOLERANCE = int(os.getenv("EVAL_TOLERANCE", 30))

MODEL_FORMAT_VERSION = 2
