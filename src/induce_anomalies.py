"""Induce labelled CPU anomalies after the model is trained.

Load is generated with busy-loop worker processes inside this container, so no
Docker CLI or TTY is needed. /proc/stat is not namespaced, so node_exporter
sees the extra load. Each anomaly is written to the shared schedule file with
its real start and end times, which gives the detector and the evaluation
script ground-truth labels.
"""
import multiprocessing as mp
import os
import random
import time

import config
from utils import now_str, record_anomaly, schedule_anomalies


def _burn(stop_at):
    while time.time() < stop_at:
        pass


def burn_cpu(duration, load_fraction=1.0):
    workers = max(1, int(round((os.cpu_count() or 1) * load_fraction)))
    stop_at = time.time() + duration
    procs = [mp.Process(target=_burn, args=(stop_at,)) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def run_anomaly(plan):
    start = time.time()
    if plan["type"] == "fluctuations":
        print(f"[{now_str()}] Inducing fluctuations: {plan['cycles']} x ({plan['high']}s high / {plan['low']}s low)", flush=True)
        for _ in range(plan["cycles"]):
            burn_cpu(plan["high"], plan["load_fraction"])
            time.sleep(plan["low"])
    else:
        print(f"[{now_str()}] Inducing {plan['type']} load for {plan['duration']}s "
              f"on {plan['load_fraction']:.0%} of cores", flush=True)
        burn_cpu(plan["duration"], plan["load_fraction"])
    end = time.time()
    record_anomaly({**plan, "start": start, "end": end,
                    "start_str": now_str(start), "end_str": now_str(end)})


def wait_for_model():
    print(f"Waiting for the trained model at {config.MODEL_PATH} ...", flush=True)
    while not config.MODEL_PATH.exists():
        time.sleep(5)
    print(f"Model found. Waiting {config.INDUCTION_WARMUP}s so the detector sees normal load first.", flush=True)
    time.sleep(config.INDUCTION_WARMUP)


def main():
    if config.SCHEDULE_PATH.exists():
        config.SCHEDULE_PATH.unlink()
    wait_for_model()
    plan = schedule_anomalies()
    print("Anomaly plan: " + ", ".join(p["type"] for p in plan), flush=True)
    for p in plan:
        run_anomaly(p)
        gap = config.ANOMALY_GAP + random.randint(0, 20)
        time.sleep(gap)
    print(f"All anomalies induced. Labels are in {config.SCHEDULE_PATH}", flush=True)


if __name__ == "__main__":
    main()
