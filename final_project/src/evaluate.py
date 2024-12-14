"""Score detections against the induced-anomaly schedule, including whether the explained pattern was right."""
import json
from collections import Counter

import config
from utils import load_schedule


def load_explanations(path=None):
    path = path or config.EXPLANATIONS_PATH
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return records


def _in_event(epoch, ev, tol):
    return ev["start"] <= epoch <= ev["end"] + tol


def evaluate(records=None, schedule=None, tolerance=None, save=True):
    records = load_explanations() if records is None else records
    schedule = load_schedule() if schedule is None else schedule
    tol = config.EVAL_TOLERANCE if tolerance is None else tolerance
    if not records:
        print("No explanations logged yet.")
        return None

    # Only score the time range the detector was running.
    t0, t1 = records[0]["epoch"], records[-1]["epoch"]
    events = [ev for ev in schedule if ev["end"] >= t0 and ev["start"] <= t1]

    flagged = [r for r in records if r["is_anomaly"]]
    true_flags = [r for r in flagged if any(_in_event(r["epoch"], ev, tol) for ev in events)]
    in_window = [r for r in records if any(ev["start"] <= r["epoch"] <= ev["end"] for ev in events)]
    caught_in_window = [r for r in in_window if r["is_anomaly"]]

    per_event = []
    for ev in events:
        hits = [r for r in flagged if _in_event(r["epoch"], ev, tol)]
        patterns = Counter(r["pattern"] for r in hits)
        # The incident tracker refines its label as the incident develops, so the last
        # flagged sample of the event carries the final explanation.
        majority = hits[-1]["pattern"] if hits else None
        per_event.append({
            "type": ev["type"],
            "start": ev.get("start_str", ev["start"]),
            "duration_s": round(ev["end"] - ev["start"], 1),
            "detected": bool(hits),
            "latency_s": round(min(r["epoch"] for r in hits) - ev["start"], 1) if hits else None,
            "flagged_samples": len(hits),
            "explained_pattern": majority,
            "pattern_votes": dict(patterns),
            "pattern_correct": majority == ev["type"],
            "top_features": Counter(r.get("top_feature") for r in hits).most_common(3),
        })

    precision = len(true_flags) / len(flagged) if flagged else None
    sample_recall = len(caught_in_window) / len(in_window) if in_window else None
    detected = [e for e in per_event if e["detected"]]
    result = {
        "samples": len(records),
        "flagged": len(flagged),
        "false_positives": len(flagged) - len(true_flags),
        "events": len(events),
        "events_detected": len(detected),
        "event_recall": len(detected) / len(events) if events else None,
        "sample_precision": precision,
        "sample_recall": sample_recall,
        "pattern_accuracy": (sum(e["pattern_correct"] for e in detected) / len(detected)) if detected else None,
        "mean_latency_s": (sum(e["latency_s"] for e in detected) / len(detected)) if detected else None,
        "tolerance_s": tol,
        "per_event": per_event,
    }
    if save:
        with open(config.EVALUATION_PATH, "w") as f:
            json.dump(result, f, indent=2)
    _print(result)
    return result


def _pct(v):
    return "n/a" if v is None else f"{v:.1%}"


def _print(r):
    print(f"Samples scored: {r['samples']}, flagged: {r['flagged']} ({r['false_positives']} false positives)")
    print(f"Events detected: {r['events_detected']}/{r['events']} (recall {_pct(r['event_recall'])}), "
          f"sample precision {_pct(r['sample_precision'])}, sample recall {_pct(r['sample_recall'])}")
    print(f"Explained pattern matched the induced type for {_pct(r['pattern_accuracy'])} of detected events")
    for e in r["per_event"]:
        status = f"detected after {e['latency_s']}s, explained as {e['explained_pattern']}" if e["detected"] else "MISSED"
        print(f"  {e['start']}  {e['type']:<12} {e['duration_s']:>5}s  {status}")


if __name__ == "__main__":
    evaluate()
