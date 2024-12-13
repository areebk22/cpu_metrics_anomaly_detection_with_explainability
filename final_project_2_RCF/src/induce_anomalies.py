import time
import random

def induce_anomalies(cpu_usage, anomaly_type):
    """
    Alters CPU usage data to simulate specific anomalies.

    Args:
        cpu_usage (float): The real-time CPU usage data point.
        anomaly_type (str): The type of anomaly ("sustained", "spike", "fluctuations").

    Returns:
        float: The modified CPU usage value to simulate an anomaly.
    """
    if anomaly_type == "sustained":
        return random.uniform(4, 10)  # Sustained high CPU usage
    elif anomaly_type == "spike":
        return random.uniform(8, 11)  # Sharp spikes in usage
    return cpu_usage

def generate_timeline(total_runtime, anomaly_percentage):
    """
    Generates a randomized timeline of anomaly and normal periods.

    Args:
        total_runtime (int): Total duration (in seconds).
        anomaly_percentage (float): Percentage of runtime for anomalies.

    Returns:
        list: A list of tuples indicating periods of normal and anomaly data.
    """
    total_anomaly_time = int(total_runtime * anomaly_percentage)
    normal_time = total_runtime - total_anomaly_time
    timeline = []

    while total_anomaly_time > 0 or normal_time > 0:
        if total_anomaly_time > 0:
            anomaly_duration = min(random.randint(1, 30), total_anomaly_time)
            anomaly_type = random.choice(["sustained", "spike"])
            timeline.append(("anomaly", anomaly_type, anomaly_duration))
            total_anomaly_time -= anomaly_duration
        if normal_time > 0:
            normal_duration = min(random.randint(1, 30), normal_time)
            timeline.append(("normal", None, normal_duration))
            normal_time -= normal_duration

    random.shuffle(timeline)
    return timeline
