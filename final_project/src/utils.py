import requests
import os
import time
from prometheus_api_client import PrometheusConnect
import json
import random

# Global variable to store the induced anomaly schedule
anomaly_schedule = []

def get_real_time_cpu_usage():
    prom_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
    prom = PrometheusConnect(url=prom_url, disable_ssl=True)
    cpu_query = 'sum(rate(node_cpu_seconds_total{mode!="idle"}[1m]))'
    
    try:
        result = prom.custom_query(query=cpu_query)
        if result:
            return float(result[0]['value'][1])
        return 0.0
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return 0.0

def send_to_prometheus(anomaly_score, is_anomaly):
    prometheus_pushgateway_url = os.getenv('PROMETHEUS_PUSHGATEWAY_URL', 'http://pushgateway:9091')
    job = "anomaly_detection"
    instance = "instance1"

    data = f"""
    # HELP cpu_anomaly_detection_score Anomaly detection score for CPU usage
    # TYPE cpu_anomaly_detection_score gauge
    cpu_anomaly_detection_score{{job="{job}", instance="{instance}"}} {anomaly_score}
    
    # HELP cpu_anomaly_detection_is_anomaly Anomaly detection flag for CPU usage
    # TYPE cpu_anomaly_detection_is_anomaly gauge
    cpu_anomaly_detection_is_anomaly{{job="{job}", instance="{instance}"}} {is_anomaly}
    """

    response = requests.post(f"{prometheus_pushgateway_url}/metrics/job/{job}/instance/{instance}", data=data)
    if response.status_code != 202:
        print("Failed to send metrics to Prometheus")

def is_anomaly_induced(timestamp):
    # Check if the timestamp matches any of the anomaly schedules
    global anomaly_schedule

    for anomaly in anomaly_schedule:
        # Inducing anomalies at specific times
        if anomaly[0] == "sustained" or anomaly[0] == "spike" or anomaly[0] == "fluctuations":
            # Logic for matching anomaly times with the timestamp
            # This can be refined with more precise time matching
            if timestamp in anomaly:
                return True
    
    return False

def schedule_anomalies(total_anomaly_duration):
    remaining_time = total_anomaly_duration
    anomaly_schedule = []

    while remaining_time > 0:
        anomaly_type = random.choice(["sustained", "spike", "fluctuations"])

        if anomaly_type == "sustained":
            duration = min(random.randint(30, 60), remaining_time)
            anomaly_schedule.append(("sustained", duration))
            remaining_time -= duration

        elif anomaly_type == "spike":
            duration = min(random.randint(10, 20), remaining_time)
            anomaly_schedule.append(("spike", duration))
            remaining_time -= duration

        elif anomaly_type == "fluctuations":
            high_duration = random.randint(5, 10)
            low_duration = random.randint(5, 10)
            cycles = min(remaining_time // (high_duration + low_duration), random.randint(1, 3))
            duration = cycles * (high_duration + low_duration)
            if duration > 0:
                anomaly_schedule.append(("fluctuations", cycles, high_duration, low_duration))
                remaining_time -= duration

    # Store anomaly schedule globally for reference
    return anomaly_schedule

