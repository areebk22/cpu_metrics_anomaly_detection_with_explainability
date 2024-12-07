import subprocess
import random
import os
import time
from utils import schedule_anomalies

# Get the TRAIN_DURATION from the environment variable (default: 21600 seconds = 6 hours)
train_duration = int(os.getenv('TRAIN_DURATION', 300))

# Calculate total anomaly duration as 1% of the training duration
total_anomaly_duration = int(train_duration * 0.01)

# Define anomaly functions
def induce_sustained_high_cpu(duration):
    print(f"Inducing sustained high CPU usage for {duration} seconds...")
    subprocess.run(["docker", "run", "--rm", "-it", "polinux/stress", "stress", "--cpu", "8", f"--timeout={duration}s"])

def induce_cpu_spike(duration):
    print(f"Inducing CPU spike for {duration} seconds...")
    subprocess.run(["docker", "run", "--rm", "-it", "polinux/stress", "stress", "--cpu", "8", f"--timeout={duration}s"])

def induce_cpu_fluctuations(cycles, high_duration, low_duration):
    print(f"Inducing CPU fluctuations: {cycles} cycles of {high_duration}s high load and {low_duration}s low load.")
    for _ in range(cycles):
        subprocess.run(["docker", "run", "--rm", "-it", "polinux/stress", "stress", "--cpu", "8", f"--timeout={high_duration}s"])
        time.sleep(low_duration)

# Generate a schedule of anomalies
anomaly_schedule = schedule_anomalies(total_anomaly_duration)

# Example usage: Execute the anomaly schedule
for anomaly in anomaly_schedule:
    if anomaly[0] == "sustained":
        induce_sustained_high_cpu(anomaly[1])
    elif anomaly[0] == "spike":
        induce_cpu_spike(anomaly[1])
    elif anomaly[0] == "fluctuations":
        induce_cpu_fluctuations(anomaly[1], anomaly[2], anomaly[3])
