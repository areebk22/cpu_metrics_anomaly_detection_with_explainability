import time
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
from utils import get_real_time_cpu_usage, is_anomaly_induced

def train_model(duration=300, data_path=None, save_path='/usr/src/app/data/realtime_data.csv', incremental=False):
    model_path = '/usr/src/app/data/trained_model.pkl'

    if os.path.exists(model_path) and not incremental:
        print(f"Model already exists at {model_path}. Skipping training.")
        return

    print("Starting model training...")

    data = []
    if data_path and os.path.exists(data_path):
        print(f"Training on historical data from {data_path}")
        data = pd.read_csv(data_path).values.tolist()
    elif not data_path or not os.path.exists(data_path):
        print("Training on real-time data")
        end_time = time.time() + duration
        while time.time() < end_time:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            cpu_usage = get_real_time_cpu_usage()

            # Check if anomaly is induced
            is_anomaly = is_anomaly_induced(timestamp)  # Function to be added in utils.py

            data.append([timestamp, cpu_usage, is_anomaly])
            time.sleep(0.5)  # Collect data every 0.5 seconds

        pd.DataFrame(data, columns=['timestamp', 'cpu_usage', 'is_anomaly']).to_csv(save_path, index=False)
        print(f"Real-time data saved to {save_path}")

    X = np.array([d[1] for d in data]).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.01, n_estimators=200, max_samples='auto', random_state=42)
    model.fit(X_scaled)

    joblib.dump((model, scaler), model_path)
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
