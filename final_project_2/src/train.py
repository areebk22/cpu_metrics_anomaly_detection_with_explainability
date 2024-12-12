import time
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils import get_real_time_cpu_usage
from induce_anomalies import generate_timeline, induce_anomalies

def train_model(
    duration=28800, 
    data_path=None, 
    save_path='/usr/src/app/data/realtime_data.csv', 
    enable_anomalies=False, 
    incremental=False
):
    """
    Trains an Isolation Forest model using either historical or real-time CPU usage data.

    Args:
        duration (int): Duration (in seconds) for collecting real-time data.
        data_path (str): Path to historical data file (if any).
        save_path (str): Path to save the collected real-time data.
        enable_anomalies (bool): If True, induces anomalies during real-time data collection.
        incremental (bool): If True, adds to the existing model instead of retraining from scratch.
    """
    model_path = '/usr/src/app/data/trained_model.pkl'

    # Skip training if a model already exists and incremental is disabled
    if os.path.exists(model_path) and not incremental:
        print(f"Model already exists at {model_path}. Skipping training.")
        return

    print("Starting model training...")
    data = []

    if data_path and os.path.exists(data_path):
        # Use historical data for training
        print(f"Training on historical data from {data_path}")
        data = pd.read_csv(data_path).values.tolist()
    else:
        # Collect real-time data
        print("Collecting real-time data...")
        timeline = generate_timeline(duration, anomaly_percentage=0.02 if enable_anomalies else 0.0)
        for state, anomaly_type, duration in timeline:
            for _ in range(duration):
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                cpu_usage = get_real_time_cpu_usage()
                if state == "anomaly":
                    cpu_usage = induce_anomalies(cpu_usage, anomaly_type)
                    label = 1
                else:
                    label = 0
                data.append([timestamp, cpu_usage, label])
                time.sleep(1)  # Collect data every 1 seconds

        # Save real-time data to file
        pd.DataFrame(data, columns=['timestamp', 'cpu_usage', 'label']).to_csv(save_path, index=False)
        print(f"Real-time data saved to {save_path}")

    # Load the data for training
    df = pd.DataFrame(data, columns=['timestamp', 'cpu_usage', 'label']) if data else pd.read_csv(save_path)
    X = df['cpu_usage'].values.reshape(-1, 1)

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest model
    model = IsolationForest(contamination=0.01, n_estimators=200, max_samples='auto', random_state=42)
    model.fit(X_scaled)

    # Save the trained model
    joblib.dump((model, scaler), model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()
