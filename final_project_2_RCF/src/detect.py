import os
import time
import joblib
import numpy as np
import pandas as pd
from utils import get_real_time_cpu_usage, send_to_prometheus

MODELS = {
    "dbscan": "trained_dbscan_model.pkl",
    "lof": "trained_lof_model.pkl",
    "ocsvm": "trained_ocsvm_model.pkl",
    "isolation_forest": "trained_isolation_forest_model.pkl"
}

def load_model(model_name):
    model_path = f"/usr/src/app/data/{MODELS[model_name]}"
    if not os.path.exists(model_path):
        print(f"Model for {model_name} not found. Exiting.")
        exit()
    return joblib.load(model_path)

def start_detection(model_name="ocsvm", save_detection_data=False, detection_data_path='/usr/src/app/data/detection_data.csv'):
    model, scaler = load_model(model_name)
    detection_data = []

    while True:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        cpu_usage = get_real_time_cpu_usage()
        X = np.array([[cpu_usage]])
        X_scaled = scaler.transform(X)

        if model_name == "dbscan":
            is_anomaly = int(model.fit_predict(X_scaled)[0] == -1)
            anomaly_score = -1  # DBSCAN doesn't provide scores
        elif model_name == "lof":
            is_anomaly = int(model.predict(X_scaled)[0] == -1)
            anomaly_score = model.negative_outlier_factor_[0]
        elif model_name == "isolation_forest":
            is_anomaly = int(model.predict(X_scaled)[0] == -1)
            anomaly_score = model.decision_function(X_scaled)[0]
        else:  # One-Class SVM
            is_anomaly = int(model.predict(X_scaled)[0] == -1)
            anomaly_score = model.decision_function(X_scaled)[0]

        send_to_prometheus(anomaly_score, is_anomaly)

        if save_detection_data:
            detection_data.append([timestamp, cpu_usage, is_anomaly, anomaly_score])

        time.sleep(1)

        if save_detection_data and len(detection_data) >= 60:
            pd.DataFrame(detection_data, columns=['timestamp', 'cpu_usage', 'is_anomaly', 'confidence_score']).to_csv(detection_data_path, mode='a', header=not os.path.exists(detection_data_path), index=False)
            detection_data = []

if __name__ == "__main__":
    start_detection(model_name="ocsvm")