import os
import time
import joblib
import numpy as np
import pandas as pd
from utils import get_real_time_cpu_usage, send_to_prometheus

def start_detection(save_detection_data=False, detection_data_path='/usr/src/app/data/detection_data.csv'):
    model_path = '/usr/src/app/data/trained_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Exiting.")
        return
    
    model, scaler = joblib.load(model_path)
    detection_data = []

    while True:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        cpu_usage = get_real_time_cpu_usage()
        X = np.array([[cpu_usage]])
        X_scaled = scaler.transform(X)
        
        is_anomaly = int(model.predict(X_scaled)[0] == -1)
        anomaly_score = model.decision_function(X_scaled)

        send_to_prometheus(anomaly_score[0], is_anomaly)

        if save_detection_data:
            detection_data.append([timestamp, cpu_usage, is_anomaly, anomaly_score])
        
        time.sleep(1)

        if save_detection_data and len(detection_data) >= 60:
            pd.DataFrame(detection_data, columns=['timestamp', 'cpu_usage', 'is_anomaly', 'confidence_score']).to_csv(detection_data_path, mode='a', header=not os.path.exists(detection_data_path), index=False)
            detection_data = []

if __name__ == "__main__":
    start_detection()
