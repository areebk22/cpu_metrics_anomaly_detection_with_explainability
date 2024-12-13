import os
from train import train_model
from detect import start_detection

def main():
    model_name = os.getenv('MODEL_NAME', 'ocsvm').lower()  # Default to One-Class SVM
    model_path = f"/usr/src/app/data/trained_{model_name}.pkl"
    train_duration = int(os.getenv('TRAIN_DURATION', 21600))  # Default to 5 minutes for training
    historical_data_path = os.getenv('HISTORICAL_DATA_PATH', None)
    incremental_training = os.getenv('INCREMENTAL_TRAINING', 'false').lower() == 'true'
    historical_training = os.getenv('HISTORICAL_TRAINING', 'false').lower() == 'true'
    enable_anomalies = os.getenv('INDUCE_ANOMALIES', 'false').lower() == 'true'

    if not os.path.exists(model_path):
        if historical_training:
            train_model(model_name=model_name, duration=0, data_path=historical_data_path, incremental=incremental_training)
        else:
            train_model(model_name=model_name, duration=train_duration, enable_anomalies=enable_anomalies, incremental=False)
    else:
        print(f"Model for {model_name} already exists. Skipping training.")

    start_detection(model_name=model_name, save_detection_data=True)

if __name__ == "__main__":
    main()