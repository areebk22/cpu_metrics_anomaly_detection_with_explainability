import config
from detect import start_detection
from train import train_model


def main():
    data_path = config.HISTORICAL_DATA_PATH if config.HISTORICAL_TRAINING else None
    explainer = train_model(duration=config.TRAIN_DURATION, data_path=data_path, force=config.FORCE_RETRAIN)
    start_detection(explainer)


if __name__ == "__main__":
    main()
