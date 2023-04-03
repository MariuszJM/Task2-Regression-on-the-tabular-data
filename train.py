import argparse
import yaml
import pandas as pd
import logging

from models.utils import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    logger.info("Loading training data")
    data = pd.read_csv(config['data']['train_data'])
    X_train = data.drop("target", axis=1)
    y_train = data["target"]

    logger.info("Initializing the model")
    model = get_model(config['model']['name'])

    logger.info("Training the model")
    model.fit(X_train, y_train)

    logger.info("Saving the model")
    model.save(config['model']['save_path'])
    logger.info("Model training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the config file')
    args = parser.parse_args()

    main(args.config)
