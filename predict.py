import yaml
import pandas as pd
import argparse
import logging

from models.utils import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    logger.info("Loading test data")
    X_test = pd.read_csv(config['data']['test_data'])

    logger.info("Loading the model")
    model = get_model(config['model']['name']).load(config['model']['save_path'])

    logger.info("Making predictions")
    predictions = model.predict(X_test)

    logger.info("Saving predictions")
    pd.DataFrame(predictions).to_csv(config['data']['prediction_output'], index=False)
    logger.info("Predictions saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the config file')
    args = parser.parse_args()

    main(args.config)
