import yaml
import pandas as pd
import argparse

from models.utils import get_model


def main(config_path: str) -> None:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    X_test = pd.read_csv(config['data']['test_data'])

    model = get_model(config['model']['name']).load(config['model']['save_path'])
    predictions = model.predict(X_test)

    pd.DataFrame(predictions).to_csv(config['data']['prediction_output'], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the config file')
    args = parser.parse_args()

    main(args.config)
