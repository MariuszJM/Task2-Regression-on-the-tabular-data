import argparse
import yaml
import pandas as pd
from models.linear_regression import LinearRegressor
from models.model_interface import BaseModel


def main(config_path: str) -> None:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    X_train = pd.read_csv(config['data']['train_data'])
    y_train = pd.read_csv(config['data']['train_target'])

    model = get_model(config['model']['name'])
    model.fit(X_train, y_train)

    model.save(config['model']['save_path'])


def get_model(model_name: str) -> BaseModel:
    if model_name == 'linear_regression':
        return LinearRegressor()
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the config file')
    args = parser.parse_args()

    main(args.config)
