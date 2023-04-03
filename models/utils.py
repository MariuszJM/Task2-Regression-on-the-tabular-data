from models.model_interface import BaseModel
from models.linear_regression import LinearRegressor


def get_model(model_name: str) -> BaseModel:
    if model_name == 'linear_regression':
        return LinearRegressor()
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")
