from sklearn.linear_model import LinearRegression
from joblib import dump, load
from models.model_interface import BaseModel
import pandas as pd
from typing import Any
import os


class LinearRegressor(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by extracting the '6' feature and squaring it."""
        return data['6'].values.reshape(-1, 1) ** 2

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_train_preprocessed = self.preprocessing(X_train)
        self.model.fit(X_train_preprocessed, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        X_test_preprocessed = self.preprocessing(X_test)
        return self.model.predict(X_test_preprocessed)

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dump(self, filename)

    @staticmethod
    def load(filename: str) -> Any:
        return load(filename)
