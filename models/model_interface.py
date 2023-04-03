from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions on the given test data."""
        pass

    @abstractmethod
    def save(self, filename: str) -> None:
        """Save the model to a file."""
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str) -> Any:
        """Load the model from a file."""
        pass
