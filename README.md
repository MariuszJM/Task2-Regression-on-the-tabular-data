# Regression on Tabular Data

This project aims to build a model that predicts a target based on 53 anonymized features.

# Exploratory Data Analysis
Linear regression has been implemented as the best solution for the data based on the analysis.

## Training the Model

To train the linear regression model, execute the `train.py` script with the appropriate configuration file:
`python train.py -c configs/linear_regression.yaml`

This will train the model on the provided training data and save the trained model in the `saved_models/` directory.

## Making Predictions

To make predictions using the trained model, execute the `predict.py` script with the appropriate configuration file:
`python predict.py -c configs/linear_regression.yaml`

This will generate predictions for the `hidden_test.csv` file and save the results in the `predictions.csv` file.

## Customization

You can customize the project by adding new models or modifying the configuration files. To add a new model, create a new Python file in the `models/` directory and implement the model based on the abstract base class in `models/model_interface.py`. Update the configuration files in the `configs/` directory to use the new model.

For any issues or questions, feel free to open an issue on the GitHub repository.
