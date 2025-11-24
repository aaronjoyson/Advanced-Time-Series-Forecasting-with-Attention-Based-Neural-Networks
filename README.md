Attention-Based Time Series Forecasting
Overview

This project builds an attention-based LSTM model for forecasting multivariate time series data.
The dataset contains sensor-like readings, with several features and a target column.
The notebook prepares the data, creates forecasting sequences, trains models, compares performance, and extracts attention weights.

The goal is to predict the next five values of the target using the last thirty timesteps of all features.

Dataset

The notebook uses the uploaded dataset:

synthetic_sensor_timeseries_5000.csv


It contains multiple numerical columns.
All columns except the last one are treated as input features.
The last column is used as the forecasting target.

Main Steps
1. Load and scale data

The dataset is loaded using pandas and scaled to the range 0–1 using MinMaxScaler.

2. Create sequences

A sliding window is used.
Each sample consists of 30 past timesteps of all features.
The model predicts the next 5 target values.

3. Train-validation-test split

The data is divided according to time order:
70 percent training, 15 percent validation, and 15 percent testing.

4. Baseline LSTM model

A simple LSTM model is trained to serve as a comparison point.

5. Custom attention-based LSTM

A custom attention layer is implemented.
This layer helps the model weigh the importance of each timestep.
The attention-LSTM model processes sequences and produces a 5-step forecast.

6. Rolling validation

Rolling window validation is used to evaluate model stability over time.
For each fold, the model trains on all data before the fold and tests on the next segment.

7. Hyperparameter tuning

A small grid of configurations is tested.
The combination with the lowest average RMSE is used for the final model.

8. Final model training

The attention-LSTM model is trained using the best hyperparameters.
Performance is measured on the test split.

9. Attention weight extraction

Attention weights are extracted for one test example.
These weights show how much each timestep contributed to the model’s prediction.

Requirements

The project uses the following dependencies:

numpy
pandas
matplotlib
tensorflow
scikit-learn

How to Run

Place the dataset file inside the same directory.

Open and run the notebook:
attention_timeseries_project.ipynb

Execute all cells in order.

Review the printed scores and attention plots.

Results

The notebook outputs:

RMSE values from rolling validation

Test performance of the final model

Attention weight plot showing the importance of previous timesteps

Notes

Time-based splitting is used instead of random splitting.
Attention helps identify which past points influence future predictions.
The code is written in a simple and readable style for clarity.
