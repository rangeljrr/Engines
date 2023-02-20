import numpy as np
import pandas as pd
import sys
from datetime import datetime as dt

from _local_config import version
from Notebooks.Support_Functions._config import model_path, performance_logs, raw_data_path
from Notebooks.Support_Functions._data_functions import feature_engineer, prep_data, train_test_split
from Notebooks.Iteration0_Train._training_functions import train_rods_prophet_model, validate_rods_prophet_model

def run_train():

    # Initialize Timestamp
    now = dt.now().date()

    data = pd.read_csv(raw_data_path)

    # Create the modeling dataset
    data = prep_data(data)

    # Engineer new features
    data = feature_engineer(data)

    performance_values = []
    # Split the data into training and testing. Since monthly, we will choose 12
    train_data, test_data = train_test_split(data, 12)

    # Returning best params, mape, model
    train_rods_prophet_model(train_data, test_data)

    # Validate Model | Create Visuals

    train_mape, test_mape = validate_rods_prophet_model(train_data, test_data)
    performance_values.append([train_mape,test_mape, now])
    performance_values = pd.DataFrame(performance_values, columns=['train_mape','test_mape','now'])

    # Log Performance
    performance_values.to_csv(performance_logs.format(version), index=False)

if __name__ == "__main__":
    run_train()