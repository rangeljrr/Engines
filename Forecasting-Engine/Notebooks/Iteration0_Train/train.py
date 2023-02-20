import numpy as np
import pandas as pd
import os, sys
from datetime import datetime as dt

from Notebooks.Iteration0_Train._local_config import version
from Notebooks.Support_Functions._config import run_log_path, performance_log_path, raw_data_path, run_log_path
from Notebooks.Support_Functions._data_functions import feature_engineer, prep_data, train_test_split
from Notebooks.Support_Functions._log_functions import append_log,init_parent_directories, init_performance_log_file, init_run_log_file
from Notebooks.Iteration0_Train._training_functions import train_rods_prophet_model, validate_rods_prophet_model

def run_train():

    # Initialize Timestamp and log path
    today = dt.now().date()
    init_parent_directories()
    status_log_path = run_log_path.format(version, str(dt.now()))
    init_run_log_file(status_log_path)
    append_log(status_log_path, f"Processed Started,"+str(dt.now()))

    try:
        # Loading the data
        data = pd.read_csv(raw_data_path)
        append_log(status_log_path, f"Data Loaded Successfully,"+str(dt.now()))
    except:
        append_log(status_log_path, f"Data Load Failed,"+str(dt.now()))

    try:
        # Create the modeling dataset
        data = prep_data(data)
        append_log(status_log_path, f"Data Prepped Successfully,"+str(dt.now()))
    except:
        append_log(status_log_path, f"Data Prep Failed,"+str(dt.now()))

    try:
        # Engineer new features
        data = feature_engineer(data)
        append_log(status_log_path, f"Feature Engineer Successful,"+str(dt.now()))
    except:
        append_log(status_log_path, f"Feature Engineer Failed,"+str(dt.now()))

    try:
        performance_values = []
        # Split the data into training and testing. Since monthly, we will choose 12
        train_data, test_data = train_test_split(data, 12)

        # Returning best params, mape, model
        train_rods_prophet_model(train_data, test_data)
        append_log(status_log_path, f"Model Trained Successfully,"+str(dt.now()))
    except:
        append_log(status_log_path, f"Model Train Failed,"+str(dt.now()))

    try:
        # Validate Model | Create Visuals
        train_mape, test_mape = validate_rods_prophet_model(train_data, test_data)
        append_values = f'{version},{train_mape},{test_mape},{today}'

        # Log Performance
        init_performance_log_file(performance_log_path)
        append_log(performance_log_path, append_values)
        
        append_log(status_log_path, f"Model Validated Successfully,"+str(dt.now()))
    except:
        append_log(status_log_path, f"Model Validation Failed,"+str(dt.now()))

    append_log(status_log_path, f"Process Ran Sucessfully,"+str(dt.now()))
if __name__ == "__main__":
    run_train()