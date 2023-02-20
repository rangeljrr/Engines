from datetime import datetime as dt
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from prophet import Prophet

from _local_config import version
from Notebooks.Support_Functions._config import model_path, model_type, plot_path, root
from Notebooks.Support_Functions._pkl_functions import load_model, serialize_model
from Notebooks.Support_Functions._plot_utils import create_legend, plot_config

now = dt.now().date()

def parameter_grid():
    """ This function will create a parameter grid"""
    
    grid = {  
            #'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            #'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['multiplicative', 'additive'],
            #'yearly_seasonality':['auto',True,False, 5,10,20],
            #'weekly_seasonality':['auto',True,False, 5,10,20]
            #'daily_seasonality': [True]
            }     
    return grid


def rods_prophet_model(params):
    """ This function creates the model object and takes in a set of parameters as inputs """
    
    # Initializing the model, can also add regressors model.add_regressor('WEEK') 
    model = Prophet(**params)

    return model


def train_rods_prophet_model(train_data, test_data):
    """ This function will take a train dataframe, test dataframe, a model object, and a 
        dictionary of parrameters to find the best set of paramters """
    
    # Generate all combinations of parameters
    param_grid = parameter_grid()
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    # Create a list to store MAPE values for each combination
    hyper_params = [] 

    # Use cross validation to evaluate all parameters
    for params in all_params:
        
        #try:
            # Fit a model using one parameter combination
            model = rods_prophet_model(params)
            model.fit(train_data)  

            # Predict on our data
            forecast = model.predict(train_data)

            mape = np.mean(np.abs(forecast['yhat'].values - train_data['y']) / train_data.replace(0,1)['y'])
            
            hyper_params.append([mape, params])
            
        #except:
            #pass

    # Finding the best model
    hyper_params = pd.DataFrame(hyper_params, columns = ['train_mape','parameters'])
    best = hyper_params[hyper_params.train_mape == hyper_params.train_mape.min()]
    best['train_date'] = now
    serialize_model(best[['parameters', 'train_date']], model_path.format(version, model_type))


def validate_rods_prophet_model(train_data, test_data):
    """ This function loads the current model deployed and validates train, test mapes"""
    # Validate Model: Load Model
    parameter_dataframe = load_model(model_path.format(version,model_type))
    parameters = parameter_dataframe['parameters'].iloc[0]

    # Vaidate Model: Fit a model using one parameter combination
    model = rods_prophet_model(parameters)
    model.fit(train_data)

    train_forecast = model.predict(train_data)
    test_forecast = model.predict(test_data)
    forecast = pd.concat([train_forecast, test_forecast])

    # Visual
    train_mape = np.round(np.mean(np.abs(train_forecast['yhat'].values - train_data['y']) / train_data['y']),3)
    test_mape = np.round(np.mean(np.abs(test_forecast['yhat'].values - test_data['y']) / test_data['y']),3)

    # Check Paths
    check_path = root + f'/Graphs/{version}/'
    import os
    if not os.path.exists(check_path):
        os.makedirs(check_path)

    # Storing visuals, Creating Figure
    figure, axis = plt.subplots(figsize = (16,6))

    # Time Histogram
    plot_config(f'Train MAPE: {train_mape} | Test MAPE: {test_mape}', axis,'','')

    # Plot Actuals
    plt.plot(train_data['ds'], train_data['y'], color='#1f77b4')
    plt.plot(test_data['ds'], test_data['y'], color='#1f77b4', linestyle='--')
    plt.plot(forecast['ds'], forecast['yhat'], color='red', alpha=.75)

    # Setting Legends
    create_legend({'Original Data':'C0', 'Forecast':'red'}, axis)
    plt.savefig(plot_path.format(version))
    return train_mape, test_mape
