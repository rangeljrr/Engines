import pandas as pd
from Notebooks.Support_Functions._config import raw_data_path, model_data_path

def prep_data(dataframe):
    """ This function will prepare the data for modeling """
    dataframe = dataframe.rename(columns={'Month':'ds','Passengers':'y'})
    dataframe['ds'] = pd.to_datetime(dataframe['ds'])
    
    return dataframe

def feature_engineer(dataframe):
    """ This function will engineer new features for modeling """

    return dataframe

def train_test_split(data, n):
    """ This function will take a timeseries tail and use the last 
        n values as the testing dataset """
    
    return data.iloc[:-n], data.iloc[-n:]