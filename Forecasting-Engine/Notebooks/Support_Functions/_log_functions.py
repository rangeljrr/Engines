import os
from os.path import exists as file_exists

from Notebooks.Support_Functions._config import root
from Notebooks.Iteration0_Train._local_config import version

def init_parent_directories():
    """ This function will initialize all directories"""
    paths = [f'{root}/Graphs/{version}',
             f'{root}/Models/{version}',
             f'{root}/Logs/Prediction_Logs',
             f'{root}/Logs/Performance_Logs',
             f'{root}/Logs/Run_Status_Logs',
             f'{root}/Logs/Run_Status_Logs/{version}']
    
    for path in paths:
        try:
            os.mkdir(path)
        except:
            pass

def init_run_log_file(path):
    """ This function will check is the Run_Status_Log file exists, otherwise create it"""
    if file_exists(path):
        pass
    else:
        with open(path,'w') as f:
            f.write('process_status, timestamp')

def init_performance_log_file(path):
    """ This function will check is the Run_Status_Log file exists, otherwise create it"""
    if file_exists(path):
        pass
    else:
        with open(path,'w') as f:
            f.write('version,train_mape,test_mape,timestamp')

def append_log(path, values):

    with open(path,'a') as f:
        f.write('\n' + values)
