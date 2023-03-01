from modules import arguments, test_arguments
from Utils import Utilities
import os
import argparse

def generate_config(config_name,config_path,existing_config_path,variables=None,test=False): 
    if existing_config_path is not None :
        # In the variables dictionary, specify all variables from the existing config file that you want to modify 
        new_config = Utilities.load_config(existing_config_path) 
        for k,v in variables.items() :
            new_config[k] = v 
        if not os.path.exists(config_path+config_name): 
            os.makedirs(config_path+config_name)
        Utilities.generate_config(new_config, config_path + config_name)

    else :
        # Read from the parameters file 
        if test :
            config = test_arguments.set_test_config_args() 
        else: 
            config = arguments.set_args() 
        if not os.path.exists(config_path+config_name): 
            os.makedirs(config_path+config_name)
        Utilities.generate_config(config, config_path + config_name)

def visualize_configs(config_path) :
    try :
        config = Utilities.load_config(config_path)
    except :
        print('Configuration Not Found')
    for k,v in config.items():
        print(k, ':' , v)

if __name__ =='__main__' : 
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--test', action='store_true', default=False, help='Whether to generate a test config') 
    args = parser.parse_args()
    
    if args.test : 
        config_name = test_name 
        config_path = '../configs/test_configs/'
    else :
        # Specify the name of this new configuration 
        config_name = EXP_NAME
        # Specify the path of storing this config is required
        config_path = '../configs/future_training/'
    # Specify if you want to build over an existing config 
    existing_config_path = None
    variables            = None 

    assert config_name is not None 
    assert config_path is not None 
    if existing_config_path is not None: 
         assert variables is not None 
    #visualize_configs('configs/standard_configs/LSTM_36MetaAgent_noGPU/config.json')
    generate_config(config_name,config_path,existing_config_path,variables,args.test)