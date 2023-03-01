from modules import arguments, driver
from Utils import Utilities

globals_dict= arguments.set_args()  # First set all arguments before importing other libraries

loading = globals_dict['load_model']
neptune_run = globals_dict['neptune_run'] 

if globals_dict['config_path'] is not None :
    globals_dict = Utilities.load_config(globals_dict['config_path']) 
    globals_dict['load_model'] = loading 
    globals_dict['neptune_run'] = neptune_run 
    if not globals_dict['load_model'] :
        globals_dict = Utilities.update_paths(globals_dict) 

run = Utilities.setup_neptune(globals_dict)
wandb_run = Utilities.setup_wandb(globals_dict)

Utilities.create_logging_utils(globals_dict) 

if not globals_dict['debug'] and not globals_dict['load_model'] :
    Utilities.create_config(globals_dict) 
if (globals_dict['Framework']=='tf'):
    driver.main(globals_dict, run, wandb_run)

