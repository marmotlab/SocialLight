# Params for logging on Neptune 
import subprocess 
import datetime 
from Utils import Utilities
from params.Final.training_params import *
from params.Final.network_params import *

# SET THE EXPERIMENT NAME
EXP_NAME               = "COMA-DJ-GAE-vBase"

EXPERIMENT_NAME        =  EXP_NAME #+ datetime.datetime.now().strftime("%d_%m_%H")

if load_model:
    EXPERIMENT_NAME        =  EXP_NAME
# PARAMS FOR LOGGING ON WANDB
wandb                  = False

# PARAMS FOR LOGGING ON NEPTUNE
neptune                = False                      # LOG ON NEPTUNE SERVERS
neptune_run            = "TRAF-93"                 # ONLY NEEDED IF REINITIALIZING A RUN
# neptune_project        = 'yifeng/Traffic'   # USER-ID
# NEPTUNE_API_TOKEN      = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMzIwMTdiNy0zZWM1LTQ1NTAtOWY5Yi1lODRmYjIyZDMyYzgifQ=='

#neptune_project        = "harshg99/Traffic" # USER-ID
#NEPTUNE_API_TOKEN      = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTI4NjE0Yi00ZjNmLTQ5NjktOTdhNy04YTk3ZGQyZTg1MDIifQ=="

neptune_project        = "harshg99/TrafficCF"
NEPTUNE_API_TOKEN      = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTI4NjE0Yi00ZjNmLTQ5NjktOTdhNy04YTk3ZGQyZTg1MDIifQ=="


text_description       = 'Train City Flow DH -net '  # BRIEF TEXT DESCRIPTION
git_head               =  subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])                              # STORE CURRENT GIT HEAD 
git_branch             =  Utilities.current_git_branch()                                                                # STORE CURRENT GIT BRANCH
#description            =  Utilities.describe(text_description,policy_weight,value_weight,entropy_weight,prediction_weight,git_head,git_branch) # GET THE COMPLETE DESCRIPTION
description              = text_description
# SET PATHS 
description_filename   = '/description_{}.txt'.format(EXPERIMENT_NAME)
model_path             = 'data/' + EXPERIMENT_NAME + '/model_traffic_{}'.format(EXPERIMENT_NAME)
gifs_path              = 'data/' + EXPERIMENT_NAME +'/gifs_traffic_{}'.format(EXPERIMENT_NAME)
train_path             = 'data/' + EXPERIMENT_NAME + '/train_traffic_{}'.format(EXPERIMENT_NAME)
TEMP_GIF_FOLDER        = 'data/' + EXPERIMENT_NAME +'/TEMPORARY_gifs_traffic_{}'.format(EXPERIMENT_NAME)