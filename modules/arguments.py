import argparse 
from argparse import Namespace 
import params.parameters as parameters 
import copy
from types import ModuleType


def set_args() :
    globals_dict= set_global_dict()
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--NUM_META_AGENTS', type=int, default=globals_dict['NUM_META_AGENTS'] ,help='Number of meta-agents to train with')
    parser.add_argument('--Framework',type = str, default=globals_dict['Framework'],help ='Pytorch or tensorflow')
    parser.add_argument('--device',type = str, default=globals_dict['device'],help ='cpu or cuda')
    parser.add_argument('--MAX_EPISODE_LENGTH', type = int , default=globals_dict['MAX_EPISODE_LENGTH'] , help='Maximum episode length ') 
    parser.add_argument('--GPU', type = bool , default=globals_dict['GPU'] , help='Train on GPU')
    parser.add_argument('--REWARD_STRUCTURE', type = str , default=globals_dict['REWARD_STRUCTURE'] , help='Reward structure to use')
    parser.add_argument('--load_model', type = bool , default=globals_dict['load_model']  , help='Load a model')
    parser.add_argument('--model_path', type = str , default=globals_dict['model_path'] , help='Path of the model to load')
    parser.add_argument('--a_size', type = int , default=globals_dict['a_size'], help='Action space dimension') 
    parser.add_argument('--RAY_RESET_EPS', type = int , default=globals_dict['RAY_RESET_EPS'] , help='Number of episodes after which ray is reset to conserve memory') 
    parser.add_argument('--OUTPUT_GIFS', type = bool , default=globals_dict['OUTPUT_GIFS']  , help='Output GIFS')
    parser.add_argument('--policy_weight',type =float , default =globals_dict['policy_weight'] , help= 'Weight on policy loss')
    parser.add_argument('--value_weight',type =float , default = globals_dict['value_weight'] , help= 'Weight on value loss')
    parser.add_argument('--text_description', type =str , default= globals_dict['text_description'] , help='Describe the experiment')
    parser.add_argument('--epochs', type=int, default=globals_dict['epochs'] ,help='Number of episodes to train for')
    parser.add_argument('--Buffer', type =str , default= globals_dict['Buffer'] , help='Describe the experiment')
    parser.add_argument('--Model', type =str , default= globals_dict['Model'] , help='Describe the experiment')
    parser.add_argument('--Metric', type =str , default= globals_dict['Metric'] , help='Describe the experiment')
    parser.add_argument('--Variable', type =str , default= globals_dict['Variable'] , help='Describe the experiment')
    parser.add_argument('--Observer', type =str , default= globals_dict['Observer'] , help='Describe the experiment')
    parser.add_argument('--config_path', type =str , default= globals_dict['config_path'] , help='Specify a config file if needed')
    parser.add_argument('--debug', action='store_true' ,default=False, help='Specify a config file if needed')
    parser.add_argument('--neptune', type = str, default=globals_dict['neptune'], help = 'Store on neptune ai server')
    parser.add_argument('--neptune_project', type=str, default=globals_dict['neptune_project'],
                        help='Store on neptune ai server')
    parser.add_argument('--NEPTUNE_API_TOKEN', type=str, default=globals_dict['NEPTUNE_API_TOKEN'],
                        help='Store on neptune ai server')
    parser.add_argument('--SINGLE_GRADIENT', action='store_true', default=globals_dict['SINGLE_GRADIENT'],
                        help='Single gradient for all agents')
    parser.add_argument('--neptune_run', type=str , default=globals_dict['neptune_run'], help='Specify run name if loading an existing model for neptune')
    parser.add_argument('--wandb', type = str, default=globals_dict['wandb'], help = 'Store on wandb ai server')

    args = parser.parse_args() 
    
    #Some hand configuration for the Runner file 
    parameters_global = vars(parameters) 
    parameters_global['GPU'] = args.GPU
    parameters_global['NUM_META_AGENTS'] = args.NUM_META_AGENTS

    for k in args.__dict__ :
        globals_dict[k] = args.__dict__[k] 
    return globals_dict 

def set_global_dict() :
    globals_dict = vars(parameters)
    new_dict = {} 
    for k,v in globals_dict.items() :
        if not k.startswith('__') and not isinstance(v, ModuleType):
            new_dict[k] = v 
    return new_dict 

if __name__=='__main__':
    args = set_args()
