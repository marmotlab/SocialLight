import argparse
from argparse import Namespace
import copy


def set_args():
    globals_dict = set_global_dict()

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--num_episodes', type=int, default=10, help='the number of episodes for testing')
    parser.add_argument('--gui', type=bool, default=False, help='whether to open gui')
    parser.add_argument('--random_routes', type=bool, default=False, help='whether to generate random routes')
    parser.add_argument('--greedy', type=bool, default=False, help='whether test with benchmark agent/greedy agent')
    parser.add_argument('--test', action='store_true', default=False, help='Run testing')
    parser.add_argument('--plot', action='store_true', default=False, help='Run plotting')
    parser.add_argument('--test_configs', type=str, nargs='*', default=['standard'],
                        help='Specify all the configs you want to run testing on, these configs should be placed inside configs/test_configs')
    parser.add_argument('--load_agents', type=str, nargs='*', default=None,
                        help='Specify all the agent name whose csv file needs to be loaded from the tested_agents dir')
    parser.add_argument('--generate_gifs', type=bool, default=False, help='whether to generate the gifs')
    parser.add_argument('--routes_dir', type=str, default='./Test_routes/', help='path for the test routes')
    parser.add_argument('--base_dir', type=str, default='./new_test', help='path for saving test results')
    parser.add_argument('--plot_dir', type=str, default='/plots', help='path for saving test resulted plots')
    parser.add_argument('--eval_dir', type=str, default='/eval_data', help='path for saving test resulted data')
    parser.add_argument('--trip_dir', type=str, default='./trip_info', help='path for saving the trip info file')
    parser.add_argument('--screenshots_dir', type=str, default="./TEMP_DATA_DO_NOT_TOUCH/",
                        help='path for saving screen-shots')
    parser.add_argument('--gifs_dir', type=str, default='/gifs', help='path for saving gifs')

    parser.add_argument('--Environment', type=str, choices=['sumo','cityflowV2'], help='indicate the test env', required=True)
    parser.add_argument('--parameter_sharing', type=bool, default=True, help='indicate whether to use parameter sharing')
    parser.add_argument('--net_name', type=str, help='indicate the net type', required=True)

    args = parser.parse_args()
    for k in args.__dict__:
        globals_dict[k] = args.__dict__[k]
    return globals_dict


def set_global_dict():
    from test_params import environment_params as parameters
    globals_dict = vars(parameters)
    new_dict = {}
    for k, v in globals_dict.items():
        if not k.startswith('__'):
            new_dict[k] = v
    return new_dict


def set_test_config_args():
    from test_params import test_parameters as parameters
    globals_dict = vars(parameters)
    new_dict = {}
    for k, v in globals_dict.items():
        if not k.startswith('__'):
            new_dict[k] = v
    return new_dict