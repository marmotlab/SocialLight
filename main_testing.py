from modules import test_arguments
from Utils import Utilities
import tensorflow as tf

from Utils.plot import plot_eval_curve
from MATSC_gym.envs.Manhattan_MATSC import MATSC as SUMO_Manhattan_Env
from MATSC_gym.envs.CityFlow_MATSC_V2 import MATSC as CityFlow_EnvV2
from modules.Testers import Tester_SUMO, Test_CityFlowV2
from modules.Test_Models import Greedy, DRL_Model
from Map.Manhattan_map.data_baseline.build_file import gen_rou_file

args_dict = test_arguments.set_args()  # First set all arguments before importing other libraries

Utilities.setup_test_logging(args_dict)

# if 'parameter_sharing' or 'Environment' not in args_dict:
#     args_dict['parameter_sharing'] = True
#     args_dict['Environment'] = 'Manhattan'

agents = []
if args_dict['Environment'] == 'sumo':
    if args_dict['test']:
        print('Create files!\n')
        seed = [4000 + i * 100 for i in range(args_dict['num_episodes'])]
        if args_dict['random_routes']:
            raise NotImplemented
        # for i in range(args_dict['num_episodes']):
        #     generateRoutes(args_dict['routes_dir'], seed[i])
        #     print('Generate random routes with seed {}!'.format(seed[i]))
        else:
            for i in range(args_dict['num_episodes']):
                gen_rou_file(path=args_dict['routes_dir'], peak_flow1=args_dict['peak_flow1'],
                             peak_flow2=args_dict['peak_flow2'], density=args_dict['init_density'],
                             seed=seed[i], thread=i)
                print('Generate ma2c routes with seed {}!'.format(seed[i]))

        for config in args_dict['test_configs']:
            model_args = Utilities.load_config('configs/test_configs/' + config + '/config.json')
            combined_args = {**model_args, **args_dict}
            print('Loaded agent', combined_args['agent_name'])
            env = SUMO_Manhattan_Env(server_number=0, args_dict=combined_args, test=True)
            drl_model = DRL_Model(combined_args, env)
            drl_tester = Tester_SUMO(env, drl_model, combined_args, combined_args['agent_name'], seed)
            drl_tester.run_all()
            agents.append(combined_args['agent_name'])
            print('Finished Testing', combined_args['agent_name'])
            tf.reset_default_graph()

        if args_dict['greedy']:
            env = SUMO_Manhattan_Env(server_number=0, args_dict=args_dict, test=True)
            greedy_model = Greedy('0', args_dict, env)
            greedy_tester = Tester_SUMO(env, greedy_model, args_dict, 'greedy', seed)
            greedy_tester.run_all()
            agents.append('greedy')

    if args_dict['load_agents'] is not None:
        for agent in args_dict['load_agents']:
            agents.append(agent)
            trip_file_name = '/large_grid_{}_trip.csv'.format(agent)
            traffic_file_name = '/large_grid_{}_traffic.csv'.format(agent)
            # os.system('cp tested_agents/{} {}/'.format(trip_file_name, args_dict['base_dir'] + args_dict['eval_dir']))
            # os.system('cp tested_agents/{} {}/'.format(traffic_file_name, args_dict['base_dir'] + args_dict['eval_dir']))

    if args_dict['plot']:
        print('Plotting results ....')
        plot_eval_curve(plot_dir=args_dict['base_dir'] + args_dict['plot_dir'],
                        cur_dir=args_dict['base_dir'] + args_dict['eval_dir'],
                        names=agents)
        print('Done! check the results at {} now!'.format(args_dict['base_dir'] + args_dict['plot_dir']))

elif args_dict['Environment'] == 'cityflowV2':
    for config in args_dict['test_configs']:
        seed = [4000 + i * 100 for i in range(args_dict['num_episodes'])]
        model_args = Utilities.load_config('configs/test_configs/' + config + '/config.json')
        combined_args = {**model_args, **args_dict}
        print('Loaded agent', combined_args['agent_name'])
        env = CityFlow_EnvV2(server_number=0, args_dict=combined_args, test=True)
        drl_model = DRL_Model(combined_args, env)
        drl_tester = Test_CityFlowV2(env, drl_model, combined_args, seed)
        drl_tester.run_all()
        agents.append(combined_args['agent_name'])
        print('Finished Testing', combined_args['agent_name'])
        tf.reset_default_graph()