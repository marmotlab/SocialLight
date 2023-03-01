import traci
import pandas as pd
import numpy as np
import xml.etree.cElementTree as ET
import Utils.Utilities as Utilities
import os

class Tester_SUMO():
    def __init__(self, env, model, args_dict, agent_name, seeds):
        self.env = env
        self.model = model
        self.args_dict = args_dict
        self.num_episodes = self.args_dict['num_episodes']
        self.agent_name = agent_name
        self.output_gifs = self.args_dict['generate_gifs']
        self.seeds = seeds
        self.gui = self.args_dict['gui']
        self.episode_reward = 0
        self.episode_step = 0
        self.action_change = 0
        self.currEpisode = 0
        self.trip_data = []
        self.steps = []
        self.saveGIF = False
        self.output_path = self.args_dict['base_dir']
        self.eval_dir = self.args_dict['base_dir'] + self.args_dict['eval_dir']

    def run_all(self):
        for i in range(self.num_episodes):
            obs = self.reset_episode(test_num=i, seed=self.seeds[i])
            self.run_episode(obs)
            self.collect_tripinfo()
        self.output_data()

    def reset_episode(self, test_num, seed):
        self.episode_reward = 0
        self.episode_step = 0
        self.saveGIF = False
        self.action_change = 0
        self.env.save_images = False
        self.model.reset()
        if self.output_gifs:
            print("making a gif")
            self.saveGIF = True
            self.env.save_images = True

        self.env.reset_tests(test_number=test_num, seed=seed)
        _, _, _, _ = self.env.step(
            action_dict=Utilities.convert_to_nt({i: 0 for i in range(self.args_dict['NUM_RL_AGENT'])}, self.args_dict,
                                                self.env))

    def run_episode(self, obs):
        done = False
        r = None
        while not done:
            obs = self.model.observe_all(r)
            action_dict = self.model.step_all(obs)
            _, r, done, info = self.env.step(action_dict)
            self.episode_reward += info[0]
            self.action_change += info[1]
            self.episode_step += 1
            # print(self.episode_step)
            if done:
                traci.close()
                self.currEpisode += 1
                self.steps.append(self.episode_step)
                break
        print("{} | reward: {}, length: {}, Average action change:{}".format(self.currEpisode, self.episode_reward,
                                                                             self.episode_step,
                                                                             self.action_change / self.episode_step))

        return

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = '../trip_info/trip.xml'
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.currEpisode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)

    def output_data(self):
        traffic_data = pd.DataFrame(self.env.traffic_data)
        traffic_data.to_csv(self.output_path + ('/eval_data/large_grid_{}_traffic.csv'.format(self.agent_name)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.eval_dir + ('/large_grid_{}_trip.csv'.format(self.agent_name)))


class Test_CityFlowV2():
    def __init__(self, env, model, args_dict, seeds):
        self.env = env
        self.seeds = seeds
        self.model = model
        self.args_dict = args_dict

        self.seed = None
        self.curr_episode = None
        self.net_name = self.args_dict['net_name']
        self.agent_name = self.args_dict['agent_name']
        self.num_episodes = self.args_dict['num_episodes']
        self.output_path = self.args_dict['base_dir']
        self.eval_dir = self.output_path + self.args_dict['eval_dir'] + '/{}'.format(self.net_name)

        self.check_base_dir()

    def check_base_dir(self):
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def reset_episode(self, test_num, seed):
        self.episode_reward = 0
        self.episode_step = 0
        self.action_change = 0
        self.seed = seed
        self.curr_episode = test_num
        self.model.reset()

        self.env.reset(seed=seed)

    def run_episode(self):
        """
        Run one episode
        """
        done = False
        r = None
        while not done:
            obs = self.model.observe_all(r)
            action_dict = self.model.step_all(obs)
            _, r, done, info = self.env.step(action_dict)
            self.episode_reward += info[0]
            self.action_change += info[1]
            self.episode_step += 1

        print("{} | Reward: {}, length: {}, Action change:{}".format(self.curr_episode, self.episode_reward,
                                                                     self.episode_step,
                                                                     self.action_change / self.episode_step))

    def run_all(self):
        """
        Run all tests
        """
        avg_travel_time = []
        for i in range(self.num_episodes):
            self.reset_episode(test_num=i, seed=self.seeds[i])
            self.run_episode()
            avg_travel_time.append(self.calculate_summary())
        self.output_evaluation_data(mean=np.mean(np.array(avg_travel_time)),
                                    min=np.min(np.array(avg_travel_time)),
                                    max=np.max(np.array(avg_travel_time)),
                                    all=np.array(avg_travel_time))

    def calculate_summary(self):
        """
        Calculate the average trip time each episode
        """
        dir_to_log_file = self.eval_dir + "/veh_info"
        if not os.path.exists(dir_to_log_file):
            os.makedirs(dir_to_log_file)
        for id in self.env.agent_index:
            # get_dic_vehicle_arrive_leave_time
            dic_veh = self.env.env.list_intersection[self.env.id_to_index[id]].\
                get_dic_vehicle_arrive_leave_time()

            # save them as csv file
            path_to_log_file = dir_to_log_file + "/vehicle_inter_{0}.csv".format(id)
            df = pd.DataFrame.from_dict(dic_veh, orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

        # handling csv file using pandas
        df_vehicle_all = []
        for id in self.env.agent_index:
            path_to_log_file = dir_to_log_file + "/vehicle_inter_{0}.csv".format(id)
            # summary items (duration) from csv
            df_vehicle_inter = pd.read_csv(path_to_log_file,
                                           sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                           names=["vehicle_id", "enter_time", "leave_time"])
            df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
            df_vehicle_inter['leave_time'].fillna(3600, inplace=True)
            df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - df_vehicle_inter["enter_time"].values
            ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
            print("------------- id: {0}\tave_duration: {1}\t"
                  .format(id, ave_duration))
            df_vehicle_all.append(df_vehicle_inter)
        df_vehicle_all = pd.concat(df_vehicle_all)
        vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
        ave_duration = vehicle_duration.mean()
        print('{} | Average Travel Time: {}'.format(self.curr_episode, ave_duration))

        return ave_duration

    def output_evaluation_data(self, **kwargs):
        """
        Output and save the evaluation results/average trip time
        """
        with open(self.eval_dir + "/result.txt", "w+") as f:
            f.write("Average travel time is {0} \n".format(kwargs['mean']))
            f.write("Max travel time is {0} \n".format(kwargs['max']))
            f.write("Min travel time is {0} \n".format(kwargs['min']))
            f.write("All average travel time is {0} \n".format(kwargs['all']))
        print("{} || average travel time is {} \n".format(self.net_name, kwargs['mean']))

if __name__ == '__main__':
    import os
    from Utils import Utilities
    import tensorflow as tf
    import test_arguments

    from MATSC_gym.envs.CityFlow_MATSC import MATSC as CityFlow_Env

    from Test_Models import DRL_Model

    args_dict = test_arguments.set_args()  # First set all arguments before importing other libraries

    agents = []
    seeds = np.zeros((10))
    for config in args_dict['test_configs']:
        model_args = Utilities.load_config('configs/test_configs/' + config + '/config.json')
        combined_args = {**model_args, **args_dict}
        print('Loaded agent', combined_args['agent_name'])
        env = CityFlow_Env(server_number=0, args_dict=combined_args, test=True)
        drl_model = DRL_Model(combined_args, env)
        drl_tester = Test_CityFlowV2(env, drl_model, combined_args, seeds)
        drl_tester.run_all()
        agents.append(combined_args['agent_name'])
        print('Finished Testing', combined_args['agent_name'])
        tf.reset_default_graph()