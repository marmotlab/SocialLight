import copy
import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
import gym
from abc import ABC
from multiprocessing import Process
from baselines.AttLight.utils.cityflow_env import CityFlowEnv
# From https://github.com/LiangZhang1996/AttentionLight/

class MATSC(gym.Env, ABC):
    """
    Gym environment for CityFlow
    """

    def __init__(self, server_number, args_dict, test=False):
        self.server_number = server_number
        self.test = test

        self.eng = None
        self.agent_index = None
        self.rl_agent_index = None


        self.TlsDict = {}
        self.neighbour_dict = {}

        self.rl_step = 0
        self.cityflow_step = 0

        self.args_dict = args_dict
        self.base_path = self.args_dict['MAP_FILE_PATH']
        self.config_path = self.args_dict['CONFIG_FILE']
        self.sim_green_duration = self.args_dict['GREEN_DURATION']
        self.sim_yellow_duration = self.args_dict['YELLOW_DURATION']
        self.max_simulation_step = self.args_dict['MAX_EPISODE_SUMO_STEPS']
        self.eval_agent = self.args_dict['EVAL_AGENTS']
        self.pre_actions_dict = None


        # set the dic coe-nf for the lower cityflow environment based on the args_dict
        self.initialization()

    @property
    def id_to_index(self):
        return self.env.id_to_index

    @property
    def index_to_id(self):
        return self.env.index_to_id

    @property
    def action_space(self):
        return self.env.action_space

    def initialization(self):

        # initalise the traffic dictionary
        dic_traffic_env_conf = {

            "LIST_MODEL": ["Fixedtime", "MaxPressure", "MaxQL", "FRAP",
                           "PressLight", "DQN", "GAT", "Attention"],
            "LIST_MODEL_NEED_TO_UPDATE": ["PressLight", "FRAP", "GAT", "Attention", "DQN"],

            "FORGET_ROUND": 20,
            "RUN_COUNTS": self.max_simulation_step,
            "MODEL_NAME": None,
            "TOP_K_ADJACENCY": 5,

            "ACTION_PATTERN": "set",
            "NUM_INTERSECTIONS": 1,

            "MIN_ACTION_TIME": self.sim_green_duration,
            "MEASURE_TIME": 15,

            "BINARY_PHASE_EXPANSION": True,

            "YELLOW_TIME": self.sim_yellow_duration,
            "ALL_RED_TIME": self.args_dict['ALL_RED_DURATION'],
            "NUM_PHASES": self.args_dict['a_size'],
            "NUM_LANES": [3, 3, 3, 3],

            "INTERVAL": 1,

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "pressure",
                "adjacency_matrix"
            ],
            "DIC_REWARD_INFO": {
                "queue_length": 0,
                "pressure": 0,
            },
            "PHASE": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            },
            "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
            "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

        }

        roadnet_size = self.args_dict['BASE_FILE'].split('/')[1]
        NUM_COL = int(roadnet_size.split('_')[1])
        NUM_ROW = int(roadnet_size.split('_')[0])
        count = 3600
        num_rounds = 80
        num_intersections = NUM_ROW * NUM_COL
        mod = 'Attention'
        gen = 1
        dic_traffic_env_conf_extra = {
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,

            "MODEL_NAME": mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": self.args_dict['FLOW_FILE'],
            "ROADNET_FILE": self.args_dict['ROAD_NET_FILE'],
            "TRAFFIC_SEPARATE": self.args_dict['FLOW_FILE'],
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_queue_vehicle_in",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },
        }

        if self.args_dict['a_size']==8:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        self.deploy_dic_traffic_env_conf = copy.deepcopy(dic_traffic_env_conf)
        self.deploy_dic_traffic_env_conf.update(dic_traffic_env_conf_extra)

        self.path_to_log = os.path.join(self.args_dict['DIR'],self.args_dict['LOG_DIR'].format(
            self.args_dict['EXP_NAME']
        ))
        self.path_to_work_dir = os.path.join(self.args_dict['DIR'],self.args_dict['WORK_DIR'].format(
            self.args_dict['EXP_NAME']
        ))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
        if not os.path.exists(self.path_to_work_dir):
            os.makedirs(self.path_to_work_dir)
        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.path_to_work_dir,
            dic_traffic_env_conf=self.deploy_dic_traffic_env_conf
        )
        self.traffic_light_node_dict = self.env._adjacency_extraction()
        self.agent_index = self.rl_agent_index = list(self.traffic_light_node_dict.keys())
        self.pre_actions_dict = { id: None for id in self.agent_index}
        self.neighbour_dict = {}
        for id in self.traffic_light_node_dict.keys():
            self.neighbour_dict[id] = self.traffic_light_node_dict[id]['neighbor_ENWS']



    def reset(self,seed = None):
        """
        Resets the environment to an initial state and returns an initial observation.
        @ return: initial observations
        """
        print('Worker {} || Reset Environment'.format(self.server_number))

        # Reset CityFlow
        seed = self.args_dict['SEED']
        if self.args_dict['RANDOM_SEED'] and seed is None:
            seed = np.random.randint(10000)
        obs  = self.env.reset(seed,self.server_number)
        self.observation_cache = self.convert_obs(obs)
        return self.observation_cache

    def check_terminal(self):
        """
        check terminal condition
        """
        done = False
        if self.test:
            if self.env.eng.get_current_time() > self.max_simulation_step - 1:
                done = True
        else:
            if self.env.eng.get_current_time() > self.max_simulation_step - 1:
                done = True
            elif self.env.eng.get_vehicle_count() <= 0:
                done = True
        return done

    def step(self, action_dict):

        sum_action_change = 0
        change_action_list = []
        for id in self.agent_index:
            if self.pre_actions_dict[id] is not None and self.pre_actions_dict[id] != action_dict[id]:
                change_action_list.append(id)
                if id in self.rl_agent_index:
                    sum_action_change += 1

        action = []
        for k in action_dict.keys():
            action.append(action_dict[k])

        next_state,rewards,done,avg_rewards_list = self.env.step(action)
        total_rewards = {}

        # Adding the neighbour rewards
        for j, ID in enumerate(self.agent_index):
            if self.args_dict['REWARD_SHARING']:
                neighbor_reward = []
                neighbor_id = self.neighbour_dict[ID]
                for neighbor in neighbor_id:
                    if neighbor is not None:
                        neighbor_reward.append(rewards[self.id_to_index[neighbor]])
                total_rewards[ID] = (rewards[j] + (
                        self.args_dict['Neighbor_factor'] * np.sum(neighbor_reward)))/self.args_dict['MAX_EPISODE_LENGTH']
                # total_rewards[ID] = individual_rewards[ID] + (self.args_dict['Neighbor_factor'] * np.sum(neighbor_reward) / (len(neighbor_reward)))
            else:
                total_rewards[ID] = rewards[j]//self.args_dict['MAX_EPISODE_LENGTH']
        rewards = total_rewards
        next_state_dict = self.observation_cache = self.convert_obs(next_state)

        traffic_data = self.measure_traffic_step()
        avg_action_change = sum_action_change / int(len(self.rl_agent_index))
        avg_multi_agent_reward = sum(rewards.values()) / int(len(self.rl_agent_index))
        done = self.check_terminal()
        self.pre_actions_dict = action_dict
        return  next_state_dict,rewards,done,[avg_multi_agent_reward,avg_action_change,traffic_data]

    def local_observe(self,id):
        return self.observation_cache[id]

    def convert_obs(self,next_state):
        next_states_dict = {}
        for j, agent_state in enumerate(next_state):
            id = self.index_to_id[j]
            next_states_dict[id] = []
            for feature in self.deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == 'cur_phase':
                    next_states_dict[id].append(self.deploy_dic_traffic_env_conf["PHASE"]
                                                [agent_state[feature][0]])
                else:
                    next_states_dict[id].append(agent_state[feature])

            next_states_dict[id] = np.concatenate(next_states_dict[id], axis=0)
        return  next_states_dict

    def measure_traffic_step(self):
        """
        Traffic metrics measurements for each RL step
        """
        # veh_list = self.eng.get_vehicles(include_waiting=False)
        # num_tot_car = len(veh_list)


        if self.server_number in self.eval_agent:
            speeds = list(self.env.eng.get_vehicle_speed().values())
            queues = list(self.env.eng.get_lane_waiting_vehicle_count().values())
            time = self.env.eng.get_average_travel_time()

            avg_speed = np.mean(np.array(speeds))
            avg_queue = np.mean(np.array(queues))
            std_queue = np.std(np.array(queues))
            avg_travel_time = np.mean(np.array(time))

            curr_traffic = {'avg_speed_mps': avg_speed,
                            'avg_wait_sec': avg_travel_time,
                            'avg_queue': avg_queue,
                            'std_queue': std_queue}

            return curr_traffic
        else:
            return None



