from  cityflow_env import CityFlowEnv
import numpy as np
import time
import os
import pandas as pd
import pickle

class CityFlowEnvAsync(CityFlowEnv):

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, server_number=0, seedNum=0):
        super(CityFlowEnvAsync, self).__init__(path_to_log,
                                               path_to_work_directory,
                                               dic_traffic_env_conf,
                                               server_number=0,
                                               seedNum=0
                                               )


    def reset(self,seedNum = 0,server = 0):
        np.random.seed(seedNum)
        state = super().reset(seedNum=seedNum,server=server)

        self.start_junction_decision = dict()

        for agentID in list(self.traffic_light_node_dict.keys()):
            self.start_junction_decision[agentID] = np.random.randint(self.dic_traffic_env_conf['MIN_ACTION_TIME']-1)



    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0] * len(action)


        action_in_sec = list_action_in_sec[i]
        action_in_sec_display = list_action_in_sec_display[i]

        instant_time = self.get_current_time()
        self.current_time = self.get_current_time()

        phase_time_index = self.current_time % self.dic_traffic_env_conf['MIN_ACTION_TIME']
        self.take_action = {}
        for agentID in list(self.traffic_light_node_dict.keys()):
            if phase_time_index == self.start_junction_decision[agentID]:
                self.take_action[agentID] = True

        before_action_feature = self.get_feature()
        # state = self.get_state()

        # if i == 0:
        #     print("time: {0}".format(instant_time))

        self._inner_step(action_in_sec)

        # get reward
        reward = self.get_reward()
        for j in range(len(reward)):
            average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
        self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
        next_state, done = self.get_state()

        # print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            if self.take_action[inter.inter_name]:
                inter.set_signal(
                    action=action[inter_ind],
                    action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                    yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                    path_to_log=self.path_to_log
                )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            if self.take_action[self.list_intersection[inter_ind].inter_name]:
                self.list_inter_log[inter_ind].append({"time": cur_time,
                                                       "state": before_action_feature[inter_ind],
                                                       "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")

            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()
