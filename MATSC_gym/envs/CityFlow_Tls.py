import numpy as np


class Tls():
    """
    Sub class for intersections
    """

    def __init__(self, tls_id, tls_config_data,eng):
        self.id = tls_id
        self.config_data = tls_config_data

        self.incoming_lane_list = self.config_data['incoming_lane_list']
        self.outgoing_lane_list = self.config_data['outgoing_lane_list']
        self.lane_links = self.config_data['lane_links']
        self.action_space = self.config_data['action_space']
        self.neighbor_list = self.config_data['neighbor_list']

        self.curr_phase = None
        self.phase_state_one_hot = None
        self.num_veh_in_lane = None
        self.num_veh_out_lane = None
        self.waiting_time_front_veh = None

        # For calculation & Vehicle information
        self.current_time = None
        self.global_lanes_num_veh_dict = None
        self.global_lanes_veh_id_dict = None
        self.global_lanes_num_wait_veh_dict = None
        self.eng = eng
        self.dic_vehicle_arrive_leave_time = dict()
        self.list_lane_vehicle_current_step = []
        self.list_lane_vehicle_previous_step = []

        # Intersection attributes
        self.action_space_n = len(self.action_space) - 1
        self.num_in_lane = len(self.incoming_lane_list)
        self.num_out_lane = len(self.outgoing_lane_list)

        # Phase setting
        self.counter = {'yellow': 0, 'green': 0}

    def observe(self, norm=False):
        """
        Calculate observations 3 X N vectors:
        1. The current phase index
        2. The first vehicle waiting time (optional) not implemented yet
        3. The number of vehicles on the incoming lanes
        4. The number of vehicles on the outgoing lanes
        """
        self.phase_state_one_hot = np.zeros(self.num_in_lane)
        self.waiting_time_front_veh = np.zeros(self.num_in_lane)
        self.num_veh_in_lane = np.zeros(self.num_in_lane)
        self.num_veh_out_lane = np.zeros(self.num_out_lane)
        self.phase_state_one_hot[self.curr_phase] = 1
        for i, (in_lane, out_lane) in enumerate(zip(self.incoming_lane_list, self.outgoing_lane_list)):
            if norm:
                self.num_veh_in_lane[i] = np.clip(self.global_lanes_num_veh_dict[in_lane] / 50, 0, 1)
                self.num_veh_out_lane[i] = np.clip(self.global_lanes_num_veh_dict[out_lane] / 50, 0, 1)
            else:
                self.num_veh_in_lane[i] = self.global_lanes_num_veh_dict[in_lane]
                self.num_veh_out_lane[i] = self.global_lanes_num_veh_dict[out_lane]
        lane_features = np.array([
            self.phase_state_one_hot,
            self.num_veh_in_lane,
            self.num_veh_out_lane
        ])
        return lane_features.T

    def local_observe(self):
        """
        Calculate observations 3 X N vectors:
        1. The current phase index
        2. The number of vehicles on the incoming lanes
        """
        self.phase_state_one_hot = np.zeros(self.num_in_lane)
        self.waiting_time_front_veh = np.zeros(self.num_in_lane)
        self.num_veh_in_lane = np.zeros(self.num_in_lane)
        self.num_veh_out_lane = np.zeros(self.num_out_lane)

        self.phase_state_one_hot[self.curr_phase] = 1
        for i, (in_lane, out_lane) in enumerate(zip(self.incoming_lane_list, self.outgoing_lane_list)):
            self.num_veh_in_lane[i] = np.clip(self.global_lanes_num_veh_dict[in_lane] / 50, 0, 1)
            self.num_veh_out_lane[i] = np.clip(self.global_lanes_num_veh_dict[out_lane] / 50, 0, 1)

        lane_features = np.array([
            self.phase_state_one_hot,
            self.num_veh_in_lane
        ])
        return lane_features.T

    # def get_reward(self):
    #     """
    #     Calculate the rewards based on truncated queue length
    #     """
    #     reward = []
    #     for i, in_lane in enumerate(self.incoming_lane_list):
    #         reward.append(min(15, self.global_lanes_num_wait_veh_dict[in_lane]) / 15)
    #     reward = -1 * np.clip(np.mean(np.array(reward)), 0, 1)
    #     return reward

    def get_reward(self):
        """
        Calculate the rewards capped to 100 vehicles per lane
        """
        reward = []
        for i, in_lane in enumerate(self.incoming_lane_list):
            reward.append(min(100, self.global_lanes_num_wait_veh_dict[in_lane]) / 100)
        reward = -1 * np.clip(np.mean(np.array(reward)), 0, 1)
        return reward

    # def get_reward(self):
    #     # Max pressure reward
    #     reward = []
    #     max_pressure_list = []
    #     for i, in_lane in enumerate(self.incoming_lane_list):
    #         num_incoming_halting = self.global_lanes_num_wait_veh_dict[in_lane]
    #
    #         for j,out_lane in enumerate(self.lane_links[in_lane]):
    #             num_outgoing_halting = self.global_lanes_num_wait_veh_dict[out_lane]
    #             max_pressure_list.append((num_incoming_halting - num_outgoing_halting) /
    #                                      len(self.lane_links))
    #
    #     reward = -1 * np.abs(np.sum(max_pressure_list)/len((self.incoming_lane_list)))
    #     return reward

    # def get_reward(self):
    #     # Max pressure reward
    #     reward = []
    #     max_pressure_list = []
    #
    #     num_incoming = [self.global_lanes_num_wait_veh_dict[in_lane] for in_lane in self.incoming_lane_list]
    #     num_outgoing = [self.global_lanes_num_wait_veh_dict[out_lane] for out_lane in self.outgoing_lane_list]
    #
    #
    #     reward = -1 * np.abs(np.sum(np.array(num_incoming)-np.array(num_outgoing)))\
    #              /len((self.incoming_lane_list))
    #     return reward

    # def get_reward(self):
    #     """
    #         Calculate the rewards based on corrected truncated queue length
    #     """
    #     reward = []
    #
    #     for i, in_lane in enumerate(self.incoming_lane_list):
    #         vehicles = self.global_lanes_veh_id_dict[in_lane]
    #
    #         # Get top 8 vehicles
    #         waiting_vehicles = 0
    #
    #         for v,vehicle in enumerate(vehicles):
    #             veh_data = self.eng.get_vehicle_info(vehicle)
    #             if veh_data[]
    #         reward.append(min(8, self.global_lanes_num_wait_veh_dict[in_lane]) / 8)
    #     reward = -1 * np.clip(np.mean(np.array(reward)), 0, 1)
    #     return reward

    def reset_vars(self):
        self.current_time = None
        self.dic_vehicle_arrive_leave_time = dict()
        self.list_lane_vehicle_current_step = []
        self.list_lane_vehicle_previous_step = []

    # get vehicle information
    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.current_time
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left):
        ts = self.current_time
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")

    def get_vehicle_list(self):
        # get vehicle list
        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_previous_step += self.list_lane_vehicle_current_step  # store current step
        self.list_lane_vehicle_current_step = []
        for lane in self.incoming_lane_list:  # renew current step
            self.list_lane_vehicle_current_step += self.global_lanes_veh_id_dict[lane]

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
