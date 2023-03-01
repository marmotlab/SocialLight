import traci
import math
import numpy as np 
#from parameters import *
from sumolib import checkBinary


class Tls:
    def __init__(self, ID,args_dict ):
        self.ID = ID
        self.lane_list = None
        self.detector_list = None
        self.lane_links = None
        self.outgoing_lane_list = None
        self.args_dict = args_dict
        self._num_Lanes = self.args_dict['NUM_LANES']
        self._num_incoming_lanes = self.args_dict['NUM_INCOMING']
        self._N_num_incoming_lanes = self._S_num_incoming_lanes = 1
        self._E_num_incoming_lanes = self._W_num_incoming_lanes = 2
        self._tlc_state_baseline = [[1, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1],
                                    [0, 0, 0, 0, 0, 0]]
        self._lane_state = np.zeros(self._num_incoming_lanes)
        self._last_lane_state = np.ones(self._num_incoming_lanes)
        self._num_links_lane = np.zeros(self._num_incoming_lanes)

        self._waiting_time_lane = np.zeros(self._num_incoming_lanes)
        self._waiting_time_lane_normalized = np.zeros(self._num_incoming_lanes)
        self._num_out_vehicles = np.zeros(self._num_Lanes)
        self.pass_time_history = []
        self._time = np.zeros(self._num_Lanes)
        self.in_flow = np.zeros(self._num_incoming_lanes)
        self.out_flow = np.zeros(self._num_incoming_lanes)
        self.in_flow_detectors_index = {0: [0, 2, 4], 1: [0, 1, 3], 2: [1, 3, 5], 3: [0, 3, 4]}

        self.halting_time = []
        self.qua_acc_waiting_per_lane = np.zeros(self._num_incoming_lanes)
        self.acc_waiting_per_lane = np.zeros(self._num_incoming_lanes)
        self.curr_suc_waiting_time = np.zeros(self._num_incoming_lanes)

        # For reward wrt accumulative waiting time
        self.curr_waiting = 0
        self.pre_waiting = 0

        #Action space 
        self.action_space = [0 for _ in range(5)]
        # For Green utilization
        self.distance_red_list = [[159, 129, 91], [165, 128, 75]]
        self.max_vehicles_red_list = [[3, 4, 5], [3, 5, 7]]
        self.max_vehicles_green_list = [5, 6]
        self._GU_time_list = []
        self._GU_0 = np.zeros(self._num_incoming_lanes)
        self._GU_1 = np.zeros(self._num_incoming_lanes)
        self._GU_2 = np.zeros(self._num_incoming_lanes)

        self.edg_agent = False
        self.pre_num_in_vehicles = np.zeros(self._num_incoming_lanes)

        # Hardcoded max pressure
        self.num_segment = 3
        self.max_num_veh_seg = 8
        self.seg_pos = [60, 120, 180]
        self.num_vehicle_seg_1 = np.zeros(self._num_incoming_lanes)
        self.num_vehicle_seg_2 = np.zeros(self._num_incoming_lanes)
        self.num_vehicle_seg_3 = np.zeros(self._num_incoming_lanes)
        self._num_outgoing_lane_vehicles = np.zeros(self._num_incoming_lanes)

    def initialization(self):
        # Acquire incoming lanes ids, outgoing lanes ids and detector ids
        self.lane_list = []
        self.halting_time = []
        self.lane_links = {}
        self.outgoing_lane_list = []
        lanes = traci.trafficlight.getControlledLanes(self.ID)
        for lane in lanes:
            if lane not in self.lane_list:
                self.lane_list.append(lane)
        for i in range(len(self.lane_list)):
            self.halting_time.append({})
            self._num_links_lane[i] = traci.lane.getLinkNumber(self.lane_list[i])
            self.lane_links[self.lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.lane_list[i])
            for info in outgoing_lanes_info:
                self.lane_links[self.lane_list[i]].append(info[0])
            for lane in self.lane_links[self.lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)
        self.detector_list = self.lane_list
        return self.lane_list, self.detector_list, self._num_links_lane

    def updateWaiting(self):
        for i in range(self._num_incoming_lanes):
            detectorID = self.detector_list[i]
            newVehicles = traci.inductionloop.getLastStepVehicleIDs(detectorID)
            # if self._lane_state[i] == 0 and newVehicles != ():
            if traci.vehicle.getWaitingTime(newVehicles) != 0 and newVehicles != ():
                self._waiting_time_lane[i] += 1
            else:
                self._waiting_time_lane[i] = 0

    def update_flow_rate(self):
        """
        1.The number of vehicle that passed through the detectors.
        2.The time that vehicles need to pass through the detectors.
        ps: The info from ILD contains [vehiclesID, vehicle len, entry time, exit time]
        """
        for i in range(self._num_incoming_lanes):
            detectorID = self.detector_list[i]
            info = traci.inductionloop.getVehicleData(detectorID)
            if info:
                info = list(info[0])
                if info[3] != -1:
                    if traci.vehicle.getWaitingTime(info[0]) == 0:
                        self._num_out_vehicles[i] += 1
                        # If waiting for a red light, calculated by the equation of motion(hardcoded)
                        if (info[3] - info[2]) > 5:
                            self.pass_time_history[i].append(math.sqrt(2 * self.args_dict['LENGTH_VEHICLE'] / self.args_dict['ACCELERATION']))
                        else:
                            self.pass_time_history[i].append(info[3] - info[2])
        self._last_lane_state = self._lane_state

    def update_start_halting_time(self):
        # Update start halting time for calculating accumulative waiting time
        curr_time = traci.simulation.getTime()
        for i in range(self._num_incoming_lanes):
            lane_id = self.lane_list[i]
            vehicle_list = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle in vehicle_list:
                if vehicle not in self.halting_time[i]:
                    if traci.vehicle.getWaitingTime(vehicle) > 0:
                        self.halting_time[i][vehicle] = curr_time - 1

    def update(self):
        # Update dynamic traffic data
        # Only activate when need to calculate accumulative waiting time
        curr_time = traci.simulation.getTime()
        for i in range(self._num_incoming_lanes):
            lane_id = self.lane_list[i]
            vehicle_list = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle in vehicle_list:
                if vehicle not in self.halting_time[i]:
                    if traci.vehicle.getWaitingTime(vehicle) > 0:
                        self.halting_time[i][vehicle] = curr_time - 1

            detectorID = self.detector_list[i]
            info = traci.inductionloop.getVehicleData(detectorID)
            if info:
                info = list(info[0])
                if info[3] != -1:
                    if traci.vehicle.getWaitingTime(info[0]) == 0:
                        self._num_out_vehicles[i] += 1

    def reset_detector(self):
        # Reset the detector, used for calculating flow rate
        # Only call this before executing phase/action
        self.pass_time_history = []
        for i in range(self._num_incoming_lanes):
            self.pass_time_history.append([])
        self._time = np.zeros(self._num_incoming_lanes)
        self._num_out_vehicles = np.zeros(self._num_incoming_lanes)
        self._last_lane_state = np.ones(self._num_incoming_lanes)

    def _set_yellow_phase(self, old_action, duration):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase(self.ID, yellow_phase_code)
        traci.trafficlight.setPhaseDuration(self.ID, duration)

    def _set_green_phase(self, action, duration):
        green_phase_code = action * 2
        traci.trafficlight.setPhase(self.ID, green_phase_code)
        traci.trafficlight.setPhaseDuration(self.ID, duration)

    def _set_all_red_phase(self, action, duration):
        red_phase_code = action
        traci.trafficlight.setPhase(self.ID, red_phase_code)
        traci.trafficlight.setPhaseDuration(self.ID, duration)

    def _get_lane_state(self):
        # retrieve the lane state to determine which lane is moving and which is waiting(0/stop 1/move)
        tlc_state = traci.trafficlight.getPhase(self.ID)
        self._lane_state = self._tlc_state_baseline[tlc_state]
        return self._lane_state

    def _get_lane_state_onehot(self):
        self._lane_state_onehot = np.zeros(self._num_incoming_lanes)
        tlc_state = traci.trafficlight.getPhase(self.ID)
        phase_code = int(tlc_state / 2)
        self._lane_state_onehot[phase_code] = 1
        return self._lane_state_onehot

    def _get_waiting_time(self):
        for i in range(self._num_incoming_lanes):
            vehicle_id = traci.inductionloop.getLastStepVehicleIDs(self.detector_list[i])
            if vehicle_id != ():
                self._waiting_time_lane[i] = traci.vehicle.getWaitingTime(vehicle_id[0])
            else:
                self._waiting_time_lane[i] = 0

            # Waiting time normalization
            self._waiting_time_lane_normalized[i] = math.tanh(0.005 * self._waiting_time_lane[i])

        return self._waiting_time_lane_normalized

    def _get_out_flow(self, step):
        # Calculate the out flow rate
        # Warning: Need to specify the maximum speed and the Length of vehicle
        n_max = np.trunc((self.args_dict['MAX_VEHICLE_SPEED'] * step) / (2 * self.args_dict['LENGTH_VEHICLE']))
        t_min = 2 * self.args_dict['LENGTH_VEHICLE'] / self.args_dict['MAX_VEHICLE_SPEED']
        for i in range(self._num_incoming_lanes):
            if self.pass_time_history[i]:
                self._time[i] = np.mean(self.pass_time_history[i], axis=0)
            else:
                self._time[i] = 0

            self.out_flow[i] = 2 * (self._num_out_vehicles[i] / n_max) * (t_min / (self._time[i] + t_min))

        return self.out_flow

    def _get_in_flow(self, neighbor_outflow):
        # Calculate the in flow rate from the neighbors' out flow rate
        # Warning: Not using right now
        # Todo use in flow rate and out flow rate to predict the number of vehicles for each lane
        if not self.edg_agent:
            in_flow = np.zeros(4)
            for i in range(4):
                indexes = self.in_flow_detectors_index[i]
                for index in indexes:
                    in_flow[i] += neighbor_outflow[i][index] / self._num_links_lane[index]
            # Hardcoded
            for i in range(self._N_num_incoming_lanes):
                self.in_flow[i] = in_flow[0]
            for i in range(self._E_num_incoming_lanes):
                self.in_flow[i + self._N_num_incoming_lanes] = in_flow[1]
            for i in range(self._S_num_incoming_lanes):
                self.in_flow[i + self._N_num_incoming_lanes + self._E_num_incoming_lanes] = in_flow[2]
            for i in range(self._W_num_incoming_lanes):
                self.in_flow[i + self._N_num_incoming_lanes + self._E_num_incoming_lanes + self._S_num_incoming_lanes] = \
                    in_flow[3]
        else:
            for index in range(self._num_incoming_lanes):
                num_vehicles = traci.lane.getLastStepVehicleNumber(self.lane_list[index])
                # print(num_vehicles, self.pre_num_in_vehicles[index], self._number_vehicles[index])
                num_in = num_vehicles + self._num_out_vehicles[index] - self.pre_num_in_vehicles[index]
                self.pre_num_in_vehicles[index] = num_vehicles
                self.in_flow[index] = np.clip((num_in / 10) * 0.8, 0, 1.0)
        return self.in_flow

    def _get_queue_lengths(self):
        # Retrieve the number of vehicle for each lane
        queueLengths = np.zeros(self._num_incoming_lanes)
        for i in range(self._num_incoming_lanes):
            queueLengths[i] = traci.lane.getLastStepVehicleNumber(self.lane_list[i])

        normalizedQueueLengths = np.tanh(0.1 * queueLengths)
        return normalizedQueueLengths

    def _get_halting_queue_length(self):
        # Retrieve the number of halting vehicle for each lane
        halting_queue = np.zeros(self._num_incoming_lanes)
        for i in range(self._num_incoming_lanes):
            halting_queue[i] = traci.lane.getLastStepHaltingNumber(self.lane_list[i])

        normalized_halting_queue = np.tanh(0.1 * halting_queue)
        return halting_queue, normalized_halting_queue

    def _get_green_utilization(self):
        # Need to specify the distance for lane segment, the maximum number of vehicles for each step
        # Warning: Not using right now
        # Todo implemented in a generalized way
        lane_length = 185
        # Red case: distance measurement for region partition/max number of vehicles can pass in a green->red phase
        # index=0/(N and S), index=1/(E and w)
        distance_red_list = [[159, 129, 91], [165, 128, 75]]
        # max_speed_list = [11, 20]
        max_vehicles_red_list = [[3, 4, 5], [3, 5, 7]]
        max_vehicles_green_list = [5, 6]

        # For 3 steps prediction(0/step0, 1/step1, 2/step2)
        self._GU_red_0 = np.zeros(self._num_incoming_lanes)
        self._GU_red_1 = np.zeros(self._num_incoming_lanes)
        self._GU_red_2 = np.zeros(self._num_incoming_lanes)

        self.GU_green_0 = np.zeros(self._num_incoming_lanes)
        self.GU_green_1 = np.zeros(self._num_incoming_lanes)
        self.GU_green_2 = np.zeros(self._num_incoming_lanes)

        self._GU_time_list = []
        # calculate prediction for 3 RL step: t, t+1, t+2
        for i in range(self._num_incoming_lanes):
            # Create a list to record the no of phase for each vehicle each lane
            self._GU_time_list.append([])

        # Current time phase
        pre_lane_state = self._get_lane_state()
        for i in range(self._num_incoming_lanes):
            laneId_GU = self.lane_list[i]
            # Get the vehicles' id at the current lane
            veh_dict = traci.lane.getLastStepVehicleIDs(laneId_GU)
            # Calculate the time required for each vehicle on each lane
            for v in veh_dict:
                # Vehicle position(from the entry of incoming lanes)
                position1 = traci.vehicle.getLanePosition(v)
                # Vehicle position(from the exit of the incoming lanes )
                position = lane_length - position1
                # Vehicle speed and acceleration
                speed = traci.vehicle.getSpeed(v)
                # acceleration = traci.vehicle.getAcceleration(v)

                # 1.Previous lane state is red
                if pre_lane_state[i] == 0:
                    # The attributes of N-S lanes are different from E-S lanes
                    if i % 3 == 0:
                        distance_red = distance_red_list[0]
                        if position1 > distance_red[0]:
                            self._GU_red_0[i] += 1
                        elif position1 > distance_red[1]:
                            self._GU_red_1[i] += 1
                        elif position1 > distance_red[2]:
                            self._GU_red_2[i] += 1
                    else:
                        distance_red = distance_red_list[1]
                        if position1 > distance_red[0]:
                            self._GU_red_0[i] += 1
                        elif position1 > distance_red[1]:
                            self._GU_red_1[i] += 1
                        elif position1 > distance_red[2]:
                            self._GU_red_2[i] += 1

                # 2. Previous lane state is green
                else:
                    # Different ways to calculate the needed travel time
                    # 1. Consider the dynamics of vehicles
                    # if (speed ** 2 + 2 * acceleration * position) >= 0:
                    #     # Estimate the time that the vehicle need to pass
                    #     time_to_pass = (-speed + (speed ** 2 + 2 * acceleration * position) ** 0.5) / acceleration
                    # else:
                    #     time_to_pass = 0
                    # 2. Use the max limited speed
                    # if i % 3 == 0:
                    #     time_to_pass = position / max_speed_list[0]
                    # else:
                    #     time_to_pass = position / max_speed_list[1]
                    # 3. Use the real time speed
                    if speed != 0:
                        time_to_pass = position / speed
                        # print('Vehicle:{}, Time:{}'.format(v, time_to_pass))
                        # Calculate the no of phase(corresponding to the duration of the phases)
                        pass_number = math.ceil(time_to_pass / self.args_dict['GREEN_DURATION'])
                        self._GU_time_list[i].append(pass_number)

            if pre_lane_state[i] == 1:
                # check the rationality of data(because of the estimation for the green case calculation)
                for j in range(len(self._GU_time_list[i])):
                    if j != (len(self._GU_time_list[i])) - 1:
                        if self._GU_time_list[i][j] < self._GU_time_list[i][j + 1]:
                            self._GU_time_list[i][j] = self._GU_time_list[i][j + 1]
                        else:
                            pass

                        if self._GU_time_list[i][j] <= 1:
                            self.GU_green_0[i] += 1
                        elif self._GU_time_list[i][j] <= 2:
                            self.GU_green_1[i] += 1
                        elif self._GU_time_list[i][j] <= 3:
                            self.GU_green_2[i] += 1

                # Normalization
                if i % 3 == 0:
                    max_vehicles_green = max_vehicles_green_list[0]
                    self._GU_0[i] = self.GU_green_0[i] / max_vehicles_green
                    self._GU_1[i] = self.GU_green_1[i] / max_vehicles_green
                    self._GU_2[i] = self.GU_green_2[i] / max_vehicles_green
                else:
                    max_vehicles_green = max_vehicles_green_list[1]
                    self._GU_0[i] = self.GU_green_0[i] / max_vehicles_green
                    self._GU_1[i] = self.GU_green_1[i] / max_vehicles_green
                    self._GU_2[i] = self.GU_green_2[i] / max_vehicles_green

            else:
                # Normalization
                if i % 3 == 0:
                    max_vehicles_red = max_vehicles_red_list[0]
                    self._GU_0[i] = self._GU_red_0[i] / max_vehicles_red[0]
                    self._GU_1[i] = self._GU_red_1[i] / max_vehicles_red[1]
                    self._GU_2[i] = self._GU_red_2[i] / max_vehicles_red[2]
                else:
                    max_vehicles_red = max_vehicles_red_list[1]
                    self._GU_0[i] = self._GU_red_0[i] / max_vehicles_red[0]
                    self._GU_1[i] = self._GU_red_1[i] / max_vehicles_red[1]
                    self._GU_2[i] = self._GU_red_2[i] / max_vehicles_red[2]
            # Clip
            np.clip(self._GU_0[i], 0, 1)
            np.clip(self._GU_1[i], 0, 1)
            np.clip(self._GU_2[i], 0, 1)

        # print("GU_0:{}".format(self._GU_0))
        # print()
        # print("GU_1:{}".format(self._GU_1))
        # print()
        # print("GU_2:{}".format(self._GU_2))
        # print()

        return self._GU_0, self._GU_1, self._GU_2

    def _get_accumulative_waiting_time(self):
        # Calculate the accumulating time
        # Warning: Only call this function while requiring the reward(at the end of the phase duration)
        # Delete the vehicles that leave the intersection from the dict
        self.del_list = []
        for i in range(self._num_incoming_lanes):
            self.del_list.append([])
            vehicle_list = traci.lane.getLastStepVehicleIDs(self.lane_list[i])
            for vehicle in self.halting_time[i]:
                if vehicle not in vehicle_list:
                    self.del_list[i].append(vehicle)

        for i in range(self._num_incoming_lanes):
            for vehicle in self.del_list[i]:
                del self.halting_time[i][vehicle]

        curr_time = traci.simulation.getTime()
        self.qua_acc_waiting_per_lane = np.zeros(self._num_incoming_lanes)
        self.acc_waiting_per_lane = np.zeros(self._num_incoming_lanes)

        for i in range(len(self.halting_time)):
            for vehicle in self.halting_time[i]:
                self.qua_acc_waiting_per_lane[i] += (curr_time - self.halting_time[i][vehicle]) ** 2
                self.acc_waiting_per_lane[i] += (curr_time - self.halting_time[i][vehicle])

    def _get_acc_time_per_lane(self):
        self._get_accumulative_waiting_time()
        return self.acc_waiting_per_lane

    def _get_qua_acc_time_per_lane(self):
        self._get_accumulative_waiting_time()
        return self.qua_acc_waiting_per_lane

    def _get_number_out_vehicles(self):
        out_number_vehicle = np.sum(self._num_out_vehicles)

        return out_number_vehicle

    def _get_num_vehicles_seg(self):
        for i in range(self._num_incoming_lanes):
            vehicles_Ids = traci.lane.getLastStepVehicleIDs(self.lane_list[i])
            for v in vehicles_Ids:
                if traci.vehicle.getLanePosition(v) <= self.seg_pos[0]:
                    self.num_vehicle_seg_3[i] += 1
                elif traci.vehicle.getLanePosition(v) <= self.seg_pos[1]:
                    self.num_vehicle_seg_2[i] += 1
                elif traci.vehicle.getLanePosition(v) <= self.seg_pos[2]:
                    self.num_vehicle_seg_1[i] += 1
            # Normalized
            self.num_vehicle_seg_1[i] = np.clip(self.num_vehicle_seg_1[i] / self.max_num_veh_seg, 0, 1.0)
            self.num_vehicle_seg_2[i] = np.clip(self.num_vehicle_seg_2[i] / self.max_num_veh_seg, 0, 1.0)
            self.num_vehicle_seg_3[i] = np.clip(self.num_vehicle_seg_3[i] / self.max_num_veh_seg, 0, 1.0)

        return self.num_vehicle_seg_1, self.num_vehicle_seg_2, self.num_vehicle_seg_3

    def _get_incoming_vehicles_lane(self):
        total_num_incoming_vehicles = np.zeros(self._num_incoming_lanes)
        for i in range(self._num_incoming_lanes):
            total_num_incoming_vehicles[i] = traci.lane.getLastStepVehicleNumber(self.lane_list[i]) / 24
        return total_num_incoming_vehicles

    def _get_num_outgoing_vehicles_lane(self):
        for i, lane in enumerate(self.outgoing_lane_list):
            self._num_outgoing_lane_vehicles[i] = traci.lane.getLastStepVehicleNumber(lane) / (
                        self.num_segment * self.max_num_veh_seg)
        return self._num_outgoing_lane_vehicles

    def observe(self):
        lane_state = self._get_lane_state_onehot()
        self._num_outgoing_lane_vehicles = self._get_num_outgoing_vehicles_lane()
        for i in range(self._num_incoming_lanes):
            max_pos = 0
            first_veh = None
            vehicles_Ids = traci.lane.getLastStepVehicleIDs(self.lane_list[i])
            for v in vehicles_Ids:
                if traci.vehicle.getLanePosition(v) <= self.seg_pos[0]:
                    self.num_vehicle_seg_3[i] += 1
                elif traci.vehicle.getLanePosition(v) <= self.seg_pos[1]:
                    self.num_vehicle_seg_2[i] += 1
                elif traci.vehicle.getLanePosition(v) <= self.seg_pos[2]:
                    self.num_vehicle_seg_1[i] += 1
                if traci.vehicle.getLanePosition(v) > max_pos:
                    max_pos = traci.vehicle.getLanePosition(v)
                    first_veh = v
            if first_veh is not None:
                self._waiting_time_lane_normalized[i] = traci.vehicle.getWaitingTime(first_veh) / 500
            else:
                self._waiting_time_lane_normalized[i] = 0
            # Normalized
            self.num_vehicle_seg_1[i] = np.clip(self.num_vehicle_seg_1[i] / self.max_num_veh_seg, 0, 1.0)
            self.num_vehicle_seg_2[i] = np.clip(self.num_vehicle_seg_2[i] / self.max_num_veh_seg, 0, 1.0)
            self.num_vehicle_seg_3[i] = np.clip(self.num_vehicle_seg_3[i] / self.max_num_veh_seg, 0, 1.0)

        laneFeatures = np.array([lane_state, self._waiting_time_lane_normalized, self._num_outgoing_lane_vehicles,
                                 self.num_vehicle_seg_1, self.num_vehicle_seg_2, self.num_vehicle_seg_3])
        return laneFeatures.T

    def small_observe(self):
        lane_state = self._get_lane_state_onehot()
        self._num_outgoing_lane_vehicles = self._get_num_outgoing_vehicles_lane() 
        for i in range(self._num_incoming_lanes):
            vehicleID = traci.lanearea.getLastStepVehicleIDs(self.detector_list[i])
            max_pos = 0
            first_veh = None
            for v in vehicleID:
                if traci.vehicle.getLanePosition(v) > max_pos:
                    max_pos = traci.vehicle.getLanePosition(v)
                    first_veh = v
            if first_veh != None:
                self._waiting_time_lane_normalized[i] = traci.vehicle.getWaitingTime(first_veh) / 500
            else:
                self._waiting_time_lane_normalized[i] = 0

        laneFeatures = np.array([lane_state, self._waiting_time_lane_normalized, self._num_outgoing_lane_vehicles]) 

        return laneFeatures

    def _get_suc_waiting_reward(self):
        # Calculate successive waiting time reward
        for i in range(self._num_incoming_lanes):
            laneId = self.lane_list[i]
            self.curr_suc_waiting_time[i] = traci.lane.getWaitingTime(laneId)
        curr_tol_suc_waiting = np.mean(self.curr_suc_waiting_time)
        reward = np.clip(-1 * math.tanh(1e-3 * curr_tol_suc_waiting), -1, 0)
        return reward

    def _get_acc_waiting_reward(self):
        # Calculate the accumulative waiting time
        # Normalized by the number of vehicles per lane
        num_vehicles = 0
        vehicles_per_lane = np.zeros((self._num_incoming_lanes))
        for i in range(self._num_incoming_lanes):
            num_vehicles += traci.lane.getLastStepHaltingNumber(self.lane_list[i])
            vehicles_per_lane[i] = traci.lane.getLastStepVehicleNumber(self.lane_list[i]) + 1

        self.qua_acc_waiting_per_lane = self._get_qua_acc_time_per_lane()
        self.pre_waiting = self.curr_waiting
        self.curr_waiting = np.sum(np.array(self.qua_acc_waiting_per_lane) / np.array(vehicles_per_lane))
        if self.pre_waiting == 0 or num_vehicles == 0 or np.sum(vehicles_per_lane) == 0:
            reward = 0
        else:
            # reward = np.clip(np.tanh(- 1e-5 * (self.curr_waiting / num_vehicles)), -1, 0)
            # reward = np.clip(((self.pre_waiting - self.curr_waiting) / self.pre_waiting), -1.0, 1.0)
            reward = np.clip(math.tanh(1.e-5 * (self.pre_waiting - self.curr_waiting)), -1.0, 1.0)
        return reward

    def _get_out_vehicles_reward(self):
        self.num_out = self._get_number_out_vehicles()
        if self.num_out < 0:
            self.num_out = 0
        reward = np.clip(self.num_out / 12, 0, 1)
        return reward

    def _get_incoming_lane_queue_reward(self):
        queue_incoming_lane = np.zeros(self._num_incoming_lanes)
        for i in range(self._num_incoming_lanes):
            # queue_incoming_lane[i] = traci.lane.getLastStepHaltingNumber(self.lane_list[i])
            vehicleIds = traci.lane.getLastStepVehicleIDs(self.lane_list[i])
            for vehicle in vehicleIds:
                if traci.vehicle.getSpeed(vehicle) < 1 and traci.vehicle.getAcceleration(vehicle) <= 1:
                    queue_incoming_lane[i] += 1
        reward = np.clip((-1 * np.sum(queue_incoming_lane) / 120), -1, 0)

        return reward

    def _get_outgoing_lane_queue_reward(self):
        queue_outgoing_lane = np.zeros(self._num_incoming_lanes)
        for i, outgoing_lane in enumerate(self.outgoing_lane_list):
            queue_outgoing_lane[i] = traci.lane.getLastStepHaltingNumber(outgoing_lane)
        reward = np.clip((-1 * np.sum(queue_outgoing_lane) / 120), -1, 0)

        return reward

    def _get_max_pressure_reward(self):
        max_pressure_list = []
        for incoming_lane in self.lane_list:
            num_incoming_halting = traci.lane.getLastStepHaltingNumber(incoming_lane)
            outgoing_lanes = self.lane_links[incoming_lane]
            for outgoing_lane in outgoing_lanes:
                num_outgoing_halting = traci.lane.getLastStepHaltingNumber(outgoing_lane)
                max_pressure_list.append((num_incoming_halting - num_outgoing_halting) /
                                         (self.max_num_veh_seg * self.num_segment))
        reward = -1 *(np.sum(max_pressure_list) / self._num_incoming_lanes)

        return reward

    def _get_product_reward(self) :
        queue_incoming_lane = np.zeros(self._num_incoming_lanes)
        for i in range(self._num_incoming_lanes):
            vehicleIds = traci.lane.getLastStepVehicleIDs(self.lane_list[i])
            for vehicle in vehicleIds:
                if traci.vehicle.getSpeed(vehicle) < 1 and traci.vehicle.getAcceleration(vehicle) <= 1:
                    queue_incoming_lane[i] += 1
        
    def _get_baseline_queue_reward(self):
        queue = []
        for i in range(self._num_incoming_lanes):
            queue.append(traci.lanearea.getLastStepHaltingNumber(self.detector_list[i]) / 7)
        reward = -1 * np.mean(np.array(queue))
        return reward
    
    def _get_local_reward(self):
        # Specify the reward structure
        local_reward = None
        if self.args_dict['REWARD_STRUCTURE'] == 'queue':
            incoming_queue_reward = self._get_incoming_lane_queue_reward()
            local_reward = incoming_queue_reward
        elif self.args_dict['REWARD_STRUCTURE'] == 'suc_waiting':
            suc_reward = self._get_suc_waiting_reward()
            local_reward = suc_reward
        elif self.args_dict['REWARD_STRUCTURE'] == 'acc_waiting+out':
            acc_reward = self._get_acc_waiting_reward()
            out_reward = self._get_out_vehicles_reward()
            local_reward = (acc_reward + out_reward) / 2
        elif self.args_dict['REWARD_STRUCTURE'] == 'queue+suc_waiting':
            incoming_queue_reward = self._get_incoming_lane_queue_reward()
            suc_reward = self._get_suc_waiting_reward()
            local_reward = (incoming_queue_reward + self.args_dict['Waiting_factor'] * suc_reward) / 2
        elif self.args_dict['REWARD_STRUCTURE'] == 'queue+acc_waiting':
            acc_reward = self._get_acc_waiting_reward()
            queue_reward = self._get_incoming_lane_queue_reward()
            local_reward = (acc_reward + queue_reward) / 2
        elif self.args_dict['REWARD_STRUCTURE'] == 'queue+acc_waiting+out':
            acc_reward = self._get_acc_waiting_reward()
            out_reward = self._get_out_vehicles_reward()
            queue_reward = self._get_incoming_lane_queue_reward()
            local_reward = (acc_reward + queue_reward + out_reward) / 3
        elif self.args_dict['REWARD_STRUCTURE'] == 'max_pressure':
            local_reward = self._get_max_pressure_reward()
        elif self.args_dict['REWARD_STRUCTURE'] == 'product' :
            local_reward = self._get_product_reward() 
        elif self.args_dict['REWARD_STRUCTURE'] == 'truncated_queue':
            local_reward = self._get_baseline_queue_reward()

        assert local_reward is not None, 'No reward?!'

        return local_reward

    def _get_tol_reward(self, neighbor_rewards):
        # Calculate the local reward with neighbor rewards
        local_reward = self._get_local_reward()
        # Normalized
        total_reward = (local_reward + self.args_dict['Neighbor_factor'] * np.sum(neighbor_rewards)) / (len(neighbor_rewards) + 1)
        return total_reward


if __name__ == '__main__':
    import os
    from arguments import set_args

    path = "{}/Manhattan_map/data_baseline/exp.sumocfg".format(os.path.abspath(os.path.join(os.getcwd(), "../..")))
    sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", path])
    ID = 'nt8'
    args_dict = set_args()
    tls = Tls(ID, args_dict=args_dict)
    tls.edg_agent = True
    tls.initialization()
    while True:
        tls.reset_detector()
        # print('Time:{}'.format(traci.simulation.getTime()))
        a = np.random.randint(0, 5, 1)[0]
        tls._set_green_phase(a, 10)
        for i in range(5):
            traci.simulationStep()
        print("Time:{}, Action:{}".format(traci.simulation.getTime(), a))
        print(tls.small_observe())