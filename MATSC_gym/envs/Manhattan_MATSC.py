# coding=utf-8
import gym
import traci
import numpy as np 
from sumolib import checkBinary
from MATSC_gym.envs.Manhattan_Tls import Tls

from Utils.utils import check_SUMO_HOME, Manhattan_neighbor_map
from Map.Manhattan_map.data_baseline.build_file import gen_rou_file


class MATSC(gym.Env):
    def __init__(self, server_number, args_dict,test=False):
        super(MATSC, self).__init__()

        check_SUMO_HOME()
        self.TlsDict = {}
        self.args_dict = args_dict
        self.num_agents = self.args_dict['NUM_TLS']
        self.gif_folder = args_dict['TEMP_GIF_FOLDER']
        self._server_number = server_number
        self.rand_numbers = [ 0 for _ in range(100)]
        self.num_incoming_lanes = args_dict['NUM_INCOMING']
        self.num_lanes =args_dict['NUM_LANES']
        self._sumo_steps_green_phase = args_dict['GREEN_DURATION']
        self._sumo_steps_yellow_phase = args_dict['YELLOW_DURATION']
        self._sumo_steps_all_red_phase =args_dict['ALL_RED_DURATION']
        self.seed = self._server_number
        self.gui = self.args_dict['GUI']
        self._step = 0
        self._sumo_step = 0 
        self.test= test 
        self._first_run = True
        self.done = False
        self.save_images = False
        self.change_action_list = []
        self.adjacency_matrix = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])

        # For setting type for multi-agent
        self.edge_agent_ID = []
        self.learning_agent_ID = []
        self.greedy_agent_ID = []
        self.agentID = []

        # For multi-agent action execution
        self.prev_actionsDict = {}

        # For collect traffic data
        self.traffic_data = []
        self.currEpisode = 0
        self.neighbor_map = Manhattan_neighbor_map()

        self.set_agent_type()
        reward_vars = self.args_dict['REWARD_STRUCTURE'].split('+')  

        self.update_tls_flag = False 
        for var in reward_vars :
            if var in ['out','acc_waiting'] :
                self.update_tls_flag = True 

        self.eval_agent = False 

        if self._server_number in self.args_dict['EVAL_AGENTS'] :
            self.eval_agent = True 
                
    def set_agent_type(self):
        self.edge_agent_ID = ['nt1', 'nt2', 'nt3', 'nt4', 'nt5', 'nt6', 'nt10', 'nt11',
                              'nt15', 'nt16', 'nt20', 'nt21', 'nt22', 'nt23', 'nt24', 'nt25']
        if self.args_dict['NUM_RL_AGENT'] == 9:
            self.learning_agent_ID = ['nt7', 'nt8', 'nt9', 'nt12', 'nt13', 'nt14', 'nt17', 'nt18', 'nt19']
            self.greedy_agent_ID = self.edge_agent_ID
        elif self.args_dict['NUM_RL_AGENT'] == 25:
            self.learning_agent_ID = ['nt1', 'nt2', 'nt3', 'nt4', 'nt5',
                                      'nt6', 'nt7', 'nt8', 'nt9', 'nt10',
                                      'nt11', 'nt12', 'nt13', 'nt14', 'nt15',
                                      'nt16', 'nt17', 'nt18', 'nt19', 'nt20',
                                      'nt21', 'nt22', 'nt23', 'nt24', 'nt25']
            self.greedy_agent_ID = []
        
        self.agent_index = self.learning_agent_ID

        self.neighbour_dict = {} 
        
        for x,y in enumerate(self.learning_agent_ID) :
            self.neighbour_dict[y] = [] 
            possible_neighbours = [-1,+1,-5,+5] 
            for element in possible_neighbours :
                if 1<= (x+1+element)<=25 :
                    if (x+1) %5 ==0 and element == 1 :
                        self.neighbour_dict[y].append(None) 
                    elif (x+1) %5 ==1 and element ==-1 :
                        self.neighbour_dict[y].append(None)
                    else : 
                        self.neighbour_dict[y].append('nt{}'.format(x+element+1)) 
                else :
                    self.neighbour_dict[y].append(None) 

        self.agent_type_dict = {} 
        for i in range(self.args_dict['NUM_RL_AGENT']) :
            self.agent_type_dict[i] = np.zeros(4) 
            for j in range(4) :
                self.agent_type_dict[i][j] = int(self.neighbour_dict['nt{}'.format(i+1)][j] != None) 
                
        self.agentID = self.learning_agent_ID + self.greedy_agent_ID
        assert (len(self.learning_agent_ID) == self.args_dict['NUM_RL_AGENT'])
        assert (len(self.agentID) == self.args_dict['NUM_TLS'])

        for i in range(self.num_agents):
            ID = 'nt{}'.format(i + 1)
            self.TlsDict[ID] = Tls(ID,self.args_dict)
            if ID in self.edge_agent_ID:
                self.TlsDict[ID].edg_agent = True

    def setWorld(self, config_file_path):
        try:
            # To check if the duration settings for yellow and all red phase are reasonable
            if (self._sumo_steps_yellow_phase + self._sumo_steps_all_red_phase) < self._sumo_steps_green_phase:
                pass
            else:
                print("please set the reasonable durations for yellow and all red phase")
            if self.gui and (self._server_number == 0):
                sumoBinary = checkBinary('sumo-gui')
            else:
                sumoBinary = checkBinary('sumo')

            traci.start([sumoBinary,
                         "-c", config_file_path,
                         "--tripinfo-output", self.args_dict['MAP_FILE_PATH'] + "trip_{}.xml".format(self._server_number),
                         "--start",
                         "--quit-on-end",
                         "--time-to-teleport", self.args_dict['TELEPORT_TIME'],
                         '--no-warnings', 'True',
                         '--no-step-log', 'True' ,
                         '--seed', str(self.seed)])
            return 0

        except:
            print("was not able to start simulation\n")
            return 1

    def reset(self):
        # print("\nRESETTING\n")
        if not self.test: 
            if not self._first_run:
                traci.close()
            else:
                self._first_run = False

        if self.args_dict['RANDOM_SEED']:
            self.seed = np.random.randint(0, 100000)
        else:
            self.seed = self.args_dict['SEED']
        self.rand_numbers[np.random.randint(0,100)] +=1

        gen_rou_file(self.args_dict['MAP_FILE_PATH'], self.args_dict['peak_flow1'], self.args_dict['peak_flow2'], self.args_dict['init_density'], self.seed, self._server_number)
        configFileName = "exp_{}.sumocfg".format(self._server_number)
        configFilePath = self.args_dict['MAP_FILE_PATH'] + "{}".format(configFileName)

        # Initialization, run self.setWorld(configPath) until it succeeds
        self.setWorld(configFilePath)
        for ID in self.agentID:
            self.TlsDict[ID].initialization()
            self.prev_actionsDict[ID] = None
        self._step = 0
        self._sumo_step = 0
        self.done = False
        self.reset_tls_detector()
        self.traffic_data = []

    def step(self, action_dict):
        self.reset_tls_detector()
        self.change_action_list = []
        rewards_dict = {}
        num_action_change = 0
        total_multi_agent_reward = 0
        done = False

        # check if change phase
        for ID in self.agentID:
            if self.prev_actionsDict[ID] is not None and self.prev_actionsDict[ID] != action_dict[ID]:
                self.change_action_list.append(ID)
                if ID in self.learning_agent_ID:
                    num_action_change += 1

        for t in range(self._sumo_steps_green_phase):
            for ID in self.agentID:
                if t == 0:
                    if ID not in self.change_action_list:
                        self.TlsDict[ID]._set_green_phase(action_dict[ID], duration=self.args_dict['GREEN_DURATION'])
                    else:
                        self.TlsDict[ID]._set_yellow_phase(self.prev_actionsDict[ID], duration=self.args_dict['YELLOW_DURATION'])
                elif t == self._sumo_steps_yellow_phase:
                    if ID in self.change_action_list:
                        self.TlsDict[ID]._set_green_phase(action_dict[ID], duration=self.args_dict['GREEN_DURATION'] - self.args_dict['YELLOW_DURATION'])

            self.step_a_sumo_step()

            if t == (self._sumo_steps_green_phase - 1):
                rewards_dict = self.get_reward(self.args_dict['REWARD_SHARING'])
                total_multi_agent_reward = sum(rewards_dict.values())

            done = self.check_terminal()

            if done :
                if ('nt1' not in rewards_dict) : 
                    rewards_dict = self.get_reward(self.args_dict['REWARD_SHARING'])
                break

        self.prev_actionsDict = action_dict
        self._step += 1
        avg_multi_agent_reward = total_multi_agent_reward / self.args_dict['NUM_RL_AGENT']
        avg_action_change = num_action_change / self.args_dict['NUM_RL_AGENT']

        if self.eval_agent: 
            traffic_data = self.measure_traffic_step()
        else :
            traffic_data = None 

        return None, rewards_dict, done, [avg_multi_agent_reward, avg_action_change, traffic_data]

    def step_a_sumo_step(self):
        if self.update_tls_flag :
            self.update_tls_data()
        traci.simulationStep()
        # save images to get Gifs
        if self.save_images:
            self.save_image()
        self._sumo_step += 1
        if self.test:
            self.measure_traffic_step_test()

    def observe(self):
        # Retrieve local observations of all agents
        observations = {}

        for ID in self.agentID:
            observations[ID] = self.TlsDict[ID].observe()

        return observations 

    def reset_tls_detector(self):
        for ID, tls in self.TlsDict.items():
            tls.reset_detector()

    def update_tls_data(self):
        for ID, tls in self.TlsDict.items():
            tls.update()

    def get_reward(self, rewardSharing=False):
        individual_rewards = {} 
        for ID in self.learning_agent_ID:
            individual_rewards[ID] = self.TlsDict[ID]._get_local_reward()
        if rewardSharing:
                total_rewards = {} 
                for ID in self.learning_agent_ID: 
                    neighbor_reward = []
                    neighbor_id = self.neighbour_dict[ID]
                    for neighbor in neighbor_id:
                        if neighbor is not None: 
                            neighbor_reward.append(individual_rewards[neighbor]) 
                    total_rewards[ID] = individual_rewards[ID] + (self.args_dict['Neighbor_factor'] * np.sum(neighbor_reward))
                    #total_rewards[ID] = individual_rewards[ID] + (self.args_dict['Neighbor_factor'] * np.sum(neighbor_reward) / (len(neighbor_reward)))
                return total_rewards 
        else :
            return individual_rewards 

    def check_terminal(self):
        # Check if it is terminal
        if self.test:
            if traci.simulation.getTime() >= self.args_dict['MAX_EPISODE_SUMO_STEPS']:
                self.done = True
        else:
            if traci.simulation.getMinExpectedNumber() <= 0:
                self.done = True
            elif traci.simulation.getTime() >= self.args_dict['MAX_EPISODE_SUMO_STEPS']:
                self.done = True
        return self.done

    def save_image(self):
        traci.gui.setSchema('View #0', 'real world')
        traci.gui.setZoom('View #0', 400.0)
        traci.gui.setOffset('View #0', 0, 0)
        size = 2000
        step = str(self._sumo_step).zfill(3)
        fileNameTemplate = 'step{}.png'.format(step)
        filePath = self.gif_folder + "/" + fileNameTemplate
        traci.gui.screenshot(viewID='View #0',
                             filename=filePath,
                             width=size, height=size)

        return filePath

    def measure_traffic_step(self):
        vehicles_Id = traci.vehicle.getIDList()
        num_tot_car = len(vehicles_Id)

        if num_tot_car > 0:
            avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles_Id])
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in vehicles_Id])
        else:
            avg_speed = 0
            avg_waiting_time = 0

        # all trip-related measurements are not supported by traci,
        # need to read from output file afterwards
        queues = []
        for i in self.agentID:
            for j in range(self.TlsDict[i]._num_incoming_lanes):
                queues.append(traci.lane.getLastStepHaltingNumber(self.TlsDict[i].lane_list[j]))

        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}

        return cur_traffic

# For test
    def reset_tests(self, test_number, seed, testPath="./Test_routes",gui=False):
        self.currEpisode = test_number
        print("\nRESETTING\n")
        configFileName = "exp_{}.sumocfg".format(test_number)
        configFilePath = "{}/{}".format(testPath, configFileName)
      
        # run self.setWorld(configPath) until it succeeds
        # self.setWorld(configFilePath)
        if gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        traci.start(
            [sumoBinary, '-c', configFilePath,
             '--tripinfo-output', self.args_dict['trip_dir'] + "/trip.xml",
             '--start',
             '--quit-on-end',
             '--time-to-teleport', self.args_dict['TELEPORT_TIME'],
             '--no-warnings', 'True',
             '--no-step-log', 'True',
             '--seed', str(seed)])

        for ID in self.agentID:
            self.TlsDict[ID].initialization()
            self.prev_actionsDict[ID] = None

        # initialization
        self._step = 0
        self._sumo_step = 0
        self.done = False
        self.total_num_vehicles = 0
        self.reset_tls_detector()

    def measure_traffic_step_test(self):
        vehicles_Id = traci.vehicle.getIDList()
        num_tot_car = len(vehicles_Id)
        num_in_car = traci.simulation.getDepartedNumber()
        num_out_car = traci.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles_Id])
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in vehicles_Id])
        else:
            avg_speed = 0
            avg_waiting_time = 0

        # all trip-related measurements are not supported by traci,
        # need to read from output file afterwards
        queues = []
        for i in self.learning_agent_ID:
            for j in range(self.TlsDict[i]._num_incoming_lanes):
                queues.append(traci.lane.getLastStepHaltingNumber(self.TlsDict[i].lane_list[j]))

        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'episode': self.currEpisode,
                       'time_sec': self._sumo_step,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}

        self.traffic_data.append(cur_traffic)


if __name__ == '__main__':
    import os

    from arguments import set_args

    args_dict = set_args()

    os.chdir('../..')
    env = MATSC(server_number=0, args_dict=args_dict)
    env.reset()
    while True:
        action_dict = {}
        for i in range(25):
            ID = 'nt{}'.format(i + 1)
            action_dict[ID] = np.random.randint(0, 5, 1)[0]
        print('Current time : {}, Actions : {}'.format(traci.simulation.getTime(), action_dict))
        next_obs, reward, done, info = env.step(action_dict)
