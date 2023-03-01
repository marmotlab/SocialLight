from Utils import Utilities
import numpy as np 

class Vanilla_Variables :

    def __init__(self,num_agents=25,metaAgentID=0,env=None,args_dict=None) :
        self.num_agents = num_agents
        self.ID = metaAgentID 
        self.env = env 
        self.args_dict = args_dict

    def reset(self,model=None) :
        self.bootstrapValue = self.dict_for_multiple('val')
        self.obs_dict       = self.dict_for_multiple('list')
        self.nextobs_dict   = self.dict_for_multiple('list') 
        self.actions_dict   = self.dict_for_multiple('val')
        self.rewards_dict   = self.dict_for_multiple('val')
        self.values_dict    = self.dict_for_multiple('val') 
        self.old_policy_dict = self.dict_for_multiple('list')

    def set_initial_value(self,observation) :
        for i in range(self.num_agents) :
            self.nextobs_dict[i] = observation[i]
        return 
        
    def step_update(self,rewards,output_dict,action_dict,new_observation) :
        for i in range(self.num_agents) :
            self.obs_dict[i] = self.nextobs_dict[i]
            self.rewards_dict[i] = rewards[Utilities.index_to_ID(i, self.env)]
            self.nextobs_dict[i] = new_observation[i] 
            self.actions_dict[i] = action_dict[i] 
            self.values_dict[i]  =  output_dict['values'][i] 
            self.old_policy_dict[i] = output_dict['actions'][i][0]
        
        return 
        
    def dict_for_multiple(self,d_type) :
        empty_dict = {} 
        if d_type == 'val':
            for i in range(self.num_agents) :
                empty_dict[i] = 0 
            return empty_dict
        elif d_type == 'list' :
            for i in range(self.num_agents) :
                empty_dict[i] = []  
            return empty_dict
        else : 
            pass 

class COMA_VariablesV3(Vanilla_Variables):
    def __init__(self, num_agents=25, metaAgentID=0, env=None, args_dict=None):
        super(COMA_VariablesV3, self).__init__()
        self.num_agents = num_agents
        self.ID = metaAgentID
        self.env = env
        self.args_dict = args_dict

    def reset(self, model=None):
        self.bootstrapValue = self.dict_for_multiple('val')
        self.obs_dict = self.dict_for_multiple('list')
        self.nextobs_dict = self.dict_for_multiple('list')
        self.actions_dict = self.dict_for_multiple('val')
        self.rewards_dict = self.dict_for_multiple('val')
        self.old_policy_dict = self.dict_for_multiple('list')
        self.neighbours_actions_dict = self.dict_for_multiple('list')
        self.neighbours_obs_dict =  self.dict_for_multiple('list')

    def set_initial_value(self, observation):
        for i in range(self.num_agents):
            self.nextobs_dict[i] = observation[i]
        return

    def step_update(self, rewards, output_dict, action_dict, new_observation):
        for i in range(self.num_agents):
            self.obs_dict[i] = self.nextobs_dict[i]
            self.rewards_dict[i] = rewards[Utilities.index_to_ID(i, self.env)]
            self.nextobs_dict[i] = new_observation[i]
            self.actions_dict[i] = action_dict[i]
            self.old_policy_dict[i] = output_dict['actions'][i][0]
        # Updating neighbour actions

        for i in range(self.num_agents):
            id = Utilities.index_to_ID(i, self.env)
            neighbours = self.env.neighbour_dict[id]
            actions = []
            observations = []
            for n in neighbours:
                if n is not None:
                    ind = self.env.id_to_index[n]
                    actions.append(self.actions_dict[ind])
                    observations.append(self.obs_dict[ind])
                else:
                    actions.append(self.args_dict['a_size'])
                    observations.append(-1*np.ones(self.obs_dict[i].shape))

            self.neighbours_actions_dict[i] = np.array(actions)
            self.neighbours_obs_dict[i] = np.stack(observations)
        return
