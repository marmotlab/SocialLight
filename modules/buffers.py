import os.path
import os
import numpy as np

class ExperienceBuffer :
    def __init__(self,variables,metaAgentID=0,num_agents=25) :
        self.ID = metaAgentID 
        self.num_agents = num_agents 
        self.vars   = variables
        self.reset()

    def reset(self):
        self.transitions = {} 
        for i in range(self.num_agents):
            self.transitions[i] = []
        return 

    def ground_truth_update(self) :
        pass 

    def bootstrap_update(self):
        pass

    def add_transition(self,data=None) :
        raise NotImplementedError

    def get_train_buffer(self, index):
        raise NotImplementedError
    
    def compute_combined_buffer(self) :
        for i in range(self.num_agents): 
            train_buffer = self.get_train_buffer(i) 
            if i ==0 :
                self.combined_buffer = train_buffer
            else :
                for k,v in train_buffer.items() :
                    self.combined_buffer[k] = np.concatenate((self.combined_buffer[k],v)) 
        
    def compute_combined_advantages(self,model,variables) :
        for i in range(self.num_agents):
            train_buffer = self.get_train_buffer(i) 
            agent_advantage_dict = model.get_advantages(train_buffer,variables.bootstrapValue[i])
            if i ==0 :
                self.advantage_buffer = agent_advantage_dict 
            else :
                for k,v in agent_advantage_dict.items() :
                    self.advantage_buffer[k] =np.concatenate((self.advantage_buffer[k],v)) 
    
    def sample_batch(self,size) :
        length = self.combined_buffer['observations'].shape[0] 
        indices = np.random.randint(0,length,size=size)   
        train_batch, advantage_batch = {} , {} 
        for k,v in self.combined_buffer.items() :
            train_batch[k] = self.combined_buffer[k][indices] 
        for k,v in self.advantage_buffer.items() :
            advantage_batch[k] = self.advantage_buffer[k][indices] 
        return train_batch,advantage_batch

class VanillaBuffer(ExperienceBuffer) :
    def __init__(self,variables,metaAgentID=0,num_agents=25) :
        super().__init__(variables,metaAgentID,num_agents) 
    
    def add_transition(self): 
        for i in range(self.num_agents): 
            self.transitions[i].append([self.vars.obs_dict[i],self.vars.actions_dict[i],self.vars.rewards_dict[i],self.vars.nextobs_dict[i],self.vars.values_dict[i],self.vars.old_policy_dict[i]]) 
        return 
    
    def get_train_buffer(self,index=0) :
        train_dict = {}
        rollout = np.array(self.transitions[index], dtype=object)  
        train_dict['observations']      = rollout[:,0]
        train_dict['actions']           = rollout[:,1]
        train_dict['rewards']           = rollout[:,2]
        train_dict['next_observations'] = rollout[:,3] 
        train_dict['values']            = rollout[:,4] 
        train_dict['old_policy']        = rollout[:, 5]
        return train_dict 


class COMABufferV3(ExperienceBuffer):
    def __init__(self, variables, metaAgentID=0, num_agents=25):
        super().__init__(variables, metaAgentID, num_agents)

    def add_transition(self):
        for i in range(self.num_agents):
            # list = [self.vars.obs_dict[i], self.vars.actions_dict[i], self.vars.rewards_dict[i], self.vars.nextobs_dict[i],
            #      self.vars.values_dict[i], self.vars.old_policy_dict[i],self.vars.neighbours_actions_dict[i],self.vars.qvalues_dict[i]]
            list = [self.vars.obs_dict[i], self.vars.actions_dict[i], self.vars.rewards_dict[i], self.vars.nextobs_dict[i],
                    self.vars.old_policy_dict[i],self.vars.neighbours_actions_dict[i],self.vars.neighbours_obs_dict[i]]
            if hasattr(self.vars,'target_values_dict'):
                list+=[self.vars.target_values_dict[i]]
            self.transitions[i].append(list)
        return

    def get_train_buffer(self, index=0):
        train_dict = {}
        rollout = np.array(self.transitions[index], dtype=object)
        train_dict['observations'] = rollout[:, 0]
        train_dict['actions'] = rollout[:, 1]
        train_dict['rewards'] = rollout[:, 2]
        train_dict['next_observations'] = rollout[:, 3]
        train_dict['old_policy'] = rollout[:, 4]
        train_dict['neighbours_actions'] = rollout[:,5]
        train_dict['neighbours_states'] = rollout[:,6]
        #train_dict['qvalues'] = rollout[:,7]
        if hasattr(self.vars,'target_values_dict'):
            train_dict['target_values'] = rollout[:,-1]
        return train_dict

    def get_dynamics_buffer(self):
        train_dict = {}
        for j, index in enumerate(self.transitions.keys()):
            rollout = np.array(self.transitions[index], dtype=object)
            if j == 0:
                train_dict['observations'] = rollout[:, 0]
                train_dict['next_observations'] = rollout[:, 3]
                train_dict['actions'] = rollout[:, 1]
                train_dict['neighbours_actions'] = rollout[:, 5]
                train_dict['neighbours_states'] = rollout[:, 6]
            else:
                train_dict['observations'] = np.concatenate((train_dict['observations'], \
                                                             rollout[:, 0]), axis=0)
                train_dict['next_observations'] = np.concatenate((train_dict['next_observations'], \
                                                                  rollout[:, 3]), axis=0)
                train_dict['actions'] = np.concatenate((train_dict['actions'], \
                                                        rollout[:, 1]), axis=0)
                train_dict['neighbours_actions'] = np.concatenate((train_dict['neighbours_actions'], \
                                                                   rollout[:, 5]), axis=0)
                train_dict['neighbours_states'] = rollout[:, 6]
        return train_dict

    def store_data(self,folder,ID):
        train_dict = self.get_dynamics_buffer()
        file = 'file{}.npy'.format(ID)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(folder+file,train_dict)