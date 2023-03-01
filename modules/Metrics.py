import numpy as np  

class VanillaTrafficMetrics() :
    def __init__(self,metaAgentID=0,num_agents=25,args_dict=None) :
        self.ID = metaAgentID 
        self.num_agents = num_agents
        self.args_dict = args_dict
        self.train_metrics_size = num_agents 
        if self.args_dict['batch_gradient'] :
            self.train_metrics_size = self.args_dict['train_iterations']

    def reset(self) :
        self.initialize_traffic_metrics() 
        self.initialize_train_metrics() 
        
    def initialize_traffic_metrics(self) :
        self.episode_reward     = 0 
        self.episode_length     = 0
        self.traffic_metrics = {'action_change': 0 , 'Ave_waiting':[] , 'Ave_speed': [] , 'Std_queue':[] , 'Avg_queue':[] }
        
    def initialize_train_metrics(self) : 
        # self.train_metrics = {'Value Loss':np.zeros(self.train_metrics_size) ,
        #                       'Policy Loss':np.zeros(self.train_metrics_size) ,
        #                       'Entropy Loss':np.zeros(self.train_metrics_size) ,
        #                       'Grad Norm':np.zeros(self.train_metrics_size) ,
        #                       'Var Norm':np.zeros(self.train_metrics_size)}

        self.train_metrics = {'Value Loss':[None] * self.train_metrics_size,
                              'Policy Loss':[None] * self.train_metrics_size,
                              'Entropy Loss':[None] * self.train_metrics_size,
                              'Grad Norm':[None] * self.train_metrics_size,
                              'Var Norm':[None] * self.train_metrics_size}


    def update_traffic_metrics(self,traffic_data,variables=None) :
        if traffic_data[2] is not None :
            self.traffic_metrics['Ave_waiting'].append(traffic_data[2]['avg_wait_sec'])
            self.traffic_metrics['Ave_speed'].append(traffic_data[2]['avg_speed_mps'])
            self.traffic_metrics['Std_queue'].append(traffic_data[2]['std_queue'])
            self.traffic_metrics['Avg_queue'].append(traffic_data[2]['avg_queue'])

        self.episode_reward += traffic_data[0] 
        self.traffic_metrics['action_change'] += traffic_data[1] 
        self.episode_length +=1

    def total_traffic_metrics(self) : 
        if self.traffic_metrics['Ave_speed'] == [] :
            self.traffic_metrics['Ave_waiting'] = 0
            self.traffic_metrics['Ave_speed'] = 0
            self.traffic_metrics['Std_queue'] = 0
            self.traffic_metrics['Avg_queue'] = 0
        else :
            self.traffic_metrics['Ave_waiting'] = np.mean(self.traffic_metrics['Ave_waiting'])
            self.traffic_metrics['Ave_speed'] = np.nanmean(self.traffic_metrics['Ave_speed']) 
            self.traffic_metrics['Std_queue'] = np.nanmean(self.traffic_metrics['Std_queue']) 
            self.traffic_metrics['Avg_queue'] = np.nanmean(self.traffic_metrics['Avg_queue']) 

        self.traffic_metrics['action_change'] = self.traffic_metrics['action_change'] / self.episode_length 
        return self.traffic_metrics
        
    def update_train_metrics(self,train_data,index) :
        self.train_metrics['Policy Loss'][index] = train_data['Policy Loss'] /self.episode_length
        self.train_metrics['Value Loss'][index] = train_data['Value Loss'] /self.episode_length
        self.train_metrics['Entropy Loss'][index] = train_data['Entropy Loss'] /self.episode_length
        self.train_metrics['Grad Norm'][index] = train_data['Grad Norm'] 
        self.train_metrics['Var Norm'][index] = train_data['Var Norm'] 

    def update_batch_metrics(self,train_data,iter) :
        for k,v in self.train_metrics.items() :
            if k in ['Policy Loss', 'Value Loss', 'Entropy Loss'] : 
                self.train_metrics[k][iter] = train_data[k]/self.args_dict['batch_size'] 
            else :
                self.train_metrics[k][iter] = train_data[k]
    
    def total_train_metrics(self) : 
        self.train_metrics['Policy Loss'] = np.nanmean(self.train_metrics['Policy Loss'],axis=0)
        self.train_metrics['Value Loss'] = np.nanmean(self.train_metrics['Value Loss'],axis=0)
        self.train_metrics['Entropy Loss'] = np.nanmean(self.train_metrics['Entropy Loss'],axis=0)
        self.train_metrics['Grad Norm'] = np.nanmean(self.train_metrics['Grad Norm'],axis=0)
        self.train_metrics['Var Norm'] = np.nanmean(self.train_metrics['Var Norm'],axis=0)
        return self.train_metrics


class ComaTrafficMetrics(VanillaTrafficMetrics):
    def __init__(self, metaAgentID=0, num_agents=25, args_dict=None):
        super().__init__(metaAgentID,num_agents,args_dict)

    def initialize_train_metrics(self):
        self.train_metrics = {'Value Loss': [None] * self.train_metrics_size,
                              'Policy Loss Team': [None] * self.train_metrics_size,
                              'Policy Loss Indv': [None] * self.train_metrics_size,
                              'Entropy Loss': [None] * self.train_metrics_size,
                              'Grad Norm': [None] * self.train_metrics_size,
                              'Q Value Loss':[None]*self.train_metrics_size,
                              'Var Norm': [None] * self.train_metrics_size}

    def update_train_metrics(self, train_data, index):
        self.train_metrics['Policy Loss Team'][index] = train_data['Policy Loss Team'] / self.episode_length
        self.train_metrics['Policy Loss Indv'][index] = train_data['Policy Loss Indv'] / self.episode_length
        self.train_metrics['Value Loss'][index] = train_data['Value Loss'] / self.episode_length
        self.train_metrics['Entropy Loss'][index] = train_data['Entropy Loss'] / self.episode_length
        self.train_metrics['Grad Norm'][index] = train_data['Grad Norm']
        self.train_metrics['Var Norm'][index] = train_data['Var Norm']
        self.train_metrics['Q Value Loss'][index] = train_data['Q Value Loss']/ self.episode_length


    def total_train_metrics(self):
        self.train_metrics['Policy Loss Team'] = np.nanmean(self.train_metrics['Policy Loss Team'], axis=0)
        self.train_metrics['Policy Loss Indv'] = np.nanmean(self.train_metrics['Policy Loss Indv'], axis=0)
        self.train_metrics['Value Loss'] = np.nanmean(self.train_metrics['Value Loss'], axis=0)
        self.train_metrics['Entropy Loss'] = np.nanmean(self.train_metrics['Entropy Loss'], axis=0)
        self.train_metrics['Grad Norm'] = np.nanmean(self.train_metrics['Grad Norm'], axis=0)
        self.train_metrics['Var Norm'] = np.nanmean(self.train_metrics['Var Norm'], axis=0)
        self.train_metrics['Q Value Loss'] = np.nanmean(self.train_metrics['Q Value Loss'], axis=0)
        return self.train_metrics

class ComaTrafficMetricsV2(VanillaTrafficMetrics):
    def __init__(self, metaAgentID=0, num_agents=25, args_dict=None):
        super().__init__(metaAgentID,num_agents,args_dict)

    def initialize_train_metrics(self):
        self.train_metrics = {'Value Loss': [None] * self.train_metrics_size,
                              'Policy Loss Indv': [None] * self.train_metrics_size,
                              'Entropy Loss': [None] * self.train_metrics_size,
                              'Grad Norm': [None] * self.train_metrics_size,
                              'Var Norm': [None] * self.train_metrics_size}

    def update_train_metrics(self, train_data, index):
        self.train_metrics['Policy Loss Indv'][index] = train_data['Policy Loss Indv'] / self.episode_length
        self.train_metrics['Value Loss'][index] = train_data['Value Loss'] / self.episode_length
        self.train_metrics['Entropy Loss'][index] = train_data['Entropy Loss'] / self.episode_length
        self.train_metrics['Grad Norm'][index] = train_data['Grad Norm']
        self.train_metrics['Var Norm'][index] = train_data['Var Norm']



    def total_train_metrics(self):
        self.train_metrics['Policy Loss Indv'] = np.nanmean(self.train_metrics['Policy Loss Indv'], axis=0)
        self.train_metrics['Value Loss'] = np.nanmean(self.train_metrics['Value Loss'], axis=0)
        self.train_metrics['Entropy Loss'] = np.nanmean(self.train_metrics['Entropy Loss'], axis=0)
        self.train_metrics['Grad Norm'] = np.nanmean(self.train_metrics['Grad Norm'], axis=0)
        self.train_metrics['Var Norm'] = np.nanmean(self.train_metrics['Var Norm'], axis=0)
        return self.train_metrics

