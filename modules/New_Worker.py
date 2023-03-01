from Utils import Utilities
import numpy as np 
import time 
import threading

class Worker:
    def __init__(self, workerID, workers_per_metaAgent, env, sess, groupLock, learningAgent,variables,buffer,metrics,model,observer,args_dict):

        self.ID = workerID
        self.workers_per_metaAgent = workers_per_metaAgent
        self.env = env
        self.groupLock = groupLock
        self.variables = variables 
        self.buffer= buffer
        self.model = model 
        self.metrics = metrics 
        self.observer = observer 
        self.args_dict = args_dict
        self.gradient = None  # gets filled in after running calculateGradient
        self.value_gradient = None
        self.learningAgent = learningAgent
        self.sess = sess
        self.saveGIF = False
        self.dynamics_buffer = None

    def reset(self,currEpisode) :
        # Set current episode 
        self.currEpisode = currEpisode

        # Everything about gifs 
        GIF_episode = None   
        self.env.save_images = False
        self.saveGIF = False
        if self.args_dict['OUTPUT_GIFS'] and (self.ID % self.args_dict['NUM_META_AGENTS'] == 0):
            print("making a gif")
            self.env.save_images = True
            self.saveGIF = True
            GIF_episode = int(self.currEpisode)

        # Resetting variables 
        self.variables.reset(self.model) 
        self.buffer.reset() 
        self.env.reset()
        self.metrics.reset() 
        self.model.reset(currEpisode) 
        np.random.seed() 
        return 0, GIF_episode

    def set_target_network(self,old_model):
        self.target_model = old_model

    def calculateGradient(self):
        self.gradient, self.dynamics_gradient = [], []
        self.value_gradient = []
        if self.args_dict['Target_Net']:
            self.model.update_target_bootstrap(self.variables.targetBootstrap)

        if self.args_dict['batch_gradient'] : 
            self.buffer.compute_combined_buffer() 
            self.buffer.compute_combined_advantages(model=self.model,variables=self.variables) 
            for j in range(self.args_dict['train_iterations']) :
                train_batch,advantage_batch = self.buffer.sample_batch(size=self.args_dict['batch_size']) 
                train_metrics,gradient = self.model.backward(0,train_batch,advantage_batch,self.sess) 
                self.gradient.append(gradient)
                self.metrics.update_batch_metrics(train_metrics,j) 
        else :
            # Train the network
            for i in range(self.args_dict['NUM_RL_AGENT']):
                train_buffer = self.buffer.get_train_buffer(i) 

                if self.args_dict['Target_Net']:
                    self.model.update_target_bootstrap(self.variables.targetBootstrap[i])

                if self.args_dict['ComputeValue']:
                    train_buffer['values'] = self.model.get_values(self.sess,train_buffer,i)
                advantage_dict = self.model.get_advantages(train_buffer,
                                                           self.variables.bootstrapValue[i])
                if self.args_dict['Duelling']:
                    train_metrics, gradient, value_gradient = self.model.backward(i,train_buffer,
                                                             advantage_dict,
                                                             self.sess)
                    self.value_gradient.append(value_gradient)
                else:
                    train_metrics, gradient, _ = self.model.backward(i,train_buffer,
                                                             advantage_dict,
                                                             self.sess)
                self.gradient.append(gradient)
                self.metrics.update_train_metrics(train_metrics,i)

        return self.metrics.total_train_metrics() 

    def calculateSingleGradient(self):
        self.gradient = []
        self.value_gradient = []
        train_buffer = self.buffer.get_train_buffer(None)
        if self.args_dict['Target_Net']:
            self.model.update_target_bootstrap(self.variables.targetBootstrap)

        advantage_dict = self.model.get_advantages(train_buffer,self.variables.bootstrapValue)
        if self.args_dict['Duelling']:
            train_metrics, gradient,value_gradient= self.model.backward(train_buffer, advantage_dict, self.sess)
            self.value_gradient.append(value_gradient)
        else:
            train_metrics, gradient,_ = self.model.backward(train_buffer, advantage_dict, self.sess)
        self.gradient.append(gradient)
        for i in range(self.args_dict['NUM_RL_AGENT']):
            self.metrics.update_train_metrics(train_metrics,i)

        return self.metrics.total_train_metrics()

    def run_episode_single_threaded(self, currEpisode):
        start_time = time.time() 
        episode_step, GIF_episode = self.reset(currEpisode) 

        # Take initial step
        _,_,_,_ = self.env.step(action_dict=Utilities.convert_to_nt(self.variables.actions_dict, self.args_dict, self.env))
        observation = self.observer.observe_all() 
        self.variables.set_initial_value(observation) 
        # Start running episode 
        while episode_step < self.args_dict['MAX_EPISODE_LENGTH']:
            
            output_dict = self.model.forward(self.sess,self.variables)
            if self.args_dict['Target_Net']:
                target_output_dict = self.target_model.forward(self.sess,self.variables)

            action_dict = Utilities.get_sampled_actions(output_dict['actions'])
            _, rewards, done, info = self.env.step(action_dict=Utilities.convert_to_nt(action_dict, self.args_dict, self.env))

            new_observation = self.observer.observe_all(model_outputs=output_dict)  

            self.variables.step_update(rewards,output_dict,action_dict,new_observation)
            if self.args_dict['Target_Net']:
                self.variables.target_values_update(target_output_dict)

            self.buffer.add_transition() 

            episode_step +=1 
    
            self.metrics.update_traffic_metrics(info, self.variables)
 
            if done or episode_step >= self.args_dict['MAX_EPISODE_LENGTH'] :
                self.buffer.bootstrap_update()
                self.variables.bootstrapValue = self.model.get_bootstrap(self.sess,self.variables)
                if self.args_dict['Target_Net']:
                    self.variables.targetBootstrap = self.target_model.get_bootstrap(self.sess,self.variables)
                break
        if self.saveGIF:
            Utilities.make_gif(self.env,
                     '{}/episode_{:d}_{:d}.mp4'.format(self.args_dict['gifs_path'], GIF_episode, episode_step))
    
        return self.metrics.episode_reward, self.metrics.episode_length, self.metrics.total_traffic_metrics() , time.time() -start_time 
    
    def work(self, currEpisode, coord, saver):
        self.coord = coord
        self.saver = saver
        perf_metrics, train_metrics, episode_reward, episode_length = None, None, None, None

        if self.args_dict['COMPUTE_TYPE'] == self.args_dict['COMPUTE_OPTIONS'].multiThreaded:
            pass
        elif self.args_dict['COMPUTE_TYPE'] == self.args_dict['COMPUTE_OPTIONS'].synchronous:
            episode_reward, episode_length, perf_metrics, time_taken = self.run_episode_single_threaded(currEpisode)
            print("MetaAgent {} | Episode {} | Reward: {} | Length: {} | Time Taken {} | Thread {}".format(self.ID,
                                                                                                           currEpisode,
                                                                                                           round(episode_reward, 3),
                                                                                                           episode_length,
                                                                                                           round(time_taken,2),
                                                                                                           threading.get_ident()))
        # fill up the experience buffer
        if self.args_dict['JOB_TYPE'] == self.args_dict['JOB_OPTIONS'].getExperience:
            # access output via self.experience_buffer
            self.episode_data = {'Traffic':perf_metrics ,
                                 'Perf':{'Reward':episode_reward,
                                         'Episode Length':episode_length,
                                 'action_change':perf_metrics['action_change']}}

            return
        elif self.args_dict['JOB_TYPE'] == self.args_dict['JOB_OPTIONS'].getGradient:
            # calculate gradients on network
            if self.args_dict['SINGLE_GRADIENT']:
                train_metrics = self.calculateSingleGradient()
            else:
                train_metrics = self.calculateGradient()

            self.episode_data = {'Traffic':perf_metrics ,
                                 'Losses':train_metrics ,
                                 'Perf':{'Reward':episode_reward,
                                         'Episode Length':episode_length,
                                'action_change':perf_metrics['action_change']}}



if __name__ == '__main__' :
    pass 