# from MATSC_gym.envs.GreedyAgent import GreedyAgent
from Utils import Utilities
import numpy as np 
from modules import setters
import tensorflow as tf 

# class Greedy(GreedyAgent):
class Greedy():
    def __init__(self,id,args_dict,env) :
        super().__init__(id)
        self.args_dict = args_dict
        self.env = env
        self.num_agents = self.args_dict['NUM_RL_AGENT']

    def reset(self) :
        pass

    def observe_all(self,rewards=None) :
        observations ={}
        for i in range(self.num_agents) :
            observations[i] = self.observe_single(i)
        return observations

    def observe_single(self,index) :
        ID = Utilities.index_to_ID(index)
        single_obs = np.reshape(self.env.TlsDict[ID].observe(),-1)
        return single_obs

    def step_all(self,obs) :
        actions_dict = {}
        for i in range(self.args_dict['NUM_TLS']):
            actions_dict[i] = super().get_action(observation=obs[i])
        return Utilities.convert_to_nt(actions_dict, args_dict=self.args_dict, env=self.env)

class DRL_Model() :

    def __init__(self,args_dict,env) : 
        self.args_dict = args_dict
        self.env = env 
        self.observer = setters.ObserverSetters.set_observer(args_dict=self.args_dict, metaAgentID=0, env=self.env)
        self.model = setters.ModelSetters.set_model(self.args_dict, trainer=tf.contrib.opt.NadamOptimizer(learning_rate=5e-4, use_locking=True),
                                                    training=False, GLOBAL_NETWORK=False,
                                                    input_shape= self.observer.shape, env=self.env,
                                                    dynamics_trainer=tf.contrib.opt.NadamOptimizer(learning_rate=5e-4, use_locking=True),
                                                    action_shapes = self.observer.action_spaces)
        self.variables = setters.VariableSetters.set_variables(self.args_dict, metaAgentID=0, env=self.env, num_agents=self.args_dict['NUM_RL_AGENT'])
        self.output_dict = None 
        self.tf_setup() 
        
    def tf_setup(self) :
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer()) 
        if self.args_dict['model_load'] :
            saver = tf.train.Saver(max_to_keep=1) 
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.args_dict['model_path'])
            p = ckpt.model_checkpoint_path
            print("Model Path: {}".format(p))
            p = p[p.find('-') + 1:]
            p = p[:p.find('.')]
            p = p[-10:]
            p = p[p.find('-') + 1:]
            curr_episode = int(p)

            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("curr_episode set to ", curr_episode)
            print('Loaded Model') 

    def reset(self) :
        self.variables.reset(self.model) 
        self.model.reset() 
        self.output_dict = None 

    def observe_all(self,rewards) :
        if self.output_dict is not None :
            observation = self.observer.observe_all(model_outputs=self.output_dict)  
            self.variables.step_update(rewards,self.output_dict,self.action_dict,observation)  
            return observation
        else :
            observation = self.observer.observe_all() 
            self.variables.set_initial_value(observation) 
            return observation

    def step_all(self,obs=None) : 
        self.output_dict = self.model.forward(self.sess,self.variables)
        best_action = True

        # For 5816, we want to sample from the action distribution
        # as there are multiple actions with the same value during the first few environemnt steps
        if '5816'  in self.args_dict['model_path']:
            best_action = False

        self.action_dict = Utilities.get_sampled_actions(self.output_dict['actions'], best=best_action)

        return Utilities.convert_to_nt(self.action_dict, self.args_dict, self.env)

