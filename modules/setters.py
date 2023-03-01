from modules.Metrics import VanillaTrafficMetrics, ComaTrafficMetrics,ComaTrafficMetricsV2
from modules.Variables_Initializer import Vanilla_Variables,COMA_VariablesV3
from modules.buffers import VanillaBuffer,COMABufferV3
from modules.Observers import TrafficObserver,SmallObserver,CityFlowV2Observer
import sys

class ModelSetters : 
    @staticmethod
    def set_model(args_dict,GLOBAL_NETWORK=None,trainer= None,training=None,input_shape=None,env=None,dynamics_trainer=None,action_shapes=None) :
        ModelSetters.check_errors(GLOBAL_NETWORK,trainer,training,input_shape,args_dict) 
        if args_dict['parameter_sharing'] :
            if args_dict['Model'] == 'ActorCritic':
                from Models.Models import VanillaTraffic
                return VanillaTraffic(scope=args_dict['GLOBAL_NET_SCOPE'],trainer= trainer, TRAINING=training, GLOBAL_NET_SCOPE= args_dict['GLOBAL_NET_SCOPE'], \
                    GLOBAL_NETWORK=GLOBAL_NETWORK,args_dict=args_dict, input_shape=input_shape,env=env,action_shapes=action_shapes)


            # TODO: Resolve the naming and fix the names of the models trained
            elif args_dict['Model'] == 'SocialLight':
                from Models.COMA import SocialLight
                #env = MATSC(server_number=0, args_dict=args_dict)
                return SocialLight(scope=args_dict['GLOBAL_NET_SCOPE'],trainer= trainer, TRAINING=training, GLOBAL_NET_SCOPE= args_dict['GLOBAL_NET_SCOPE'], \
                    GLOBAL_NETWORK=GLOBAL_NETWORK,args_dict=args_dict, input_shape=input_shape,env=env,action_shapes=action_shapes)

            elif args_dict['Model'] == 'DecCOMA':
                from Models.COMA import DecCOMA
                #env = MATSC(server_number=0, args_dict=args_dict)
                return DecCOMA(scope=args_dict['GLOBAL_NET_SCOPE'],trainer= trainer, TRAINING=training, GLOBAL_NET_SCOPE= args_dict['GLOBAL_NET_SCOPE'], \
                    GLOBAL_NETWORK=GLOBAL_NETWORK,args_dict=args_dict, input_shape=input_shape,env=env,action_shapes=action_shapes)

        else : 
            sys.exit('Model not found, please add model to setters')

    @staticmethod 
    def check_errors(GLOBAL_NETWORK,trainer,training,input_shape,args_dict) :
        if trainer is None and GLOBAL_NETWORK:
            sys.exit('Trainer should not be none')
        if GLOBAL_NETWORK is None:
            sys.exit('Global Map Boolean should not be none')
        if training is None :
            sys.exit('Training boolean should not be none')
        if input_shape is None :
            sys.exit('Input shape cannot be none')
        return 

class MetricSetters :
    @staticmethod
    def set_metrics(args_dict,metaAgentID=None, num_agents=25): 
        MetricSetters.check_errors(metaAgentID)
        if args_dict['Metric'] == 'VanillaTrafficMetrics':
            return VanillaTrafficMetrics(metaAgentID=metaAgentID,num_agents=num_agents,args_dict=args_dict)
        elif args_dict['Metric'] == 'ComaTrafficMetrics':
            return ComaTrafficMetrics(metaAgentID=metaAgentID,num_agents=num_agents,args_dict=args_dict)
        elif args_dict['Metric'] == 'ComaTrafficMetricsV2':
            return ComaTrafficMetricsV2(metaAgentID=metaAgentID,num_agents=num_agents,args_dict=args_dict)
        else :
            sys.exit('Metric class not found , please add to setters')

    @staticmethod
    def check_errors(metaAgentID):
        if metaAgentID is None:
            sys.exit('Please set meta-agent ID')

class VariableSetters : 
    @staticmethod
    def set_variables(args_dict,metaAgentID=None, num_agents=25,env=None): 
        VariableSetters.check_errors(metaAgentID)
        if args_dict['Variable'] == 'Vanilla_Variables':
            return Vanilla_Variables(metaAgentID=metaAgentID,num_agents=num_agents,env=env,args_dict=args_dict)
        elif args_dict['Variable'] == 'COMA_VariablesV3':
                return COMA_VariablesV3(metaAgentID=metaAgentID, num_agents=num_agents, env=env, args_dict=args_dict)

        else :
            sys.exit('Variable class not found , please add to setters')

    @staticmethod
    def check_errors(metaAgentID):
        if metaAgentID is None:
            sys.exit('Please set meta-agent ID')

class BufferSetters : 
    @staticmethod
    def set_buffer(args_dict,metaAgentID=None, num_agents=25,variables=None,input_shape=180): 
        BufferSetters.check_errors(metaAgentID,variables)
        if args_dict['Buffer'] == 'VanillaBuffer':
            return VanillaBuffer(metaAgentID=metaAgentID,num_agents=num_agents,variables=variables)
        elif args_dict['Buffer'] == 'COMABufferV3':
            return COMABufferV3(metaAgentID=metaAgentID, num_agents=num_agents, variables=variables)
        else :
            sys.exit('Buffer class not found , please add to setters')

    @staticmethod
    def check_errors(metaAgentID,variables):
        if metaAgentID is None:
            sys.exit('Please set meta-agent ID')
        if variables is None : 
            sys.exit('Please pass in the variables class')

class ObserverSetters : 
    @staticmethod
    def set_observer(args_dict,metaAgentID=None, num_agents=25,env=None,dummy=False): 
        ObserverSetters.check_errors(metaAgentID,env,dummy,args_dict)
        if args_dict['Observer'] == 'TrafficObserver':
            return TrafficObserver(metaAgentID=metaAgentID,num_agents=num_agents,env=env,append_neighbour=args_dict['NEIGH_OBS'],args_dict=args_dict)
        elif args_dict['Observer'] == 'SmallObserver':
            return SmallObserver(metaAgentID=metaAgentID,num_agents=num_agents,env=env,append_neighbour=args_dict['NEIGH_OBS'],args_dict=args_dict)
        elif args_dict['Observer'] == 'CityFlowV2Observer':
            return CityFlowV2Observer(metaAgentID=metaAgentID, env=env, append_neighbour=args_dict['NEIGH_OBS'], args_dict=args_dict)
        else :
            sys.exit('Observer class not found , please add to setters')

    def check_errors(metaAgentID,env,dummy,args_dict):
        if not dummy: 
            if metaAgentID is None:
                sys.exit('Please set meta-agent ID')
            if env is None : 
                sys.exit('Please pass in the environment')
        if args_dict['share_horizon'] == True and args_dict['Model'] != 'LongHorizon':
            sys.exit('Incompatible model,not possible to share horizon unless model is LongHorizon')

# TODO: fix the name of the networks here
class NetworkSetters : 
    @staticmethod    
    def set_network_class(args_dict,input_shape,output_shape,scope,trainer,n_action_shapes = None) :
        from modules.Networks import vanilla_fc,custom_mlp,coma_mlp
        if args_dict['Network'] == 'vanilla_fc':
            return vanilla_fc(args_dict,input_shape,output_shape,scope,trainer) 
        elif args_dict['Network'] == 'custom_mlp':
            return custom_mlp(args_dict,input_shape,output_shape,scope,trainer)
        elif args_dict['Network'] == 'coma_mlp' :
            return coma_mlp(args_dict,input_shape,output_shape,scope,trainer)

    @staticmethod
    def check_errors(model):
        if model.args_dict['Model'] == 'VanillaTraffic':
            if model.args_dict['Map'] not in ['vanilla_fc', 'custom_mlp'] :
                sys.exit('Model and network not compatible')
        if model.args_dict['Model'] == 'SocialLight' or model.args_dict['Model']=='DecCOMA':
            if model.args_dict['Map'] not in ['coma_mlp'] :
                sys.exit('Model and network not compatible')
        else : 
            pass 

class EnvSetters : 
    @staticmethod
    def set_environment(args_dict,id): 
        EnvSetters.check_errors(args_dict)
        if args_dict['Environment'] == 'Manhattan':
            from MATSC_gym.envs.Manhattan_MATSC import MATSC as Manhattan
            return Manhattan(server_number=id, args_dict=args_dict)
        elif args_dict['Environment'] == 'CityFlowV2':
            from MATSC_gym.envs.CityFlow_MATSC_V2 import MATSC as CityFlow
            return CityFlow(server_number=id, args_dict=args_dict)
        else :
            sys.exit('Environment not found , please add to setters')

    @staticmethod
    def check_errors(args_dict): 
        pass 

#TODO: Maybe not needed
class ReplayBufferSetters :
    @staticmethod
    def set_replaybuffer(args_dict,data_spec,capacity=10000):
        EnvSetters.check_errors(args_dict)
        if args_dict['ReplayBuffer'] == 'ReplayBuffer':
            from modules.replay_buffer import ReplayBuffer
            return ReplayBuffer(data_spec,capacity)
        elif args_dict['ReplayBuffer'] == 'MACPGDynReplayBuffer':
            from modules.replay_buffer import MACPGDynReplayBuffer
            return MACPGDynReplayBuffer(data_spec=data_spec,capacity=capacity)
        else :
            sys.exit('Environment not found , please add to setters')

    @staticmethod
    def check_errors(args_dict):
        pass
if __name__ == '__main__' :
    pass 