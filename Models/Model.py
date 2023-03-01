import numpy as np

class Model:
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False,args_dict=None,env=None ):
        self.scope = scope
        self.trainer = trainer
        self.TRAINING = TRAINING
        self.GLOBAL_NETWORK = GLOBAL_NETWORK
        self.GLOBAL_NET_SCOPE = GLOBAL_NET_SCOPE
        self.args_dict = args_dict 
        self.env = env 

    def _build_net(self):
        pass

    def loss_function(self):
        pass

    def gradients(self):
        pass

    def forward(self,variables):
        pass

    def backward(self,buffer):
        pass

    def compute_all_intrinsic_rewards(self,buffer,sess,ep_length) : 
        raise NotImplementedError

    def reset(self,episode_number) : 
        raise NotImplementedError 