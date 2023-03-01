import tensorflow as tf
from Utils import Utilities
import numpy as np
from modules import setters
import copy

from Models.Model import Model

class ActorCritic(Model):

    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):

        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, env)
        assert Utilities.check_shape_equality(input_shape), 'Parameters cannot be shared, input shapes differ'
        assert len(set(list(action_shapes.values()))) == 1, 'Parameters cannot be shared, output shapes differ'
        input_shape = list(input_shape.values())[0]
        self.a_size = list(action_shapes.values())[0]
        self.num_agents = self.args_dict['NUM_RL_AGENT']
        self.initialize(input_shape)

    def reset(self, episode_number=None):
        self.episode_number = episode_number

    def initialize(self, input_shape):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        self.input_shape = [None]
        self.input_shape.extend(input_shape)
        self.hidden_sizes = self.args_dict['hidden_sizes']
        with tf.variable_scope(str(self.scope) + '/qvalues'):
            self.network = self.make_net()
            if self.TRAINING:
                self.network.set_loss()
                self.network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.network.set_global()
        self.networks = [self.network]

    def make_net(self):
        return setters.NetworkSetters.set_network_class(self.args_dict, self.input_shape, self.a_size, self.scope,
                                                        self.trainer)

    def get_advantages(self, train_buffer, bootstrapValue):
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"

        self.rewards_plus = np.asarray(train_buffer['rewards'].tolist() + [bootstrapValue])
        discounted_rewards = Utilities.discount(self.rewards_plus, self.args_dict['gamma'])[:-1]
        self.value_plus = np.asarray(train_buffer['values'].tolist() + [bootstrapValue])

        # vectorized version of below comment
        # gamma * self.value_plus[i+1] - self.value_plus[i]
        advantages = train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:] - self.value_plus[:-1]
        advantages = Utilities.discount(advantages, self.args_dict['gamma'])

        train_dict = {'advantages': advantages, 'discounted_rewards': discounted_rewards}
        return train_dict

    def forward(self, sess, variables):
        vars_required = ['actions', 'values']
        output_dict = Utilities.get_forward_dict(self.num_agents, vars_required)
        for i in range(self.num_agents):
            return_dict = self.forward_step(sess, variables, i)
            output_dict['actions'][i] = return_dict['action']
            output_dict['values'][i] = return_dict['value'][0, 0]
        return output_dict

    def forward_step(self, sess, variables, index):
        return_dict = {}
        return_dict['action'], return_dict['value'] = sess.run([self.network.policy, self.network.value],
                                                               feed_dict={self.network.inputs: [
                                                                   variables.nextobs_dict[index]]})
        return return_dict

    def get_bootstrap(self, sess, variables):
        outputs = self.forward(sess, variables)
        return outputs['values']




    def backward(self, index, train_buffer, advantage_dict, sess):
        if not self.args_dict['GLOBAL_PPO']:
            feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                         self.network.inputs: np.stack(train_buffer['observations']),
                         self.network.actions: train_buffer['actions'],
                         self.network.advantages: advantage_dict['advantages']}

            if self.args_dict['PPO']:
                feed_dict[self.network.old_policy] = np.stack(train_buffer['old_policy'])

            v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                [self.network.value_loss,
                 self.network.policy_loss,
                 self.network.entropy,
                 self.network.grad_norms,
                 self.network.var_norms,
                 self.network.gradients],
                feed_dict=feed_dict)

            train_metrics = {'Value Loss': v_l,
                             'Policy Loss': p_l,
                             'Entropy Loss': e_l,
                             'Grad Norm': g_n,
                             'Var Norm': v_n}
            return train_metrics, gradient

        else:
            # global ppo
            assert self.args_dict['PPO'] == True
            v_l_list, p_l_list, e_l_list, g_n_list, v_n_list = [], [], [], [], []
            for i in range(self.args_dict['N_EPOCH']):
                feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                             self.network.inputs: np.stack(train_buffer['observations']),
                             self.network.actions: train_buffer['actions'],
                             self.network.advantages: advantage_dict['advantages'],
                             self.network.old_policy: np.stack(train_buffer['old_policy'])}

                v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                    [self.network.value_loss,
                     self.network.policy_loss,
                     self.network.entropy,
                     self.network.grad_norms,
                     self.network.var_norms,
                     self.network.gradients],
                    feed_dict=feed_dict)

                # update network
                feed_dict = {
                    self.network.tempGradients[i]: g for i, g in enumerate(gradient)
                }
                sess.run([self.network.apply_grads], feed_dict=feed_dict)

                v_l_list.append(v_l), \
                p_l_list.append(p_l), \
                e_l_list.append(e_l), \
                g_n_list.append(g_n), \
                v_n_list.append(v_n)

            train_metrics = {'Value Loss': np.mean(np.array(v_l_list)),
                             'Policy Loss': np.mean(np.array(p_l_list)),
                             'Entropy Loss': np.mean(np.array(e_l_list)),
                             'Grad Norm': np.mean(np.array(g_n_list)),
                             'Var Norm': np.mean(np.array(v_n_list))}

        return train_metrics, None

class ActorCriticQVal(ActorCritic):
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def forward(self, sess, variables):
        vars_required = ['actions', 'values']
        output_dict = Utilities.get_forward_dict(self.num_agents, vars_required)
        for i in range(self.num_agents):
            return_dict = self.forward_step(sess, variables, i)
            output_dict['actions'][i] = return_dict['action']
            output_dict['values'][i] = return_dict['value'][0]
            #output_dict['qvalues'][i] = return_dict['qvalue'][0,0]
        return output_dict

    def forward_step(self, sess, variables, index):
        return_dict = {}
        return_dict['action'], return_dict['value']= sess.run([self.network.policy, self.network.qvalue],
                                                               feed_dict={self.network.inputs: [
                                                                   variables.nextobs_dict[index]]})
        return return_dict

    def initialize(self, input_shape):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        self.input_shape = [None]
        self.input_shape.extend(input_shape)
        self.hidden_sizes = self.args_dict['hidden_sizes']

        with tf.variable_scope(str(self.scope) + '/qvalues'):
            self.network = self.make_net()
            if self.TRAINING:
                self.network.set_loss()
                self.network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.network.set_global()
        self.networks = [self.network]

    def update_target_bootstrap(self,bootstrap):
        self.target_bootstrap = bootstrap

    def get_advantages(self, train_buffer, bootstrapValue):
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"


        self.rewards_plus = np.asarray(train_buffer['rewards'].tolist() + [np.max(bootstrapValue)])
        discounted_rewards = Utilities.discount(self.rewards_plus, self.args_dict['gamma'])[:-1]

        self.qvalues = train_buffer['values']

        self.policy = np.asarray(train_buffer['old_policy'])
        self.values = np.sum(np.stack(self.qvalues*self.policy),axis =1)
        self.value_plus = np.asarray(self.values.tolist() + [np.max(bootstrapValue)])

        if self.args_dict['Target_Net']:
            self.targetqvalues = train_buffer['target_values']
            self.targetpolicy = np.asarray(train_buffer['policy'])
            self.targetvalues = np.sum(np.stack(self.targetqvalues * self.targetpolicy), axis=1)
            self.targetvalues_plus = np.asarray(self.targetvalues.tolist() + [np.max(self.target_bootstrap)])
        # vectorized version of below comment
        # gamma * self.value_plus[i+1] - self.value_plus[i]
        advantages = train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:] - self.value_plus[:-1]
        advantages = Utilities.discount(advantages, self.args_dict['gamma'])
        advantages2 = np.stack(train_buffer['values'])[np.arange(self.values.shape[0]).tolist(),train_buffer['actions'].tolist()] - self.value_plus[:-1]
        advantages2 = Utilities.discount(advantages2,self.args_dict['gamma'])

        #TODO: must be a general discountd lambda return, this is too hacky
        if self.args_dict['Target_Net']:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.target_value_plus[1:])+ \
                          (1 - self.args_dict['beta']) * discounted_rewards
        else:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:]) + \
                          (1 - self.args_dict['beta']) * discounted_rewards

        train_dict = {'advantages': advantages,'qval_target': qval_target, 'discounted_rewards': discounted_rewards}
        return train_dict

    def backward(self, index, train_buffer, advantage_dict, sess):

        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.indv_advantages: advantage_dict['advantages'],
                     self.network.target_qval: advantage_dict['qval_target'],
                     self.network.neighbour_actions: np.stack(train_buffer['neighbours_actions'])}


        v_l, p_li, e_l, g_n, v_n, gradient= sess.run([self.network.qvalue_loss,
                                                      self.network.policy_loss_indv,
                                                      self.network.entropy,
                                                      self.network.grad_norms,
                                                      self.network.var_norms,
                                                      self.network.gradients],
                                                     feed_dict=feed_dict)
        value_gradient = None
        # For debugging
        # v_l, p_lt, p_li, qv_l, e_l, g_n, v_n, gradient,adv,advt,qval,val = sess.run([self.network.value_loss,
        #                                                            self.network.policy_loss_team,
        #                                                            self.network.policy_loss_indv,
        #                                                            self.network.qvalue_loss,
        #                                                            self.network.entropy,
        #                                                            self.network.grad_norms,
        #                                                            self.network.var_norms,
        #                                                            self.network.gradients,
        #                                                            self.network.advantages,
        #                                                            self.network.advantages_team,
        #                                                            self.network.qvalue,
        #                                                            self.network.value
        #                                                            ],
        #                                                           feed_dict=feed_dict)
        train_metrics = {'Value Loss': v_l, 'Policy Loss Indv':p_li, 'Entropy Loss': e_l, 'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient,value_gradient


class SocialLight(ActorCriticQVal):
    '''
    SocialLight with the original training setting
    '''
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def forward(self, sess, variables):
        vars_required = ['actions', 'values']
        output_dict = Utilities.get_forward_dict(self.num_agents, vars_required)
        for i in range(self.num_agents):
            return_dict = self.forward_step(sess, variables, i)
            output_dict['actions'][i] = return_dict['action']
            #output_dict['qvalues'][i] = return_dict['qvalue'][0,0]
        return output_dict

    def forward_step(self, sess, variables, index):
        return_dict = {}
        return_dict['action']= sess.run(self.network.policy,feed_dict={self.network.inputs: [
                                                                   variables.nextobs_dict[index]]})
        return return_dict

    def get_values(self,sess,train_buffer,index = None):
        return_dict = sess.run(self.network.qvalue, feed_dict={self.network.inputs:np.stack(train_buffer['observations']),
                                                                           self.network.neighbour_actions: np.stack(train_buffer['neighbours_actions'])})
        return return_dict

    def initialize(self, input_shape):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        self.input_shape = [None]
        self.input_shape.extend(input_shape)
        self.hidden_sizes = self.args_dict['hidden_sizes']

        with tf.variable_scope(str(self.scope) + '/qvalues'):
            self.network = self.make_net()
            if self.TRAINING:
                self.network.set_loss()
                self.network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.network.set_global()
        self.networks = [self.network]

    def update_target_bootstrap(self,bootstrap):
        self.target_bootstrap = bootstrap

    def get_advantages(self, train_buffer, bootstrapValue):
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"


        self.rewards_plus = np.asarray(train_buffer['rewards'].tolist() + [np.max(bootstrapValue)])
        discounted_rewards = Utilities.discount(self.rewards_plus, self.args_dict['gamma'])[:-1]

        self.qvalues = train_buffer['values']

        self.policy = np.stack(train_buffer['old_policy'])
        self.values = np.sum(np.stack(self.qvalues*self.policy),axis =1)
        self.value_plus = np.asarray(self.values.tolist() + [bootstrapValue])

        if self.args_dict['Target_Net']:
            self.targetqvalues = train_buffer['target_values']
            self.targetpolicy = np.asarray(train_buffer['policy'])
            self.targetvalues = np.sum(np.stack(self.targetqvalues * self.targetpolicy), axis=1)
            self.targetvalues_plus = np.asarray(self.targetvalues.tolist() + [np.max(self.target_bootstrap)])
        # vectorized version of below comment
        # gamma * self.value_plus[i+1] - self.value_plus[i]
        advantages = train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:] - self.value_plus[:-1]
        advantages = Utilities.discount(advantages, self.args_dict['gamma'])
        #TODO: Replace with lambda return

        if self.args_dict['Target_Net']:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.target_value_plus[1:])+ \
                          (1 - self.args_dict['beta']) * discounted_rewards
        else:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:]) + \
                          (1 - self.args_dict['beta']) * discounted_rewards

        if self.args_dict['LAMBDA']:
            if self.args_dict['Target_Net']:
                qval_target = Utilities.lambda_return(train_buffer['rewards'],\
                                                      self.targetvalues_plus,self.args_dict['gamma'],\
                                                      self.args_dict['beta'])
            else:
                qval_target = Utilities.lambda_return(train_buffer['rewards'], \
                                                      self.value_plus, self.args_dict['gamma'], \
                                                      self.args_dict['beta'])

        train_dict = {'advantages': advantages,'qval_target': qval_target, 'discounted_rewards': discounted_rewards}
        return train_dict

    def backward(self, index, train_buffer, advantage_dict, sess):

        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.indv_advantages: advantage_dict['advantages'],
                     self.network.target_qval: advantage_dict['qval_target'],
                     self.network.neighbour_actions: np.stack(train_buffer['neighbours_actions'])}


        v_l, p_li, e_l, g_n, v_n, gradient= sess.run([self.network.qvalue_loss,
                                                      self.network.policy_loss_indv,
                                                      self.network.entropy,
                                                      self.network.grad_norms,
                                                      self.network.var_norms,
                                                      self.network.gradients],
                                                     feed_dict=feed_dict)
        value_gradient = None
        # For debugging
        # v_l, p_lt, p_li, qv_l, e_l, g_n, v_n, gradient,adv,advt,qval,val = sess.run([self.network.value_loss,
        #                                                            self.network.policy_loss_team,
        #                                                            self.network.policy_loss_indv,
        #                                                            self.network.qvalue_loss,
        #                                                            self.network.entropy,
        #                                                            self.network.grad_norms,
        #                                                            self.network.var_norms,
        #                                                            self.network.gradients,
        #                                                            self.network.advantages,
        #                                                            self.network.advantages_team,
        #                                                            self.network.qvalue,
        #                                                            self.network.value
        #                                                            ],
        #                                                           feed_dict=feed_dict)
        train_metrics = {'Value Loss': v_l, 'Policy Loss Indv':p_li, 'Entropy Loss': e_l, 'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient,value_gradient





class DecCOMA(ActorCriticQVal):
    '''
        Original COMA training in a fuly decentralised setting
    '''
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def forward(self, sess, variables):
        vars_required = ['actions', 'values']
        output_dict = Utilities.get_forward_dict(self.num_agents, vars_required)
        for i in range(self.num_agents):
            return_dict = self.forward_step(sess, variables, i)
            output_dict['actions'][i] = return_dict['action']
            #output_dict['qvalues'][i] = return_dict['qvalue'][0,0]
        return output_dict

    def forward_step(self, sess, variables, index):
        return_dict = {}
        return_dict['action']= sess.run(self.network.policy,feed_dict={self.network.inputs: [
                                                                   variables.nextobs_dict[index]]})
        return return_dict

    def get_values(self,sess,train_buffer,index = None):
        return_dict = sess.run(self.network.qvalue, feed_dict={self.network.inputs:np.stack(train_buffer['observations']),
                                                                           self.network.neighbour_actions: np.stack(train_buffer['neighbours_actions'])})
        return return_dict

    def initialize(self, input_shape):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        self.input_shape = [None]
        self.input_shape.extend(input_shape)
        self.hidden_sizes = self.args_dict['hidden_sizes']

        with tf.variable_scope(str(self.scope) + '/qvalues'):
            self.network = self.make_net()
            if self.TRAINING:
                self.network.set_loss()
                self.network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.network.set_global()
        self.networks = [self.network]

    def update_target_bootstrap(self,bootstrap):
        self.target_bootstrap = bootstrap

    def get_advantages(self, train_buffer, bootstrapValue):
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"


        self.rewards_plus = np.asarray(train_buffer['rewards'].tolist() + [np.max(bootstrapValue)])
        discounted_rewards = Utilities.discount(self.rewards_plus, self.args_dict['gamma'])[:-1]

        self.qvalues = train_buffer['values']

        self.policy = np.stack(train_buffer['old_policy'])
        self.values = np.sum(np.stack(self.qvalues*self.policy),axis =1)
        self.value_plus = np.asarray(self.values.tolist() + [bootstrapValue])

        self.values_coma1 = self.qvalues[np.arange(self.values.shape[0]).tolist(),train_buffer['actions'].tolist()]
        self.values_coma1plus = np.asarray(self.values_coma1.tolist() + [bootstrapValue])


        if self.args_dict['Target_Net']:
            self.targetqvalues = train_buffer['target_values']
            self.targetpolicy = np.asarray(train_buffer['policy'])
            self.targetvalues = np.sum(np.stack(self.targetqvalues * self.targetpolicy), axis=1)
            self.targetvalues_plus = np.asarray(self.targetvalues.tolist() + [np.max(self.target_bootstrap)])
        # vectorized version of below comment
        # gamma * self.value_plus[i+1] - self.value_plus[i]
        advantages2 = np.stack(train_buffer['values'])[np.arange(self.values.shape[0]).tolist(),train_buffer['actions'].tolist()] - self.value_plus[:-1]
        #TODO: Replace with lambda return

        if self.args_dict['Target_Net']:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.target_value_plus[1:])+ \
                          (1 - self.args_dict['beta']) * discounted_rewards
        else:
            qval_target = self.args_dict['beta']*(train_buffer['rewards'] + self.args_dict['gamma'] * self.value_plus[1:]) + \
                          (1 - self.args_dict['beta']) * discounted_rewards

        if self.args_dict['LAMBDA']:
            if self.args_dict['Target_Net']:
                qval_target = Utilities.lambda_return(train_buffer['rewards'],\
                                                      self.targetvalues_plus,self.args_dict['gamma'],\
                                                      self.args_dict['beta'])
            else:
                qval_target = Utilities.lambda_return(train_buffer['rewards'], \
                                                      self.values_coma1plus, self.args_dict['gamma'], \
                                                      self.args_dict['beta'])

        train_dict = {'advantages': advantages2,'qval_target': qval_target, 'discounted_rewards': discounted_rewards}
        return train_dict

    def backward(self, index, train_buffer, advantage_dict, sess):

        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.indv_advantages: advantage_dict['advantages'],
                     self.network.target_qval: advantage_dict['qval_target'],
                     self.network.neighbour_actions: np.stack(train_buffer['neighbours_actions'])}


        v_l, p_li, e_l, g_n, v_n, gradient= sess.run([self.network.qvalue_loss,
                                                      self.network.policy_loss_indv,
                                                      self.network.entropy,
                                                      self.network.grad_norms,
                                                      self.network.var_norms,
                                                      self.network.gradients],
                                                     feed_dict=feed_dict)
        value_gradient = None
        # For debugging
        # v_l, p_lt, p_li, qv_l, e_l, g_n, v_n, gradient,adv,advt,qval,val = sess.run([self.network.value_loss,
        #                                                            self.network.policy_loss_team,
        #                                                            self.network.policy_loss_indv,
        #                                                            self.network.qvalue_loss,
        #                                                            self.network.entropy,
        #                                                            self.network.grad_norms,
        #                                                            self.network.var_norms,
        #                                                            self.network.gradients,
        #                                                            self.network.advantages,
        #                                                            self.network.advantages_team,
        #                                                            self.network.qvalue,
        #                                                            self.network.value
        #                                                            ],
        #                                                           feed_dict=feed_dict)
        train_metrics = {'Value Loss': v_l, 'Policy Loss Indv':p_li, 'Entropy Loss': e_l, 'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient,value_gradient
