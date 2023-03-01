import tensorflow as tf
from Utils import Utilities
import numpy as np
from modules import setters
import copy

from Models.Model import Model


class VanillaTraffic(Model):

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
            return train_metrics, gradient,None

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


class LSTMTraffic(VanillaTraffic):
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def forward(self, sess, variables):
        vars_required = ['actions', 'values', 'state_out']
        output_dict = Utilities.get_forward_dict(self.num_agents, vars_required)
        for i in range(self.num_agents):
            return_dict = self.forward_step(sess, variables, i)
            output_dict['actions'][i] = return_dict['action']
            output_dict['values'][i] = return_dict['value'][0, 0]
            output_dict['state_out'][i] = return_dict['state_out']
        return output_dict

    def forward_step(self, sess, variables, index):
        return_dict = {}
        return_dict['action'], return_dict['value'], return_dict['state_out'] = sess.run(
            [self.network.policy, self.network.value, self.network.state_out],
            feed_dict={self.network.inputs: [variables.nextobs_dict[index]],
                       self.network.state_in[0]: variables.current_rnn_dict[index][0],
                       self.network.state_in[1]: variables.current_rnn_dict[index][1]})
        return return_dict

    def backward(self, index, train_buffer, advantage_dict, sess):
        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.advantages: advantage_dict['advantages'],
                     self.network.state_in[0]: train_buffer['initial_rnn'][0],
                     self.network.state_in[1]: train_buffer['initial_rnn'][1]}

        if self.args_dict['PPO']:
            feed_dict[self.network.old_policy] = np.stack(train_buffer['old_policy'])

        v_l, p_l, e_l, g_n, v_n, gradient = sess.run([self.network.value_loss,
                                                      self.network.policy_loss,
                                                      self.network.entropy,
                                                      self.network.grad_norms,
                                                      self.network.var_norms,
                                                      self.network.gradients],
                                                     feed_dict=feed_dict)

        train_metrics = {'Value Loss': v_l, 'Policy Loss': p_l, 'Entropy Loss': e_l, 'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient


class DynamicsTraffic(VanillaTraffic):
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, dynamics_trainer=None, action_shapes=None):
        self.dynamics_trainer = dynamics_trainer
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def initialize(self, input_shape):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        self.input_shape, self.input_policy_shape = [None], [None]
        self.input_shape.extend(input_shape)
        self.input_policy_shape.extend([5 * self.a_size])
        if self.env is not None and self.args_dict['intrinsic_dynamics_rewards']:
            self.init_intrinsic_reward()

        with tf.variable_scope(str(self.scope) + '/qvalues'):
            self.inputs = tf.placeholder(shape=self.input_shape, dtype=tf.float32)
            self.network = self.make_net()
            if self.TRAINING:
                self.network.set_loss()
                self.network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.network.set_global()
            self.networks = [self.network]

        with tf.variable_scope('dynamics'):
            # for attention model, the input size will be [batch, num_nodes, num_features]
            self.dynamics_shape = [None, int(np.prod(input_shape))]
            self.dynamics_network = setters.NetworkSetters.set_dynamics_network(self.args_dict,
                                                                                # self.input_shape,
                                                                                self.dynamics_shape,
                                                                                self.input_policy_shape,
                                                                                # self.input_shape[1],
                                                                                self.dynamics_shape[1],
                                                                                scope='dynamics',
                                                                                trainer=self.dynamics_trainer)
            if self.TRAINING:
                self.dynamics_network.set_loss()
                self.dynamics_network.set_gradients()
            if self.GLOBAL_NETWORK:
                self.dynamics_network.set_global()
            self.dynamics_networks = [self.dynamics_network]

    def backward(self, index, train_buffer, advantage_dict, sess):
        if not self.args_dict['dynamics']:
            return super().backward(index, train_buffer, advantage_dict, sess)
        elif self.args_dict['dynamics']:
            return self.dynamics_backward(index, train_buffer, advantage_dict, sess)

    def dynamics_backward(self, index, train_buffer, advantage_dict, sess):
        train_metrics, gradient, d_gradient = None, None, None
        if not self.args_dict["GLOBAL_PPO"]:
            feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                         self.network.inputs: np.stack(train_buffer['observations']),
                         self.network.actions: train_buffer['actions'],
                         self.network.advantages: advantage_dict['advantages']}

            if self.args_dict['PPO']:
                feed_dict[self.network.old_policy] = np.stack(train_buffer['old_policy'])

            if not self.args_dict['ext_value_head']:
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

            if self.args_dict['ext_value_head']:
                feed_dict[self.network.target_ext_v] = advantage_dict['discounted_ext_rewards']
                e_v_l, v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                    [self.network.ext_value_loss,
                     self.network.value_loss,
                     self.network.policy_loss,
                     self.network.entropy,
                     self.network.grad_norms,
                     self.network.var_norms,
                     self.network.gradients],

                    feed_dict=feed_dict)

                train_metrics = {'External Value Loss': e_v_l,
                                 'Value Loss': v_l,
                                 'Policy Loss': p_l,
                                 'Entropy Loss': e_l,
                                 'Grad Norm': g_n,
                                 'Var Norm': v_n}

            # DYNAMICS NET TRAINING
            feed_dict = {
                self.dynamics_network.target_state: np.reshape(train_buffer['state_targets'], (-1, self.dynamics_shape[1])),
                self.dynamics_network.state_inputs: train_buffer['state_inputs'],
                self.dynamics_network.policy_inputs: train_buffer['policy_inputs']}

            d_l, d_g_n, d_v_n, d_gradient = sess.run([self.dynamics_network.loss,
                                                      self.dynamics_network.grad_norms,
                                                      self.dynamics_network.var_norms,
                                                      self.dynamics_network.gradients],
                                                     feed_dict=feed_dict)

            train_metrics.update({'Dynamics Loss': d_l,
                                  'Dynamics Grad Norm': d_g_n,
                                  'Dynamics Var Norm': d_v_n})

            if self.args_dict['intrinsic_dynamics_rewards']:
                train_metrics['Intrinsic Rewards'] = self.total_intrinsic_rewards[index]

            return train_metrics, gradient, d_gradient

        else:
            assert self.args_dict['PPO'] == True
            v_l_list, p_l_list, e_l_list, g_n_list, v_n_list, e_v_l_list = [], [], [], [], [], []
            d_l_list, d_g_n_list, d_v_n_list, i_r_list = [], [], [], []
            for i in range(self.args_dict['N_EPOCH']):
                feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                             self.network.inputs: np.stack(train_buffer['observations']),
                             self.network.actions: train_buffer['actions'],
                             self.network.advantages: advantage_dict['advantages'],
                             self.network.old_policy: np.stack(train_buffer['old_policy'])}

                if not self.args_dict['ext_value_head']:
                    v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                        [self.network.value_loss,
                         self.network.policy_loss,
                         self.network.entropy,
                         self.network.grad_norms,
                         self.network.var_norms,
                         self.network.gradients],
                        feed_dict=feed_dict)

                    v_l_list.append(v_l)
                    p_l_list.append(p_l)
                    e_l_list.append(e_l)
                    g_n_list.append(g_n)
                    v_n_list.append(v_n)

                if self.args_dict['ext_value_head']:
                    feed_dict[self.network.target_ext_v] = advantage_dict['discounted_ext_rewards']
                    e_v_l, v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                        [self.network.ext_value_loss,
                         self.network.value_loss,
                         self.network.policy_loss,
                         self.network.entropy,
                         self.network.grad_norms,
                         self.network.var_norms,
                         self.network.gradients],
                        feed_dict=feed_dict)

                    e_v_l_list.append(e_v_l)
                    v_l_list.append(v_l)
                    p_l_list.append(p_l)
                    e_l_list.append(e_l)
                    g_n_list.append(g_n)
                    v_n_list.append(v_n)

                # update decision network
                feed_dict = {
                    self.network.tempGradients[i]: g for i, g in enumerate(gradient)
                }
                sess.run([self.network.apply_grads], feed_dict=feed_dict)

                # DYNAMICS NET TRAINING
                feed_dict = {
                    self.dynamics_network.target_state: np.reshape(train_buffer['state_targets'],
                                                                   (-1, self.dynamics_shape[1])),
                    self.dynamics_network.state_inputs: train_buffer['state_inputs'],
                    self.dynamics_network.policy_inputs: train_buffer['policy_inputs']}

                d_l, d_g_n, d_v_n, d_gradient = sess.run([self.dynamics_network.loss,
                                                          self.dynamics_network.grad_norms,
                                                          self.dynamics_network.var_norms,
                                                          self.dynamics_network.gradients],
                                                         feed_dict=feed_dict)

                # update dynamic network
                feed_dict = {
                    self.dynamics_network.tempGradients[i]: g for i, g in enumerate(d_gradient)
                }
                sess.run([self.dynamics_network.apply_grads], feed_dict=feed_dict)

                if self.args_dict['intrinsic_dynamics_rewards']:
                    # train_metrics['Intrinsic Rewards'] = self.total_intrinsic_rewards[index]
                    i_r_list.append(self.total_intrinsic_rewards[index])

                d_l_list.append(d_l)
                d_g_n_list.append(d_g_n)
                d_v_n_list.append(d_v_n)

            if not self.args_dict['ext_value_head']:
                train_metrics = {'Value Loss': np.mean(np.array(v_l_list)),
                                 'Policy Loss': np.mean(np.array(p_l_list)),
                                 'Entropy Loss': np.mean(np.array(e_l_list)),
                                 'Grad Norm': np.mean(np.array(g_n_list)),
                                 'Var Norm': np.mean(np.array(v_n_list))}
            if self.args_dict['ext_value_head']:
                train_metrics = {'External Value Loss': np.mean(np.array(e_v_l_list)),
                                 'Value Loss': np.mean(np.array(v_l_list)),
                                 'Policy Loss': np.mean(np.array(p_l_list)),
                                 'Entropy Loss': np.mean(np.array(e_l_list)),
                                 'Grad Norm': np.mean(np.array(g_n_list)),
                                 'Var Norm': np.mean(np.array(v_n_list))}
            if self.args_dict['intrinsic_dynamics_rewards']:
                train_metrics['Intrinsic Rewards'] = np.mean(np.array(i_r_list))

            train_metrics.update({'Dynamics Loss': np.mean(np.array(d_l_list)),
                                  'Dynamics Grad Norm': np.mean(np.array(d_g_n_list)),
                                  'Dynamics Var Norm': np.mean(np.array(d_v_n_list))})

            return train_metrics, None, None

    def compute_all_intrinsic_rewards(self, buffer, sess, ep_length):
        self.episode_length = ep_length
        for i in range(self.num_agents):
            train_buffer = buffer.get_train_buffer(i)
            self.compute_future_vals(train_buffer, i, sess)

    def init_intrinsic_reward(self):
        self.agent_neighbours, self.intrinsic_neighbour_rewards, self.intrinsic_value_variances, self.total_intrinsic_rewards = {}, {}, {}, []
        self.num_neighbours = len(self.env.neighbour_dict[Utilities.index_to_ID(0, self.env)])
        for i in range(self.num_agents):
            # intrinsic neighbour rewards collects rewards given to agent at index i from its neighbours 
            self.intrinsic_neighbour_rewards[i], self.intrinsic_value_variances[i] = {}, {}
            self.total_intrinsic_rewards.append(0)
            agent_id = Utilities.index_to_ID(i, self.env)
            neighbours = self.env.neighbour_dict[agent_id]
            self.agent_neighbours[i] = [None for _ in range(len(neighbours))]
            for j, neighbour in enumerate(neighbours):
                if neighbour is not None:
                    policies_matrix = np.zeros((self.a_size, self.a_size))
                    np.fill_diagonal(policies_matrix, 1)
                    self.agent_neighbours[i][
                        j] = policies_matrix  # agent_neighbours stores all possible policies j of the neighbours of agent at index i
                    neighbour_ind = Utilities.ID_to_index(neighbour, self.env)
                    self.intrinsic_neighbour_rewards[i][neighbour_ind] = [0 for _ in range(self.a_size)]
                    self.intrinsic_value_variances[i][neighbour_ind] = [0]

    def compute_future_vals(self, train_buffer, index, sess):
        state_inputs = train_buffer['state_inputs']
        policy_inputs = np.reshape(train_buffer['policy_inputs'], (-1, self.num_neighbours + 1, self.a_size))
        for i, neighbour in enumerate(self.agent_neighbours[index]):
            if neighbour is not None:
                for j in range(self.a_size):  # For each possible policy of that neighbour
                    temp_policy_inputs = copy.deepcopy(policy_inputs)
                    temp_policy_inputs[0:, i + 1, 0:] = self.agent_neighbours[index][i][
                        j]  # The counterfactual for the dynamics network
                    temp_policy_inputs = np.reshape(temp_policy_inputs, (-1, self.a_size * 5))  # Some reshaping
                    forward_states = sess.run([self.dynamics_network.dynamics_prediction],
                                              # Get future state based on counterfactual
                                              feed_dict={self.dynamics_network.state_inputs: np.reshape(
                                                  train_buffer['state_targets'], (-1, self.dynamics_shape[1])),
                                                  self.dynamics_network.policy_inputs: temp_policy_inputs})
                    if self.args_dict['ext_value_head']:
                        # Get value from that future state  
                        if self.args_dict['Model'] == 'LSTMDynamics':
                            values = sess.run([self.network.ext_value],
                                              feed_dict={self.network.state_in[0]: train_buffer['first_rnn'][0],
                                                         self.network.state_in[1]: train_buffer['first_rnn'][1],
                                                         self.network.inputs: np.reshape(forward_states,
                                                                                         (-1, self.input_shape[1]))})
                        else:
                            values = sess.run([self.network.ext_value], feed_dict={
                                # self.network.inputs: np.reshape(forward_states, (-1, self.input_shape[1]))})
                                self.network.inputs: np.reshape(forward_states, ([-1] + self.input_shape[1:]))
                            })
                    else:
                        if self.args_dict['Model'] == 'LSTMDynamics':
                            values = sess.run([self.network.value],
                                              feed_dict={self.network.state_in[0]: train_buffer['first_rnn'][0],
                                                         self.network.state_in[1]: train_buffer['first_rnn'][1],
                                                         self.network.inputs: np.reshape(forward_states,
                                                                                         (-1, self.input_shape[1]))})
                        else:
                            values = sess.run([self.network.value], feed_dict={
                                self.network.inputs: np.reshape(forward_states, (-1, self.input_shape[1]))})
                    neighbours = self.env.neighbour_dict[Utilities.index_to_ID(index, self.env)]
                    neighbour_ind = Utilities.ID_to_index(neighbours[i], self.env)
                    # Store this value at the neighbour index , this is the value of an agent i (index) when its neighbour samples a different action
                    self.intrinsic_neighbour_rewards[neighbour_ind][index][j] = values[0].reshape(1, -1)
                # Get all values of agent i obtained using all possible counterfactuals of neighbour j
                vals = np.reshape(np.array(self.intrinsic_neighbour_rewards[neighbour_ind][index]), (-1, self.a_size))
                # Get the variance of these values, this variance measures the sensitivity of agent i on its neighbour j's actions 
                variances = np.std(vals, axis=1)
                # Store these values for possible later use to assign weights 
                self.intrinsic_value_variances[neighbour_ind][index] = variances.reshape(1, -1)

    def add_intrinsic_rewards(self, index, train_buffer):
        # Here we compute the intrinsic reward for agent i (index) based on the values assigned to it by its neighbours 
        v = None
        keys = [k for k, v in self.intrinsic_neighbour_rewards[index].items()]
        value_matrix = np.zeros((self.a_size,
                                 self.episode_length))  # Number of values assigned to agent i (based on the size of its action space)
        if self.args_dict['variance_attention']:
            self.compute_variance_weights(index)  # Compute weights for variance if set to true 
        for j in range(self.a_size):  # For each action of agent i (index) , compute the sum of values of its neighbours
            for key in keys:
                if self.args_dict['variance_attention']:
                    if v is not None:
                        v += self.intrinsic_neighbour_rewards[index][key][j] * self.intrinsic_value_variances[index][
                            key]
                    else:
                        v = self.intrinsic_neighbour_rewards[index][key][j] * self.intrinsic_value_variances[index][key]
                else:
                    if v is not None:
                        v += self.intrinsic_neighbour_rewards[index][key][j]
                    else:
                        v = self.intrinsic_neighbour_rewards[index][key][j]
            value_matrix[j] = v  # Store the sum of values of neighbours for action j of agent i
        max_vals = np.amax(value_matrix,
                           0)  # The max possible sum - the action of agent i which would maximize these sum of values
        actual_vals = np.zeros((1, self.episode_length))  # The actual sum based on the real action taken by agent i
        actions = train_buffer['actions']
        for i in range(self.episode_length):
            actual_vals[0][i] = value_matrix[actions[i]][i]  # Compute the actual sum
        intrinsic_rewards = actual_vals - max_vals  # Rewards are the difference between these two sums
        self.total_intrinsic_rewards[index] = self.args_dict['intrinsic_reward_weight'] * float(
            np.sum(intrinsic_rewards))  # Store rewards for logging\
        # Add these intrinsic rewards to the extrinsic rewards using both a weighting factor and a decay factor to stabilize learning
        intrinsic_rewards_weighted = self.args_dict['intrinsic_reward_weight'] * (
                1 - np.exp(self.args_dict['intrinsic_reward_decay'] * np.sqrt(self.episode_number))) * np.squeeze(
            intrinsic_rewards)
        intrinsic_rewards_weighted *= (self.episode_number > self.args_dict['pretraining_epsiodes'])
        train_buffer['rewards'] += intrinsic_rewards_weighted
        return train_buffer

    def compute_variance_weights(self, index):
        # Compute weights for neighbours of agent i (index) to be used in computation of intrinsic rewards
        # These weights are based on how sensitive each neighbour is at any given timestep to the actions of agent i at index
        var_weights = np.zeros((4, self.episode_length))
        keys = [k for k, v in self.intrinsic_neighbour_rewards[index].items()]
        for i, key in enumerate(keys):
            var_weights[i] = self.intrinsic_value_variances[index][key]
        for key in keys:
            # Compute weights and normalize to prevent Nan errors 
            self.intrinsic_value_variances[index][key] = self.intrinsic_value_variances[index][key] / (
                np.clip(np.sum(var_weights, axis=0), 0.0001, 50))

    def get_advantages(self, train_buffer, bootstrapValue, index=None):
        if self.args_dict['intrinsic_dynamics_rewards']:
            assert index is not None
            if self.args_dict['ext_value_head']:
                advantage_dict = super().get_advantages(train_buffer, bootstrapValue)
            train_buffer = self.add_intrinsic_rewards(index, train_buffer)
            intrinsic_advantage_dict = super().get_advantages(train_buffer, bootstrapValue)
            if self.args_dict['ext_value_head']:
                intrinsic_advantage_dict['discounted_ext_rewards'] = advantage_dict['discounted_rewards']
            return intrinsic_advantage_dict
        else:
            return super().get_advantages(train_buffer, bootstrapValue)


class LSTMDynamics(DynamicsTraffic, LSTMTraffic):
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, dynamics_trainer=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         dynamics_trainer, action_shapes)

    def forward(self, sess, variables):
        return LSTMTraffic.forward(self, sess, variables)

    def forward_step(self, sess, variables, index):
        return LSTMTraffic.forward_step(self, sess, variables, index)

    def backward(self, index, train_buffer, advantage_dict, sess):
        if not self.args_dict['dynamics']:
            return LSTMTraffic.backward(self, index, train_buffer, advantage_dict, sess)
        elif self.args_dict['dynamics']:
            return self.dynamics_backward(index, train_buffer, advantage_dict, sess)

    def dynamics_backward(self, index, train_buffer, advantage_dict, sess):
        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.advantages: advantage_dict['advantages'],
                     self.network.state_in[0]: train_buffer['initial_rnn'][0],
                     self.network.state_in[1]: train_buffer['initial_rnn'][1]}

        if self.args_dict['PPO']:
            feed_dict[self.network.old_policy] = np.stack(train_buffer['old_policy'])

        if not self.args_dict['ext_value_head']:
            v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                [self.network.value_loss, self.network.policy_loss, self.network.entropy, self.network.grad_norms,
                 self.network.var_norms, self.network.gradients], feed_dict=feed_dict)

            train_metrics = {'Value Loss': v_l, 'Policy Loss': p_l, 'Entropy Loss': e_l, 'Grad Norm': g_n,
                             'Var Norm': v_n}

        if self.args_dict['ext_value_head']:
            feed_dict[self.network.target_ext_v] = advantage_dict['discounted_ext_rewards']
            e_v_l, v_l, p_l, e_l, g_n, v_n, gradient = sess.run(
                [self.network.ext_value_loss, self.network.value_loss, self.network.policy_loss, self.network.entropy,
                 self.network.grad_norms,
                 self.network.var_norms, self.network.gradients], feed_dict=feed_dict)
            train_metrics = {'External Value Loss': e_v_l, 'Value Loss': v_l, 'Policy Loss': p_l, 'Entropy Loss': e_l,
                             'Grad Norm': g_n, 'Var Norm': v_n}

        ## DYNAMICS NET TRAINING 
        feed_dict = {self.dynamics_network.target_state: train_buffer['state_targets'],
                     self.dynamics_network.state_inputs: train_buffer['state_inputs'],
                     self.dynamics_network.policy_inputs: train_buffer['policy_inputs']}

        d_l, d_g_n, d_v_n, d_gradient = sess.run([self.dynamics_network.loss, self.dynamics_network.grad_norms,
                                                  self.dynamics_network.var_norms, self.dynamics_network.gradients],
                                                 feed_dict=feed_dict)
        train_metrics.update({'Dynamics Loss': d_l, 'Dynamics Grad Norm': d_g_n, 'Dynamics Var Norm': d_v_n})

        if self.args_dict['intrinsic_dynamics_rewards']:
            train_metrics['Intrinsic Rewards'] = self.total_intrinsic_rewards[index]

        return train_metrics, gradient, d_gradient


class PPOTraffic(VanillaTraffic):
    def __init__(self, scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK=False, args_dict=None,
                 input_shape=[6, 6], env=None, action_shapes=None):
        super().__init__(scope, trainer, TRAINING, GLOBAL_NET_SCOPE, GLOBAL_NETWORK, args_dict, input_shape, env,
                         action_shapes)

    def backward(self, index, train_buffer, advantage_dict, sess):
        feed_dict = {self.network.target_v: advantage_dict['discounted_rewards'],
                     self.network.inputs: np.stack(train_buffer['observations']),
                     self.network.actions: train_buffer['actions'],
                     self.network.advantages: advantage_dict['advantages'],
                     self.network.old_policy: np.stack(train_buffer['old_policy'])}

        v_l, p_l, e_l, g_n, v_n, gradient = sess.run([self.network.value_loss,
                                                      self.network.policy_loss,
                                                      self.network.entropy,
                                                      self.network.grad_norms,
                                                      self.network.var_norms,
                                                      self.network.gradients],
                                                     feed_dict=feed_dict)

        train_metrics = {'Value Loss': v_l, 'Policy Loss': p_l, 'Entropy Loss': e_l, 'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient
