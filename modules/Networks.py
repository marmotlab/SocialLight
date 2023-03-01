import tensorflow as tf
import tensorflow.contrib.layers as layers
from Utils import Utilities


# Todo: remove garbage networks, keep the networks needed for the necessary ablations

class vanilla_fc:
    def __init__(self, args_dict, input_size, output_size, scope, trainer):
        self.args_dict = args_dict
        self.input_size = input_size
        self.output_size, self.a_size = output_size, output_size
        self.scope = scope
        self.inputs = tf.placeholder(shape=input_size, dtype=tf.float32)
        self.trainer = trainer
        self.ext_val_head = self.args_dict['ext_value_head']
        self.build_net()

    def build_net(self):
        w_init = layers.variance_scaling_initializer()

        flatInputs = tf.nn.relu(layers.flatten(self.inputs))
        fc1 = layers.fully_connected(inputs=flatInputs, num_outputs=64)
        fc2 = layers.fully_connected(inputs=flatInputs, num_outputs=32)

        policy_layer = layers.fully_connected(inputs=fc2, num_outputs=self.output_size,
                                              weights_initializer=Utilities.normalized_columns_initializer(
                                                  1. / float(self.output_size)),
                                              biases_initializer=None, activation_fn=None)
        self.policy = tf.nn.softmax(policy_layer)
        self.policy_sig = tf.sigmoid(policy_layer)
        self.value = layers.fully_connected(inputs=fc1, num_outputs=1,
                                            weights_initializer=Utilities.normalized_columns_initializer(1.0),
                                            biases_initializer=None,
                                            activation_fn=None)

    def set_loss(self):
        if self.args_dict['PPO']:
            return self.ppo_loss()
        else:
            return self.standard_loss()

    def set_gradients(self):
        # Get gradients from local self using local losses and
        # normalize the gradients using clipping
        self.grad_clip = self.args_dict['GRAD_CLIP']
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, self.grad_clip)
        print("Hello World... From  " + str(self.scope))  # :)

    def set_global(self):
        self.grad_clip = self.args_dict['GRAD_CLIP']
        weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.tempGradients = [tf.placeholder(shape=w.get_shape(), dtype=tf.float32) for w in weightVars]
        self.clippedGrads, norms = tf.clip_by_global_norm(self.tempGradients, self.grad_clip)
        self.apply_grads = self.trainer.apply_gradients(zip(self.clippedGrads, weightVars))
        print("Hello World... From  " + str(self.scope))

    def standard_loss(self):
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
        self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        self.optimal_actions = tf.placeholder(tf.int32, [None])
        self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, self.a_size, dtype=tf.float32)

        if self.ext_val_head and self.args_dict['Model'] in ['DynamicsTraffic', 'LSTMDynamics']:
            self.target_ext_v = tf.placeholder(tf.float32, [None], 'Vexttarget')

        # Loss Functions
        self.value_loss = self.args_dict['value_weight'] * tf.reduce_sum(
            tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
        self.entropy = - self.args_dict['entropy_weight'] * tf.reduce_sum(
            self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))
        self.policy_loss = - self.args_dict['policy_weight'] * tf.reduce_sum(
            tf.log(tf.clip_by_value(self.responsible_outputs, 1e-15, 1.0)) * self.advantages)
        self.loss = self.value_loss + self.policy_loss - self.entropy

        if self.ext_val_head:
            self.ext_value_loss = self.args_dict['ext_value_weight'] * tf.reduce_sum(
                tf.square(self.target_ext_v - tf.reshape(self.ext_value, shape=[-1])))
            self.loss += self.ext_value_loss
        return

    def ppo_loss(self):
        print('Setting PPO Loss')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
        self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        self.optimal_actions = tf.placeholder(tf.int32, [None])
        self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, self.a_size, dtype=tf.float32)
        self.old_policy = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32)
        self.prob_a = tf.reduce_sum(self.old_policy * self.actions_onehot, [1])

        if self.ext_val_head and self.args_dict['Model'] in ['DynamicsTraffic', 'LSTMDynamics']:
            self.target_ext_v = tf.placeholder(tf.float32, [None], 'Vexttarget')

        # Loss Functions
        self.value_loss = tf.reduce_sum(
            tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
        self.entropy = - 1 * tf.reduce_sum(
            self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))

        # ppo policy loss
        ratio = tf.exp(tf.log(self.responsible_outputs) - tf.log(self.prob_a))
        surr1 = ratio * self.advantages
        surr2 = tf.clip_by_value(ratio, 0.9, 1.1) * self.advantages
        self.policy_loss = -1 * tf.reduce_sum(tf.minimum(surr1, surr2))

        # self.loss = 0.001 * self.value_loss + self.policy_loss - 0.0001 * self.entropy
        self.loss = self.args_dict['value_weight'] * self.value_loss + self.args_dict[
            'policy_weight'] * self.policy_loss - self.args_dict['entropy_weight'] * self.entropy

        if self.ext_val_head:
            self.ext_value_loss = self.args_dict['ext_value_weight'] * tf.reduce_sum(
                tf.square(self.target_ext_v - tf.reshape(self.ext_value, shape=[-1])))
            self.loss += self.ext_value_loss
        return

    def layer_norm(self, inp):
        return tf.contrib.layers.layer_norm(inp, center=True, scale=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, d_model)
            K (tf.tensor): of shape (h * batch, k_size, d_model)
            V (tf.tensor): of shape (h * batch, k_size, d_model)
            mask (tf.tensor): of shape (h * batch, q_size, k_size)
        """

        d = self.d_model // self.h
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        # out = tf.layers.dropout(out, training=self._is_training)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out

    def multi_head_attention(self, query, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, d_model)
            memory (tf.tensor): of shape (batch, m_size, d_model)
            mask (tf.tensor): shape (batch, q_size, k_size)
        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            Q = tf.layers.dense(query, self.d_model, activation=tf.nn.relu)
            K = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)
            V = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)

            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, d_model/num_heads]
            Q_split = tf.concat(tf.split(Q, self.h, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.h, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.h, axis=2), axis=0)
            # mask_split = tf.tile(mask, [self.h, 1, 1])
            mask_split = None
            # Apply scaled dot product attention
            out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_split)

            # Merge the multi-head back to the original shape
            out = tf.concat(tf.split(out, self.h, axis=0), axis=2)  # [bs, q_size, d_model]

            # The final linear layer and dropout.
            # out = tf.layers.dense(out, self.d_model)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

        return out

    def feed_forward(self, inp, scope='ff'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).
        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            # out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
            # out = tf.layers.dense(out, self.d_model, activation=None)

            # by default, use_bias=True
            out = tf.layers.conv1d(out, filters=self.d_ff, kernel_size=1, activation=tf.nn.relu)
            out = tf.layers.conv1d(out, filters=self.d_model, kernel_size=1)

        return out

    def encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forward
            # query: [batch, 1, dim], memory: [batch, 5, dim]
            out = self.layer_norm(out + self.multi_head_attention(query=tf.expand_dims(out[:, 0, :], 1), memory=out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forward(out))
        return out


class custom_mlp(vanilla_fc):
    def __init__(self, args_dict, input_size, output_size, scope, trainer):
        self.hidden_sizes = args_dict['hidden_sizes']
        self.ext_value_head = args_dict['ext_value_head']

        self.att = args_dict['ATTENTION']
        self.h = args_dict['NUM_HEADS']
        self.d_model = args_dict['MODEL_DIM']
        self.d_ff = args_dict['FF_DIM']
        self.att_hidden_sizes = args_dict['ATT_HIDDEN_SIZE']
        super().__init__(args_dict, input_size, output_size, scope, trainer)

    def build_net(self):
        if not self.att:
            w_init = layers.variance_scaling_initializer()
            flatInputs = tf.nn.relu(layers.flatten(self.inputs))
            fcs = []
            for i in range(len(self.hidden_sizes)):
                if i == 0:
                    fcs.append(layers.fully_connected(inputs=flatInputs, num_outputs=self.hidden_sizes[i]))
                else:
                    fcs.append(layers.fully_connected(inputs=fcs[i - 1], num_outputs=self.hidden_sizes[i]))

            policy_layer = layers.fully_connected(inputs=fcs[-1], num_outputs=self.output_size,
                                                  weights_initializer=Utilities.normalized_columns_initializer(
                                                      1. / float(self.output_size)),
                                                  biases_initializer=None, activation_fn=None)

            self.policy = tf.nn.softmax(policy_layer)
            self.policy_sig = tf.sigmoid(policy_layer)
            self.value = layers.fully_connected(inputs=fcs[-1], num_outputs=1,
                                                weights_initializer=Utilities.normalized_columns_initializer(1.0),
                                                biases_initializer=None,
                                                activation_fn=None)
            if self.ext_value_head:
                self.ext_value = layers.fully_connected(inputs=fcs[-1], num_outputs=1,
                                                        weights_initializer=Utilities.normalized_columns_initializer(1.0),
                                                        biases_initializer=None,
                                                        activation_fn=None)
        else:
            print('Set up Attention network')
            w_init = layers.variance_scaling_initializer()
            # Multi-head attention to process all neighbor observations
            eb_1 = layers.fully_connected(inputs=self.inputs, num_outputs=self.att_hidden_sizes[0])
            eb_2 = layers.fully_connected(inputs=eb_1, num_outputs=self.att_hidden_sizes[1])
            eb_3 = layers.fully_connected(inputs=eb_2, num_outputs=self.att_hidden_sizes[2])
            eb_4 = layers.fully_connected(inputs=eb_3, num_outputs=self.att_hidden_sizes[3])

            out = self.encoder_layer(inp=eb_4, input_mask=None, scope='attn')
            # flatInputs = tf.nn.relu(layers.flatten(out))
            flatInputs = layers.flatten(out)

            policy_layer = layers.fully_connected(inputs=flatInputs,
                                                  num_outputs=self.output_size,
                                                  weights_initializer=Utilities.normalized_columns_initializer(
                                                      1. / float(self.output_size)),
                                                  biases_initializer=None,
                                                  activation_fn=None)

            self.policy = tf.nn.softmax(policy_layer)
            self.policy_sig = tf.sigmoid(policy_layer)

            self.value = layers.fully_connected(inputs=flatInputs,
                                                num_outputs=1,
                                                weights_initializer=Utilities.normalized_columns_initializer(1.0),
                                                biases_initializer=None,
                                                activation_fn=None)

            if self.ext_value_head:
                self.ext_value = layers.fully_connected(inputs=flatInputs, num_outputs=1,
                                                        weights_initializer=Utilities.normalized_columns_initializer(
                                                            1.0),
                                                        biases_initializer=None,
                                                        activation_fn=None)


class coma_mlp(custom_mlp):
    '''
    Base network for both SocialLight and DecCOMA
    '''
    def __init__(self, args_dict, input_size, output_size, scope, trainer):
        #MATSC has atmost 4 neighbors
        self.args_dict = args_dict
        self.hidden_sizes = args_dict['hidden_sizes']
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.neighbour_actions = tf.placeholder(shape=[None,4], dtype=tf.int32)
        super(coma_mlp,self).__init__(args_dict, input_size, output_size, scope, trainer)

    def build_net(self):
        w_init = layers.variance_scaling_initializer()
        flatInputs = tf.nn.relu(layers.flatten(self.inputs))
        fcs = []
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                fcs.append(layers.fully_connected(inputs=flatInputs, num_outputs=self.hidden_sizes[i]))
            else:
                fcs.append(layers.fully_connected(inputs=fcs[i - 1], num_outputs=self.hidden_sizes[i]))

        policy_layer = layers.fully_connected(inputs=fcs[-1], num_outputs=self.output_size,
                                              weights_initializer=Utilities.normalized_columns_initializer(
                                                  1. / float(self.output_size)),
                                              biases_initializer=None, activation_fn=None)

        self.policy = tf.nn.softmax(policy_layer)
        self.policy_sig = tf.sigmoid(policy_layer)
        self.n_actions_onehot = tf.one_hot(self.neighbour_actions, self.a_size + 1, dtype=tf.float32, axis=2)
        #action_layer = layers.fully_connected(inputs=layers.flatten(self.n_actions_onehot), num_outputs=self.hidden_sizes[-1])
        flatinputsvalue = tf.concat([fcs[-1], layers.flatten(self.n_actions_onehot),tf.nn.relu(layers.flatten(self.inputs))], axis=1)
        #flatinputsvalue = fcs[-1]

        valuelayer = [flatinputsvalue]
        valuelayer.append(layers.fully_connected(inputs=valuelayer[-1], num_outputs=self.hidden_sizes[-1]))
        self.qvalue = layers.fully_connected(inputs=valuelayer[-1], num_outputs=self.output_size,
                                             weights_initializer=Utilities.normalized_columns_initializer(1.0),
                                             biases_initializer=None,
                                             activation_fn=None)


    def set_loss(self):
        return self.standard_loss()

    def standard_loss(self):
        self.actions_onehot2 = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
        self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
        self.target_qval = tf.placeholder(tf.float32,[None],'Q-VTarget')

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot2, [1])
        self.indv_advantages = tf.placeholder(shape=[None],dtype=tf.float32)
        # Loss Functions

        self.qvalue_loss = self.args_dict['q_value_weight'] * tf.reduce_sum(
            tf.square(self.target_qval - tf.reshape(tf.reduce_sum(self.qvalue*self.actions_onehot2,[1]), shape=[-1])))
        self.entropy = - self.args_dict['entropy_weight'] * tf.reduce_sum(
            self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))

        self.policy_loss_indv = -self.args_dict['policy_weight']*tf.reduce_sum(
            tf.log(tf.clip_by_value(self.responsible_outputs, 1e-15, 1.0)) * self.indv_advantages)
        self.loss =  self.policy_loss_indv - self.entropy + self.qvalue_loss

        return

