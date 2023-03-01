import ray
import tensorflow as tf

from modules.New_Worker import Worker
from modules import arguments, setters

from params.parameters import *

@ray.remote(num_cpus=1, num_gpus=int(GPU)/(NUM_META_AGENTS + 1))
class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""
    def __init__(self, metaAgentID, args_dict):
        self.ID = metaAgentID
        self.args_dict = args_dict
        self.workersPerMetaAgent = self.args_dict['NUM_ACTIVE'] + self.args_dict['NUM_PASSIVE']

        # Set up all required Classes 
        self.framework_setup()

        self.currEpisode = int(metaAgentID)

        self.tf_setup()

        # Worker setup
        groupLock = None
        self.worker = Worker(self.ID,
                             workers_per_metaAgent=self.workersPerMetaAgent,
                             sess=self.sess,
                             learningAgent=True,
                             groupLock=groupLock,
                             env=self.env,
                             model=self.model,
                             metrics=self.metrics,
                             observer=self.observer,
                             variables=self.variables,
                             buffer=self.buffer,
                             args_dict=self.args_dict)

        if self.args_dict['Target_Net']:
             self.worker.set_target_network(self.old_model)

    def tf_setup(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = int(self.args_dict['GPU']) / (
            self.args_dict['NUM_META_AGENTS'])
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=1)
        self.coord = tf.train.Coordinator()

        weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights = self.sess.run(weightVars)
        self.weightSetters = [tf.placeholder(shape=w.shape, dtype=tf.float32) for w in weights]
        self.set_weights_ops = [var.assign(w) for var, w in zip(weightVars, self.weightSetters)]

    def framework_setup(self):
        self.env = setters.EnvSetters.set_environment(self.args_dict,
                                                      self.ID)

        self.observer = setters.ObserverSetters.set_observer(args_dict=self.args_dict,
                                                             metaAgentID=self.ID,
                                                             env=self.env,
                                                             num_agents=self.args_dict['NUM_RL_AGENT'])


        self.model = setters.ModelSetters.set_model(self.args_dict,
                                                    trainer=None,
                                                    training=True,
                                                    GLOBAL_NETWORK=False,
                                                    input_shape=self.observer.shape,
                                                    env=self.env,
                                                    action_shapes=self.observer.action_spaces)

        if self.args_dict['Target_Net']:
            with tf.variable_scope('old'):
                self.old_model = setters.ModelSetters.set_model(self.args_dict,
                                                                trainer=None,
                                                                training=True,
                                                                GLOBAL_NETWORK=False,
                                                                input_shape=self.observer.shape,
                                                                env=self.env,
                                                                action_shapes=self.observer.action_spaces)

        self.metrics = setters.MetricSetters.set_metrics(self.args_dict,
                                                         metaAgentID=self.ID,
                                                         num_agents=self.args_dict['NUM_RL_AGENT'])

        self.variables = setters.VariableSetters.set_variables(self.args_dict,
                                                               metaAgentID=self.ID,
                                                               env=self.env,
                                                               num_agents=self.args_dict['NUM_RL_AGENT'])

        self.buffer = setters.BufferSetters.set_buffer(self.args_dict,
                                                       metaAgentID=self.ID,
                                                       variables=self.variables,
                                                       input_shape=self.observer.shape,
                                                       num_agents=self.args_dict['NUM_RL_AGENT'])

        self.variables_dummy = setters.VariableSetters.set_variables(self.args_dict,
                                                                     metaAgentID=self.ID,
                                                                     env=None,
                                                                     num_agents=self.args_dict['NUM_RL_AGENT'])

        self.buffer_dummy = setters.BufferSetters.set_buffer(args_dict=self.args_dict,
                                                             metaAgentID=self.ID,
                                                             variables=self.variables_dummy,
                                                             input_shape=self.observer.shape,
                                                             num_agents=self.args_dict['NUM_RL_AGENT'])

    def set_weights(self, weights):

        feed_dict = {
            self.weightSetters[i]: w for i, w in enumerate(weights)
        }
        self.sess.run([self.set_weights_ops], feed_dict=feed_dict)

    def singleThreadedJob(self, episodeNumber):
        # synchronous jobs only have one worker and
        # do not require a lock for synchronization
        self.worker.work(episodeNumber, self.coord, self.saver)

        # use list to be consistent with multi-threaded job
        jobResults = []
        if self.args_dict['JOB_TYPE'] == self.args_dict['JOB_OPTIONS'].getGradient:
            jobResults.append(self.worker.gradient)
            if self.args_dict['dynamics']:
                if not 'globalDynGradient' in self.args_dict.keys()\
                        or not self.args_dict['globalDynGradient']:
                    jobResults.append(self.worker.dynamics_gradient)
                else:
                    jobResults.append(self.worker.dynamics_buffer)
            if self.args_dict['Duelling']:
                jobResults.append(self.worker.value_gradient)

        elif self.args_dict['JOB_TYPE'] == self.args_dict['JOB_OPTIONS'].getExperience:
            # if get experience is True, return dummy buffer, variables, and metrics
            self.buffer_dummy.transitions = self.buffer.transitions
            if self.args_dict['intrinsic_dynamics_rewards']:
                self.buffer_dummy.dynamics_buffer = self.buffer.dynamics_buffer
            self.variables_dummy.bootstrapValue = self.variables.bootstrapValue

            jobResults.append(self.buffer_dummy)
            jobResults.append(self.variables_dummy)
            jobResults.append(self.metrics.episode_length)

        return jobResults, self.worker.episode_data

    def job(self, global_weights, episodeNumber):
        jobResults, metrics = None, None

        # set the local weights to the global weights from the master network
        self.set_weights(global_weights)

        if self.args_dict['COMPUTE_TYPE'] == self.args_dict['COMPUTE_OPTIONS'].synchronous:
            jobResults, metrics = self.singleThreadedJob(episodeNumber)

        # currently not implemented
        elif self.args_dict['COMPUTE_TYPE'] == self.args_dict['COMPUTE_OPTIONS'].multiThreaded:
            raise NotImplementedError

        # Get the job results from the learning agents
        # and send them back to the master network        
        info = {"id": self.ID}

        return jobResults, metrics, info

if __name__ == '__main__':
    globals_dict = arguments.set_args()
    rs = Runner(0,globals_dict)
    rs.singleThreadedJob(0)