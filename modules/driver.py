import ray
from Utils import Utilities
from modules import setters
import random
import numpy as np
import tensorflow as tf

from modules.Runner import Runner

def setup(args_dict):
    ray.init(num_gpus=int(args_dict['GPU']))
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = int(args_dict['GPU']) / (args_dict['NUM_META_AGENTS'])
    config.gpu_options.allow_growth = True
    return config


def apply_gradients(global_model, gradients, sess, index=0):
    feed_dict = {
        global_model.networks[index].tempGradients[i]: g for i, g in enumerate(gradients)
    }
    sess.run([global_model.networks[index].apply_grads], feed_dict=feed_dict)

def apply_gradients_duelling(global_model,gradients,sess,index=0):
    feed_dict = {
        global_model.networks[index].tempGradientsValue[i]: g for i, g in enumerate(gradients)
    }
    sess.run([global_model.networks[index].apply_grads_value], feed_dict=feed_dict)

def apply_dynamics_gradients(global_model, gradients, sess, index=0):
    feed_dict = {
        global_model.dynamics_networks[index].tempGradients[i]: g for i, g in enumerate(gradients)
    }
    sess.run([global_model.dynamics_networks[index].apply_grads], feed_dict=feed_dict)


def set_ray_jobs(meta_agents, weights, curr_episode):
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_episode))
        curr_episode += 1
    return job_list, curr_episode


def reinit_ray(jobList, meta_agents, args_dict):
    if jobList == []:
        print('REINITIALIZING RAY')
        ray.shutdown()
        ray.init(num_gpus=int(args_dict['GPU']))
        meta_agents = [Runner.remote(i, args_dict) for i in range(args_dict['NUM_META_AGENTS'])]
        return True, meta_agents

    else:
        return False, meta_agents

def calculateDynamicsGradient(model,args_dict,buffer,sess):
    if args_dict['parameter_sharing']:

        train_buffer = {
            'observations': buffer['observations'],
            'next_observations': buffer['next_observations'],
            'actions': np.stack(buffer['actions']),
            'neighbours_actions': np.stack(buffer['neighbours_actions'])
        }
        dyn_metrics, dyn_gradient = model.dynamics_backward_only(train_buffer, sess)
        feed_dict = {
            model.dynamics_network.tempGradients[i]: g for i, g in
            enumerate(dyn_gradient)
        }
        sess.run([model.dynamics_network.apply_grads], feed_dict=feed_dict)

    else:
        buffer_index_list = [i for i in range(args_dict['NUM_RL_AGENT'])]
        for index in buffer_index_list:
            train_buffer = {
                'observations': buffer[index]['observations'],
                'next_observations': buffer[index]['next_observations'],
                'actions': np.stack(buffer[index]['actions']),
                'neighbours_actions': np.stack(buffer[index]['neighbours_actions'])
            }
            dyn_metrics, dyn_gradient = model.dynamics_backward_only(train_buffer, sess, index)
            feed_dict = {
                model.dynamics_networks[index].tempGradients[i]: g for i, g in
                enumerate(dyn_gradient)
            }
            sess.run([model.dynamics_network.apply_grads], feed_dict=feed_dict)
    return dyn_metrics

def calculateRewardsGradient(model,args_dict,buffer,sess):
    if args_dict['parameter_sharing']:
        train_buffer = {
            'observations': buffer['observations'],
            'rewards': buffer['rewards'],
            'actions': np.stack(buffer['actions']),
            'neighbours_actions': np.stack(buffer['neighbour_actions'])
        }
        rewards_metrics, dyn_gradient = model.rewards_backward_only(train_buffer, sess)
        feed_dict = {
            model.rewards_network.tempGradients[i]: g for i, g in
            enumerate(dyn_gradient)
        }
        sess.run([model.rewards_network.apply_grads], feed_dict=feed_dict)

    else:
        buffer_index_list = [i for i in range(args_dict['NUM_RL_AGENT'])]
        for index in buffer_index_list:
            train_buffer = {
                'observations': buffer[index]['observations'],
                'rewards': buffer[index]['rewards'],
                'actions': np.stack(buffer[index]['actions']),
                'neighbours_actions': np.stack(buffer[index]['neighbour_actions'])
            }
            rewards_metrics, dyn_gradient = model.rewards_backward_only(train_buffer, sess, index)
            feed_dict = {
                model.rewards_networks[index].tempGradients[i]: g for i, g in
                enumerate(dyn_gradient)
            }
            sess.run([model.rewards_network.apply_grads], feed_dict=feed_dict)
    return rewards_metrics


def calculateGradient(args_dict, sess, global_model, buffer, variables, metrics):
    if args_dict['batch_gradient']:
        buffer.compute_combined_buffer()
        buffer.compute_combined_advantages(model=global_model,
                                           variables=variables)
        for j in range(args_dict['train_iterations']):
            train_batch, advantage_batch = buffer.sample_batch(size=args_dict['batch_size'])
            train_metrics, _ = global_model.backward(0, train_batch, advantage_batch, sess)
            # gradient.append(gradient)
            metrics.update_batch_metrics(train_metrics, j)
    else:
        if args_dict['intrinsic_dynamics_rewards']:
            global_model.compute_all_intrinsic_rewards(buffer, sess, metrics.episode_length)
        buffer_index_list = [i for i in range(args_dict['NUM_RL_AGENT'])]
        random.shuffle(buffer_index_list)
        for i in buffer_index_list:
            train_buffer = buffer.get_train_buffer(i)
            if args_dict['dynamics']:
                advantage_dict = global_model.get_advantages(train_buffer, variables.bootstrapValue[i], i)
                train_metrics, _, _ = global_model.backward(i, train_buffer, advantage_dict, sess)
                # self.gradient.append(gradient)
                # self.dynamics_gradient.append(dynamics_gradient)
            else:
                advantage_dict = global_model.get_advantages(train_buffer, variables.bootstrapValue[i])
                train_metrics, _ = global_model.backward(i, train_buffer, advantage_dict, sess)
                # self.gradient.append(gradient)
            metrics.update_train_metrics(train_metrics, i)

    return metrics.total_train_metrics()


def main(args_dict, run=None, wandb_run=None):
    config = setup(args_dict)

    with tf.device("/cpu:0"):
        lr = tf.constant(args_dict['LR_Q'])
        lr_2 = tf.constant(args_dict['dynamics_LR'])
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr,
                                                use_locking=True)
        dynamics_trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr_2,
                                                         use_locking=True)
        dummy_env = setters.EnvSetters.set_environment(args_dict,
                                                       -1)
        dummy_observer = setters.ObserverSetters.set_observer(args_dict,
                                                              dummy=True,
                                                              num_agents=args_dict['NUM_RL_AGENT'],
                                                              env=dummy_env)
        if args_dict['GLOBAL_PPO']:
            training = True
        else:
            training = False
        global_model = setters.ModelSetters.set_model(args_dict,
                                                      trainer=trainer,
                                                      training=training,
                                                      GLOBAL_NETWORK=True,
                                                      input_shape=dummy_observer.shape,
                                                      dynamics_trainer=dynamics_trainer,
                                                      action_shapes=dummy_observer.action_spaces,
                                                      env=dummy_env)
        # global_step = tf.placeholder(tf.float32)
        global_metrics = setters.MetricSetters.set_metrics(args_dict=args_dict,
                                                           metaAgentID='global',
                                                           num_agents=args_dict['NUM_RL_AGENT'])

        global_summary = tf.summary.FileWriter(args_dict['train_path'])
        saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=config) as sess:
        graph_writer = tf.summary.FileWriter(args_dict['train_path'], sess.graph)  # Log the computation to Tensorboard
        sess.run(tf.global_variables_initializer())
        graph_writer.close()

        tensorboard_writer = Utilities.Tensorboard(args_dict, global_summary)

        # TODO train dynamics model if not trained or load up new dynamics model
        tensorboard_writer.reset()

        if args_dict['load_model']:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(args_dict['model_path'])
            p = ckpt.model_checkpoint_path
            p = p[p.find('-') + 1:]
            p = p[:p.find('.')]
            p = p[-10:]
            p = p[p.find('-') + 1:]
            curr_episode = int(p)

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("curr_episode set to ", curr_episode)
        else:
            curr_episode = 0

        # launch all of the threads:
        meta_agents = [Runner.remote(i, args_dict) for i in range(args_dict['NUM_META_AGENTS'])]


        # get the initial weights from the global network
        weight_names = tf.trainable_variables()
        weights = sess.run(weight_names)  # Gets weights in numpy arrays CHECK
        if args_dict['Target_Net']:
            old_weight_names = tf.trainable_variables()
            old_weights = sess.run(old_weight_names)

        # launch the first job (e.g. getGradient) on each runner
        if args_dict['Target_Net']:
            jobList, curr_episode = set_ray_jobs(meta_agents, weights+old_weights, curr_episode)  # Ray ObjectIDs
        else:
            jobList, curr_episode = set_ray_jobs(meta_agents, weights, curr_episode)  # Ray ObjectIDs

        if 'ReplayBuffer' in args_dict.keys():
            global_replay_buffer = setters.ReplayBufferSetters.set_replaybuffer(args_dict,
                                                                        global_model.getReplayBufferKeys(),
                                                                        args_dict['Capacity'])
        reinit_count = 0
        returns, best_return = [], -9999

        if args_dict['Duelling']:
            value_gradients = []

        try:
            while curr_episode < (args_dict['epochs'] + args_dict['NUM_META_AGENTS']) and jobList != []:
                # wait for any job to be completed - unblock as soon as the earliest arrives
                done_id, jobList = ray.wait(jobList)
                # get the results of the task from the object store
                jobResults, metrics, info = ray.get(done_id)[0]

                # for saving the best model
                returns.append(metrics['Perf']['Reward'])
                if args_dict['JOB_TYPE'] == args_dict['JOB_OPTIONS'].getGradient:
                    if args_dict['parameter_sharing']:
                        # apply gradient on the global network
                        for gradients in jobResults[0]:
                            apply_gradients(global_model, gradients, sess)

                        if args_dict['Duelling']:
                            for gradients in jobResults[1]:
                                value_gradients.append(gradients)
                            if curr_episode%args_dict['Duelling_Update_Freq'] == 0:
                                for gradients in value_gradients:
                                    apply_gradients_duelling(global_model,gradients,sess)
                                value_gradients = []


                    elif not args_dict['parameter_sharing']:
                        for index, gradient in enumerate(jobResults[0]):
                            apply_gradients(global_model, gradient, sess, index)
                        if args_dict['Duelling']:
                            for gradients in jobResults[1]:
                                value_gradients.append(gradients)
                            if curr_episode % args_dict['Duelling_Update_Freq'] == 0:
                                for gradients in value_gradients:
                                    apply_gradients_duelling(global_model, gradients, sess)
                                value_gradients = []
                    else:
                        pass

                elif args_dict['JOB_TYPE'] == args_dict['JOB_OPTIONS'].getExperience:
                    global_metrics.reset()
                    global_metrics.episode_length = jobResults[2]
                    global_model.episode_number = curr_episode
                    train_metrics = calculateGradient(args_dict=args_dict,
                                                      sess=sess,
                                                      global_model=global_model,
                                                      buffer=jobResults[0],
                                                      variables=jobResults[1],
                                                      metrics=global_metrics)
                    metrics['Losses'] = train_metrics

                # update to tensorboard
                tensorboard_writer.update(metrics, curr_episode, run, wandb_run)

                # get the updated weights from the global network
                weight_names = tf.trainable_variables()
                weights = sess.run(weight_names)

                # updating old weights
                if args_dict['Target_Net']:
                    if curr_episode % args_dict['Target_Update_Freq'] == 0:
                        old_weights = weights

                if reinit_count > args_dict['RAY_RESET_EPS']:
                    reinitialize, meta_agents = reinit_ray(jobList, meta_agents, args_dict)
                    if reinitialize:
                        reinit_count = args_dict['NUM_META_AGENTS']
                        if args_dict['Target_Net']:
                            jobList, curr_episode = set_ray_jobs(meta_agents, weights + old_weights,
                                                                 curr_episode)  # Ray ObjectIDs
                        else:
                            jobList, curr_episode = set_ray_jobs(meta_agents, weights, curr_episode)  # Ray ObjectIDs

                # start a new job on the recently completed agent with the updated weights
                else:
                    if curr_episode < (args_dict['epochs']):
                        if args_dict['Target_Net']:
                            jobList.extend([meta_agents[info["id"]].job.remote(weights+old_weights, curr_episode)]) # Ray ObjectIDs
                        else:
                            jobList.extend([meta_agents[info["id"]].job.remote(weights, curr_episode)])
                        reinit_count += 1
                        curr_episode += 1

                # if curr_episode % 100 == 0:
                if curr_episode % args_dict['SAVE_STEP'] == 0:
                    avg_return = np.mean(returns[-min(len(returns), args_dict['NUM_META_AGENTS'])])
                    returns = []
                    if avg_return > best_return and args_dict['store_best_model']:
                        best_return = avg_return
                        print('Best Return:', best_return)
                        print('New Best Model Found, Saving it', end='\n')
                        saver.save(sess, args_dict['model_path'] + '/model-' + str(int(curr_episode)) + '.cptk')
                        print('Saved Model', end='\n')

                    elif not args_dict['store_best_model']:
                        if avg_return > best_return:
                            best_return = avg_return
                        print('Saving Model', end='\n')
                        saver.save(sess, args_dict['model_path'] + '/model-' + str(int(curr_episode)) + '.cptk')
                        print('Saved Model', end='\n')

                    print('Current Avg Return:', round(avg_return, 2), 'Best return so far:', round(best_return, 2))

            print('FINISHED THE ASSIGNED JOB!')

        except KeyboardInterrupt:
            print("CTRL-C pressed. killing remote workers")
            for a in meta_agents:
                ray.kill(a)
if __name__ == "__main__":
    main(10)