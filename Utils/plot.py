import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
# ma2c plots
episode_sec = 3600


def fixed_agg(xs, window, agg):
    xs = np.reshape(xs, (-1, window))
    if agg == 'sum':
        return np.sum(xs, axis=1)
    elif agg == 'mean':
        return np.mean(xs, axis=1)
    elif agg == 'median':
        return np.median(xs, axis=1)


def varied_agg(xs, ts, window, agg):
    x_stat = None
    t_bin = window
    x_bins = []
    cur_x = []
    xs = list(xs) + [0]
    ts = list(ts) + [episode_sec + 1]
    i = 0
    while i < len(xs):
        x = xs[i]
        t = ts[i]
        if t <= t_bin:
            cur_x.append(x)
            i += 1
        else:
            if not len(cur_x):
                x_bins.append(0)
            else:
                if agg == 'sum':
                    x_stat = np.sum(np.array(cur_x))
                elif agg == 'mean':
                    x_stat = np.mean(np.array(cur_x))
                elif agg == 'median':
                    x_stat = np.median(np.array(cur_x))
                x_bins.append(x_stat)
            t_bin += window
            cur_x = []
    return np.array(x_bins)


def plot_series(df, name, tab, label, color, window=None, agg='sum', reward=False):
    episodes = list(df.episode.unique())
    num_episode = len(episodes)
    num_time = episode_sec
    # always use avg over episodes
    if tab != 'trip':
        res = df.loc[df.episode == episodes[0], name].values
        for episode in episodes[1:]:
            res += df.loc[df.episode == episode, name].values
        res = res / num_episode
        print('\nagent: {}'.format(label))
        print('metric:{}'.format(name))
        print('mean: %.3f' % np.mean(res))
        print('std: %.3f' % np.std(res))
        print('min: %.3f' % np.min(res))
        print('max: %.3f' % np.max(res))
    else:
        res = []
        for episode in episodes:
            res += list(df.loc[df.episode == episode, name].values)
        print('\nagent: {}'.format(label))
        print('metric:{}'.format(name))
        print('mean: %.3f' % np.mean(res))
        print('std: %.3f' % np.std(res))
        print('max: %.3f' % np.max(res))

    if reward:
        num_time = 720
    if window and (agg != 'mv'):
        num_time = num_time // window
    x = np.zeros((num_episode, num_time))
    for i, episode in enumerate(episodes):
        t_col = 'arrival_sec' if tab == 'trip' else 'time_sec'
        cur_df = df[df.episode == episode].sort_values(t_col)
        if window and (agg == 'mv'):
            cur_x = cur_df[name].rolling(window, min_periods=1).mean().values
        else:
            cur_x = cur_df[name].values
        if window and (agg != 'mv'):
            if tab == 'trip':
                cur_x = varied_agg(cur_x, df[df.episode == episode].arrival_sec.values, window, agg)
            else:
                cur_x = fixed_agg(cur_x, window, agg)
        #         print(cur_x.shape)
        x[i] = cur_x
    if num_episode > 1:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
    else:
        x_mean = x[0]
        x_std = np.zeros(num_time)
    if (not window) or (agg == 'mv'):
        t = np.arange(1, episode_sec + 1)
        if reward:
            t = np.arange(5, episode_sec + 1, 5)
    else:
        t = np.arange(window, episode_sec + 1, window)
    #     if reward:
    #         print('%s: %.2f' % (label, np.mean(x_mean)))
    plt.plot(t, x_mean, color=color, linewidth=3, label=label)
    if num_episode > 1:
        x_lo = x_mean - x_std
        if not reward:
            x_lo = np.maximum(x_lo, 0)
        x_hi = x_mean + x_std
        plt.fill_between(t, x_lo, x_hi, facecolor=color, edgecolor='none', alpha=0.1)
        return np.nanmin(x_mean - 0.5 * x_std), np.nanmax(x_mean + 0.5 * x_std)
    else:
        return np.nanmin(x_mean), np.nanmax(x_mean)


def plot_combined_series(dfs, agent_names, col_name, tab_name, agent_labels, y_label, fig_name,
                         window=None, agg='sum', reward=False, plot_dir=None, color=None):
    plt.figure(figsize=(9, 6))
    ymin = np.inf
    ymax = -np.inf
    for i, aname in enumerate(agent_names):
        df = dfs[aname][tab_name]
        y0, y1 = plot_series(df, col_name, tab_name, agent_labels[i], color[aname], window=window, agg=agg,
                             reward=reward)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    plt.xlim([0, episode_sec])
    if (col_name == 'average_speed') and ('global' in agent_names):
        plt.ylim([0, 6])
    elif (col_name == 'wait_sec') and ('global' not in agent_names):
        plt.ylim([0, 2000])
    else:
        plt.ylim([ymin, ymax])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Simulation time (sec)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    # plot_dir = './mytest/plots'
    plt.savefig(plot_dir + ('/%s.pdf' % fig_name))
    plt.close()


def sum_reward(x):
    x = [float(i) for i in x.split(',')]
    return np.sum(x)


def plot_eval_curve(scenario='large_grid', plot_dir=None, cur_dir=None, names=None):
    # cur_dir = './mytest/eval_data'
    # names = ['DRL', 'Greedy', 'ia2c', 'ma2c']
    # labels = ['DRL', 'Greedy', 'ia2c', 'ma2c']
    # color = {'DRL': color_cycle[0], 'Greedy': color_cycle[1], 'ma2c': color_cycle[2], 'ia2c': color_cycle[3]}
    labels = names
    sns.set_color_codes()
    color = {}
    color_cycle = sns.color_palette()
    for i in range(len(names)):
        color[names[i]] = color_cycle[i]

    dfs = {}
    for file in os.listdir(cur_dir):
        if not file.endswith('.csv'):
            continue
        if not file.startswith(scenario):
            continue
        name = file.split('_')[2]
        measure = file.split('_')[3].split('.')[0]
        if name in names:
            df = pd.read_csv(cur_dir + '/' + file)
            #             if measure == 'traffic':
            #                 df['ratio_stopped_car'] = df.number_stopped_car / df.number_total_car * 100
            #             if measure == 'control':
            #                 df['global_reward'] = df.reward.apply(sum_reward)
            if name not in dfs:
                dfs[name] = {}
            dfs[name][measure] = df

    # plot avg queue
    plot_combined_series(dfs, names, 'avg_queue', 'traffic', labels,
                         'Average queue length (veh)', scenario + '_queue', window=60, agg='mv', plot_dir=plot_dir, color=color)
    # # # plot avg speed
    plot_combined_series(dfs, names, 'avg_speed_mps', 'traffic', labels,
                         'Average car speed (m/s)', scenario + '_speed', window=60, agg='mv', plot_dir=plot_dir, color=color)
    # # # plot avg waiting time
    plot_combined_series(dfs, names, 'avg_wait_sec', 'traffic', labels,
                         'Average intersection delay (s/veh)', scenario + '_wait', window=60, agg='mv', plot_dir=plot_dir, color=color)
    # # plot trip completion
    plot_combined_series(dfs, names, 'number_arrived_car', 'traffic', labels,
                         'Trip completion rate (veh/5min)', scenario + '_tripcomp', window=300, agg='sum', plot_dir=plot_dir, color=color)
    # # plot trip time
    plot_combined_series(dfs, names, 'duration_sec', 'trip', labels,
                         'Avg trip time (sec)', scenario + '_triptime', window=60, agg='mean', plot_dir=plot_dir, color=color)
    # #     plot trip waiting time
    plot_combined_series(dfs, names, 'wait_sec', 'trip', labels,
                         'Avg trip delay (s)', scenario + '_tripwait', window=60, agg='mean', plot_dir=plot_dir, color=color)


# Original Plots
def write_file(path, object):
    filename = path
    with open(filename, 'w') as file_obj:
        json.dump(object, file_obj)


def load_file(path):
    filename = path
    with open(filename) as file_obj:
        object = json.load(file_obj)
    return object


def plot_steps(steps, label):
    plt.figure()
    plt.plot(np.arange(len(steps[0])), steps[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(steps[1])), steps[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(steps[2])), steps[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode vias Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 50])
    plt.show()


def plot_rewards(rewards, label):
    plt.figure()
    plt.plot(np.arange(len(rewards[0])), rewards[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(rewards[1])), rewards[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(rewards[2])), rewards[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode via Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, -50])
    plt.show()


def plot_steps_scatter(steps, steps_average, label, label_average):
    plt.figure()
    plt.scatter(np.arange(len(steps[0])), steps[0], label=label[0], alpha=0.8, s=3, c='r')
    plt.scatter(np.arange(len(steps[1])), steps[1], label=label[1], alpha=0.8, s=3, c='g')
    plt.scatter(np.arange(len(steps[2])), steps[2], label=label[2], alpha=0.8, s=3, c='b')

    plt.plot(np.arange(len(steps_average[0])), steps_average[0], 'r', label=label_average[0], linewidth=1)
    plt.plot(np.arange(len(steps_average[1])), steps_average[1], 'g', label=label_average[1], linewidth=1)
    plt.plot(np.arange(len(steps_average[2])), steps_average[2], 'b', label=label_average[2], linewidth=1)

    plt.text(90, steps_average[0][1] + 0.05, '%.4f' % steps_average[0][1], ha='center', va='top', fontsize=9)
    plt.text(90, steps_average[1][1] + 0.05, '%.4f' % steps_average[1][1], ha='center', va='bottom', fontsize=9)
    plt.text(90, steps_average[2][1] + 0.05, '%.4f' % steps_average[2][1], ha='center', va='bottom', fontsize=9)

    plt.title('Episode via Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 50])
    plt.show()


def plot_rewards_scatter(rewards, rewards_average, label, label_average):
    plt.figure()
    plt.scatter(np.arange(len(rewards[0])), rewards[0], label=label[0], alpha=0.8, s=3, c='r')
    plt.scatter(np.arange(len(rewards[1])), rewards[1], label=label[1], alpha=0.8, s=3, c='g')
    plt.scatter(np.arange(len(rewards[2])), rewards[2], label=label[2], alpha=0.8, s=3, c='b')

    plt.plot(np.arange(len(rewards_average[0])), rewards_average[0], 'r', label=label_average[0], linewidth=1)
    plt.plot(np.arange(len(rewards_average[1])), rewards_average[1], 'g', label=label_average[1], linewidth=1)
    plt.plot(np.arange(len(rewards_average[2])), rewards_average[2], 'b', label=label_average[2], linewidth=1)

    plt.text(50, rewards_average[0][1] + 0.05, '%.4f' % rewards_average[0][1], ha='center', va='top', fontsize=9)
    plt.text(70, rewards_average[1][1] + 0.05, '%.4f' % rewards_average[1][1], ha='center', va='top', fontsize=9)
    plt.text(90, rewards_average[2][1] + 0.05, '%.4f' % rewards_average[2][1], ha='center', va='bottom', fontsize=9)

    plt.title('Episode via Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-5, 1])
    plt.show()
