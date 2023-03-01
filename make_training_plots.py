import pandas
import argparse
import typing
from typing import List,Tuple
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_path",       type=str,           default="Plots")
    parser.add_argument("-plot_path",        type=str,           default="Rewards-A3CvsSocialLight-ManhattanSUMO")
    return parser.parse_args()

colors_list = ['tab:red', 'tab:green','tab:purple']
def read_csv(folder_name):
    '''
    :param file_name: name of the file
    :return: list
    '''

    # Obtain CSV files
    csv_files = []
    if os.path.exists(folder_name):
        csv_files = os.listdir(folder_name)

    if csv_files is None:
        raise Exception('No CSV files')

    # Empty DF list


    df_dict = dict()
    for csv in csv_files:
        csv_path = folder_name + '/' + csv
        key = csv_path.split('_')[0].split('/')[-1]
        if csv[-4:] =='.csv':
            df_dict[key]=pd.read_csv(csv_path)




    return df_dict

def plot_time_series(time:dict, std_series:dict,series:dict,args,metric:str):
    '''

    :param time: Time series dictionary
    :param std_series: Rolling standard deviation
    :param min_series: Rolling mean of the series
    :return:
    '''
    plt.figure()

    legend = time.keys()
    idx = 0
    for model in legend:

        if model=='SocialLight':
            color = 'tab:blue'
            plt.plot(time[model], series[model], label=model, linewidth=1,
                     alpha=0.8,color = color)
            plt.fill_between(time[model], series[model]-2*std_series[model], series[model]+1.0*std_series[model]
                             , alpha=0.2, color=color)
        else:
            plt.plot(time[model], series[model], label=model, linewidth=1,
                     alpha=0.8,color=colors_list[idx])
            plt.fill_between(time[model], series[model] - 2*std_series[model], series[model] + 1.0*std_series[model]
                             , alpha=0.2,color=colors_list[idx])
            idx += 1

    split_path = args.plot_path.split('-')
    plt.title('Average ' + split_path[0], fontsize=15)
    plt.xlabel(f'Env Iterations', fontsize=15)
    plt.ylabel('Average ' + metric, fontsize=15)
    # if split_path[0].contains('speed'):
    #     plt.ylabel(split_path[0] +'m/s', fontsize=15)
    # elif split_path[0].contains('time'):
    #     plt.ylabel(split_path[0] +'time', fontsize=15)
    # else:
    #     plt.ylabel(split_path[0], fontsize=15)
    plt.legend(legend)
    plt.savefig('./'+args.folder_path+'/'+args.plot_path+'/plots.jpg')
    #plt.show()



def get_time_series_data(df_dict: List[pd.DataFrame]) -> Tuple[np.array,np.array]:
    '''

    :param df_list: List of dataframes from which we extract the time series data
    :return: Returns a single dataframe consisting of all time series data with moving averages
    '''

    # Run through all dataframes
    series_dict = dict()
    #max_series_dict = dict()
    std_series_dict = dict()
    time_series = dict()
    for key in df_dict.keys():
        metric = df_dict[key].columns[-1]
        series = df_dict[key][metric]\
            .rolling(window=10,min_periods=1)\
            .mean()\
            .values
        std_series = df_dict[key][metric] \
            .rolling(window=10, min_periods=1) \
            .std() \
            .values
        # min_series = df_dict[key]['Reward'] \
        #     .rolling(window=10, min_periods=1) \
        #     .min() \
        #     .values
        series_dict[key] = series
        std_series_dict[key] = std_series
        #min_series_dict.append(min_series)
        time_series[key] = df_dict[key]['Training Env Iteration'].values
    # Ceate a  new dataframe with the rolling window sums

    return time_series,series_dict,std_series_dict,metric

if __name__ == "__main__":

    args = parse_args()

    folder_name = args.folder_path + "/" + args.plot_path
    df_list = read_csv(folder_name)
    time_series, series_dict, std_series_dict,metric = get_time_series_data(df_list)
    plot_time_series(time_series, std_series_dict, series_dict,args,metric)




