from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os
import pandas as pd
import numpy as np

#Copy paste from summary
def get_metrics(duration_list, traffic_name, total_summary_metrics, num_of_out):
    # calculate the mean final 10 rounds
    validation_duration_length = 10
    duration_list = np.array(duration_list)
    validation_duration = duration_list[-validation_duration_length:]
    validation_through = num_of_out[-validation_duration_length:]
    final_through = np.round(np.mean(validation_through), decimals=2)
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    total_summary_metrics["traffic"].append(traffic_name.split(".json")[0])
    total_summary_metrics["final_duration"].append(final_duration)
    total_summary_metrics["final_duration_std"].append(final_duration_std)
    total_summary_metrics["final_through"].append(final_through)

    return total_summary_metrics

def test(model_dir,
         cnt_round,
         run_cnt,
         _dic_traffic_env_conf,
         seeds=None,
         final_tests = False,
         env = None,
         round_num=None):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    if final_tests:
        if round_num is None:
            round_num = _dic_traffic_env_conf['NUM_ROUNDS'] - 1
        model_round = "round_%d" % (round_num)
        test_round_id = "round_%d" % (cnt_round)
    else:
        model_round = "round_%d" % (cnt_round)
    test_round_id = "round_%d" % (cnt_round)
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}

    if not os.path.exists(records_dir):
        os.makedirs(records_dir)

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if final_tests:
        test_results_dir = "test_final_round"
    else:
        test_results_dir = "test_round"

    if dic_traffic_env_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        print(dic_agent_conf)
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    try:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
        if final_tests:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], test_results_dir, test_round_id)
        else:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], test_results_dir, model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        if env is None:
            env = CityFlowEnv(
                path_to_log=path_to_log,
                path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                dic_traffic_env_conf=dic_traffic_env_conf
            )

        done = False

        step_num = 0

        total_time = dic_traffic_env_conf["RUN_COUNTS"]
        if seeds is not None:

            state = env.reset(seedNum=seeds)
        else:
            state = env.reset()

        traffic_light_node_dict = env._adjacency_extraction()
        agentID = list(traffic_light_node_dict.keys())
        pre_actions_dict =  [ None for _ in agentID]
        total_action_change = []

        while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                if dic_traffic_env_conf["MODEL_NAME"] in ["PressLight", "GAT", "DQN", "FRAP", "Attention"]:
                    one_state = state
                    action_list = agents[i].choose_action(step_num, one_state)
                    sum_action_change = 0
                    for id,ID in enumerate(agentID):
                        if pre_actions_dict[id] is not None and pre_actions_dict[id] != action_list[id]:
                            sum_action_change += 1
                    pre_actions_dict = action_list
                    sum_action_change/=len(action_list)
                    total_action_change.append(sum_action_change)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1

        print("Total Action Change : {}".format(np.mean(total_action_change)))

        env.batch_log_2()
        env.end_cityflow()
    except:
        print("============== error occurs in model_test ============")
        print()


def calculate_summary(memo,traffic_file,round_file = None):
    """
    Calculate the average trip time each episode
    """
    # dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    # records_dir = model_dir.replace("model", "records")
    # model_round = "round_%d" % cnt_round
    # dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}

    records_dir = os.path.join("records", memo)

    total_summary = {
        "traffic": [],
        "final_duration": [],
        "final_duration_std": [],
        "final_through": [],
    }

    if ".json" not in traffic_file:
        return
    print(traffic_file)

    traffic_env_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')
    dic_traffic_env_conf = json.load(traffic_env_conf)
    run_counts = dic_traffic_env_conf["RUN_COUNTS"]
    num_intersection = dic_traffic_env_conf['NUM_INTERSECTIONS']
    duration_each_round_list = []
    num_of_vehicle_in = []
    num_of_vehicle_out = []
    test_round_dir = os.path.join(records_dir, traffic_file, "test_final_round")

    try:
        round_files = os.listdir(test_round_dir)
    except:
        print("no test round in {}".format(traffic_file))
        return

    round_files = [f for f in round_files if "round" in f]
    round_files.sort(key=lambda x: int(x[6:]))
    if round_file is not None:
        df_vehicle_all = []
        for inter_index in range(num_intersection):
            try:
                round_dir = os.path.join(test_round_dir, round_file)
                df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                               sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                               names=["vehicle_id", "enter_time", "leave_time"])
                # [leave_time_origin, leave_time, enter_time, duration]
                df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)
                df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                               df_vehicle_inter["enter_time"].values
                tmp_idx = []
                for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                    if "shadow" in v:
                        tmp_idx.append(i)
                df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                # print("------------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                df_vehicle_all.append(df_vehicle_inter)
            except:
                print("======= Error occured during reading vehicle_inter_{}.csv")

        if len(df_vehicle_all) == 0:
            print("====================================EMPTY")

        df_vehicle_all = pd.concat(df_vehicle_all)
        # calculate the duration through the entire network
        vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
        ave_duration = vehicle_duration.mean()  # mean amomng all the vehicle

        duration_each_round_list.append(ave_duration)

        num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
        num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

        # print("==== round: {0}\tave_duration: {1}\tnum_of_vehicle_in:{2}\tnum_of_vehicle_out:{2}"
        #       .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))
        duration_flow = vehicle_duration.reset_index()
        duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x: x.split('_')[1])
        duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
        print("Average duration {}".format(np.mean(duration_each_round_list)))
    else:
        for round_rl in round_files:
            df_vehicle_all = []
            for inter_index in range(num_intersection):
                try:
                    round_dir = os.path.join(test_round_dir, round_rl)
                    df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                   sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                   names=["vehicle_id", "enter_time", "leave_time"])
                    # [leave_time_origin, leave_time, enter_time, duration]
                    df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                    df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)
                    df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                                   df_vehicle_inter["enter_time"].values
                    tmp_idx = []
                    for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                        if "shadow" in v:
                            tmp_idx.append(i)
                    df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                    ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                    #print("------------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                    df_vehicle_all.append(df_vehicle_inter)
                except:
                    print("======= Error occured during reading vehicle_inter_{}.csv")

            if len(df_vehicle_all) == 0:
                print("====================================EMPTY")
                continue

            df_vehicle_all = pd.concat(df_vehicle_all)
            # calculate the duration through the entire network
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()  # mean amomng all the vehicle

            duration_each_round_list.append(ave_duration)

            num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
            num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

            # print("==== round: {0}\tave_duration: {1}\tnum_of_vehicle_in:{2}\tnum_of_vehicle_out:{2}"
            #       .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))
            duration_flow = vehicle_duration.reset_index()
            duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x: x.split('_')[1])
            duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
            #print(duration_flow_ave)
        print("Average duration {}".format(np.mean(duration_each_round_list)))

    result_dir = os.path.join("summary", memo, traffic_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    _res = {
        "duration": duration_each_round_list,
        "vehicle_in": num_of_vehicle_in,
        "vehicle_out": num_of_vehicle_out
    }
    result = pd.DataFrame(_res)
    result.to_csv(os.path.join(result_dir, "test_results.csv"))
    total_summary_rl = get_metrics(duration_each_round_list, traffic_file, total_summary, num_of_vehicle_out)
    total_result = pd.DataFrame(total_summary_rl)
    total_result.to_csv(os.path.join("summary", memo, "total_final_test_results.csv"))
    return np.mean(duration_each_round_list)