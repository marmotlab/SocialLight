from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from multiprocessing import Process
import argparse
import os
from utils.model_test import test,calculate_summary,get_metrics
from utils.cityflow_env import CityFlowEnv
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo",       type=str,           default='benchmark_1001')
    parser.add_argument("-mod",        type=str,           default="Attention")
    parser.add_argument("-eightphase",  action="store_true", default=False)
    parser.add_argument("-gen",        type=int,            default=1)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("-workers",    type=int,            default=3)
    parser.add_argument("-hangzhou",    action="store_true", default=False)
    parser.add_argument("-jinan",       action="store_true", default=True)
    parser.add_argument("-newyork", action="store_true", default=False)
    parser.add_argument("-start_seed",type=int,default=4000)
    parser.add_argument("-num_tests",type=int,default=20)
    parser.add_argument("-round_num",type=int,default=499)
    return parser.parse_args()

def main(in_args=None):

    if in_args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",
                             "anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 500
        template = "Hangzhou"
    elif in_args.jinan:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json"]
        num_rounds = 500
        template = "Jinan"
    elif in_args.newyork:
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        num_rounds = 500
        template = "newyork_28_7"

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []
    for traffic_file in traffic_file_list:


        dic_traffic_env_conf_extra = {
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,

            "MODEL_NAME": in_args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "TRAFFIC_SEPARATE": traffic_file,
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_queue_vehicle_in",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },
        }

        if in_args.mod =='GAT':
            dic_traffic_env_conf_extra["LIST_STATE_FEATURE"]= [
                "cur_phase",
                "lane_num_vehicle_in",
                "adjacency_matrix",
            ]


        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                    1: [0, 1, 0, 1, 0, 0, 0, 0],
                    2: [0, 0, 0, 0, 0, 1, 0, 1],
                    3: [1, 0, 1, 0, 0, 0, 0, 0],
                    4: [0, 0, 0, 0, 1, 0, 1, 0],
                    5: [1, 1, 0, 0, 0, 0, 0, 0],
                    6: [0, 0, 1, 1, 0, 0, 0, 0],
                    7: [0, 0, 0, 0, 0, 0, 1, 1],
                    8: [0, 0, 0, 0, 1, 1, 0, 0]
                }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        # dic_path_extra = {
        #     "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_"
        #                                   + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        #     "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_"
        #                                            + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        #     "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
        #     "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        # }
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        }

        deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")


        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
        test_round_id = "round_%d" % (0)
        path_to_log = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "test_final_round", test_round_id)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=deploy_dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=deploy_dic_traffic_env_conf
        )
        duration_list = []
        for j in range(in_args.num_tests):
            test(model_dir= deploy_dic_path["PATH_TO_MODEL"],
                 cnt_round=j,
                 run_cnt= deploy_dic_traffic_env_conf["RUN_COUNTS"],
                 _dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                 seeds=in_args.start_seed+j*500,
                 final_tests=True,
                 env = env,
                 round_num = in_args.round_num
                 )
            duration_list.append(calculate_summary(in_args.memo,traffic_file,round_file=test_round_id))
        result_dir = os.path.join("summary", in_args.memo, traffic_file)
        output_evaluation_data(mean=np.mean(np.array(duration_list)),
                                min=np.min(np.array(duration_list)),
                                max=np.max(np.array(duration_list)),
                                all=np.array(duration_list),
                               result_dir = result_dir)

def output_evaluation_data(**kwargs):
    """
    Output and save the evaluation results/average trip time
    """
    result_dir = kwargs['result_dir']
    with open(result_dir + "/result.txt", "w+") as f:
        f.write("Average travel time is {0} \n".format(kwargs['mean']))
        f.write("Max travel time is {0} \n".format(kwargs['max']))
        f.write("Min travel time is {0} \n".format(kwargs['min']))
        f.write("All average travel time is {0} \n".format(kwargs['all']))
    print("|| average travel time is {} \n".format( kwargs['mean']))

if __name__ == "__main__":
    args = parse_args()

    main(args)