# CITYFLOW_PARAMS:

# ['G6U', 'G6B', 'DN', 'DH', 'DJ']
NET_NAME               = 'DH'
MAX_EPISODE_LENGTH     = 720
MAX_EPISODE_SUMO_STEPS = 3600


# SUMO SETTINGS
GUI                    = False   # TRAIN WITH GUI
GREEN_DURATION         = 5      # GREEN LIGHT PHASE DURATION
YELLOW_DURATION        = 2       # YELLOW LIGHT PHASE DURATION
ALL_RED_DURATION       = 2       # RED LIGHT PHASE DURATION
a_size                 = 8

if NET_NAME == 'DN':
    BASE_FILE       = 'newyork_28_7/28_7/'
    NET_FILE        = BASE_FILE + 'roadnet_28_7.json'
    FLOW_FILE       = BASE_FILE + 'anon_28_7_newyork_real_double.json'
    # FLOW_FILE       = BASE_FILE + 'anon_28_7_newyork_real_triple.json'
    CONFIG_FILE     = BASE_FILE + 'roadnet_28_7_net_config.json'
    NUM_RL_AGENT    = 196
    NUM_TLS         = 196

elif NET_NAME == 'DH':
    BASE_FILE       = 'Hangzhou/4_4/'
    NET_FILE        = BASE_FILE + 'roadnet_4_4.json'
    #FLOW_FILE       = BASE_FILE + 'anon_4_4_hangzhou_real.json'
    # FLOW_FILE       = BASE_FILE + anon_4_4_hangzhou_real_5734.json'
    FLOW_FILE       = BASE_FILE + 'anon_4_4_hangzhou_real_5816.json'
    CONFIG_FILE     = BASE_FILE + 'roadnet_4_4_net_config.json'
    NUM_RL_AGENT    = 16
    NUM_TLS         = 16
    epochs          = 40000
    pretraining_episodes = 5000

elif NET_NAME == 'DJ':
    BASE_FILE       = 'Jinan/3_4/'
    NET_FILE        = BASE_FILE + 'roadnet_3_4.json'
    FLOW_FILE       = BASE_FILE + 'anon_3_4_jinan_real.json'
    # FLOW_FILE       = BASE_FILE + 'anon_3_4_jinan_real_2000.json'
    # FLOW_FILE       = BASE_FILE + 'anon_3_4_jinan_real_2500.json'
    CONFIG_FILE     = BASE_FILE + 'roadnet_3_4_net_config.json'
    NUM_RL_AGENT    = 12
    NUM_TLS         = 12
    epochs          = 40000
    pretraining_episodes = 5000


else:
    NET_FILE        = None
    FLOW_FILE       = None
    CONFIG_FILE     = None

DIR                 = './'
MAP_FILE_PATH       = 'Map/Cityflowv2_map'
LOG_DIR             = 'CFLOGS/{}'
WORK_DIR            = MAP_FILE_PATH
ROAD_NET_FILE       = NET_FILE
FLOW_FILE           = FLOW_FILE
CONFIG_FILE         = CONFIG_FILE
INTERVAL            = 1
SEED                = 0
RANDOM_SEED         = True
LANE_CHANGE         = False
RL_TRAFFIC_LIGHT    = True
SAVE_REPLAY         = False
ROAD_NET_LOG_FILE   = MAP_FILE_PATH + '/' + "frontend/web/roadnetLogFile_{}.json"
REPLY_LOG_FILE      = MAP_FILE_PATH + '/' + "frontend/web/replayLogFile_{}.txt"