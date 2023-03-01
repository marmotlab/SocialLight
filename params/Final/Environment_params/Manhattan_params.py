# MANHATTAN_PARAMS:
# Environment parameters
MAX_EPISODE_LENGTH     = 720

MAX_EPISODE_SUMO_STEPS = 3600

a_size                 = 5      # ACTION SIZE
NUM_TLS                = 25     # NUMBER OF INTERSECTIONS
NUM_INCOMING           = 6       # INCOMING LANES
NUM_LANES              = 12      # TOTAL LANES

# SUMO SETTINGS
GUI                    = False   # TRAIN WITH GUI
GREEN_DURATION         = 5      # GREEN LIGHT PHASE DURATION
YELLOW_DURATION        = 2       # YELLOW LIGHT PHASE DURATION
ALL_RED_DURATION       = 2       # RED LIGHT PHASE DURATION

# Baseline route setting
peak_flow1             = 1100
peak_flow2             = 925
init_density           = 0
TELEPORT_TIME          = "300"
RANDOM_SEED            = True
SEED                   = 12

# Net file path
MAP_FILE_PATH          = "./Map/Manhattan_map/data_baseline/"