# MAJOR TRAINING PARAMETERS 
Framework              = 'tf'           # Options [torch or tf]
device                 = 'gpu'          # Cuda or pytorch for tensorflow 2 and pytorch (chnage GPU parameter)

NUM_META_AGENTS        = 2             # *** VERY IMPORTANT PARAMETER - SETS NUMBER OF THREADS

parameter_sharing      = True           # *** VERY IMPORTANT PARAMETER - DECIDES WHETHER SINGLE NETWORK (SHARED PARAMETERS)/ MULTIPLE NETWORKS TO BE USED
load_model             = False       # load an existing model
OUTPUT_GIFS            = False          # output gifs for visualization
epochs                 = 45000            # *** VERY IMPORTANT PARAMETER - NUMBER OF EPISODES TO TRAIN FOR
store_best_model       = True

# OBSERVER AND REWARD STRUCTURE PARAMETERS
NEIGH_OBS              = True               # Append the observation of neighbours to an agent's observation
REWARD_SHARING         = True             # Add the average discounted reward of neighbours to an agent's reward
Neighbor_factor        = 1.0                # Discount factor if REWARD_SHARING = TRUE
REWARD_STRUCTURE       = 'truncated_queue'  # Options ['queue', 'truncated_queue', 'suc_waiting', 'acc_waiting+out', 'queue+suc_waiting', 'queue+acc_waiting', 'queue+acc_waiting+out', 'max_pressure']
Waiting_factor         = 1
share_type             = False              # Share type of agent (this is based on the number of neighbours it has and their orientation)

# Whether to use agent gradients or batch gradients
batch_gradient         = False           # Use a random minibatch for training
batch_size             = 512            # Number of samples in a batch
train_iterations       = 10             # Number of iterations of SGD

# TESTING OR TRAINING
TRAINING               = True
TESTING                = False
TEST_NUM               = 10

# SPECIFY TYPE OF JOB
class JOB_OPTIONS:
    getExperience      = 1
    getGradient        = 2

class COMPUTE_OPTIONS:
    multiThreaded      = 1
    synchronous        = 2

# JOB_TYPE = JOB_OPTIONS.getGradient\
JOB_TYPE               = JOB_OPTIONS.getGradient
COMPUTE_TYPE           = COMPUTE_OPTIONS.synchronous

#LESS USEFUL CONSTANT PARAMS
NUM_ACTIVE             = 1              # each scenario only has 1 learning agent
NUM_PASSIVE            = 0
SINGLE_GRADIENT        = False          # one gradient for all agents or per agent within a single meta-agent
WRITE_WAIT_TIMES       = False
SUMMARY_WINDOW         = 10             # tensorboard update frequency
GPU                    = False          # use of GPU
RAY_RESET_EPS          = 5000           # resetting Ray
EXPERIENCE_BUFFER_SIZE = 3600           # maximum size of experience buffer (not really imporant)
EVAL_AGENTS            = [0,1,2,3]      # meta-agents which compute traffic metrics (high compute)

# Duelling Params
Duelling = False
Duelling_Update_Freq = 10

# Target Network
#Training value functions with an old netowrk
Target_Net = False
Target_Update_Freq = 50