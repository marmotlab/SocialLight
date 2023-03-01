# Network hyperparamters

hidden_sizes                = [64,128,128]    # Optional argument, only required for custom built networks
gamma                       = .95            # discount rate for advantage estimation and reward discounting
beta                        = 0.8
RNN_SIZE                    = 64             # Only required if using LSTM/long horizon methods


# Set to very negative large number if a constant weight is needed
pretraining_episodes        = 20000            # Don't give intrinsic rewards for some N pretraining episodes
SAVE_STEP                   = 500

# Network Weights
policy_weight               = 1.0
value_weight                = 0.01
q_value_weight                = 0.04
entropy_weight              = 0.0005

# COMA Network params
coop_weight_decay = 1.0/15000.0
cooperation = 1.0

# Lambda returns
LAMBDA = True

# Only required for long horizon methods
prediction_weight           = 0.2
policy_prediction_weight    = 0

# PARAMS WHICH STAY CONSTANT
GLOBAL_NET_SCOPE            = 'global'
ADAPT_COEFF                 = 5.e-5          # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
LR_Q                        = 1.e-5          # / (EXPERIENCE_BUFFER_SIZE / 128) # default (conservative, for 256 steps): 1e-5
GRAD_CLIP                   = 100.0

# For attention network
ATTENTION                   = False
ATT_HIDDEN_SIZE             = [32, 64, 128, 128]
NUM_HEADS                   = 8
MODEL_DIM                   = 128
FF_DIM                      = 256

# For Global ppo
GLOBAL_PPO                  = False
PPO                         = False
N_EPOCH                     = 6