# IMPORTANT CLASSES TO SET 
CLASS_CONFIGURATION  = 'SocialLight'
Observer             = 'CityFlowV2Observer'
Environment          = 'CityFlowV2' # Options [Manhattan,CityFlowV2]
ComputeValue = False

if Environment == 'Manhattan':
    NUM_RL_AGENT           = 25     # NUMBER OF RL AGENTS
elif Environment == 'Monaco':
    NUM_RL_AGENT           = 28     # NUMBER OF RL AGENTS
else :
    pass

if CLASS_CONFIGURATION == 'ActorCritic' :
    Model       = 'ActorCritic'              # Options [VanillaTraffic, LSTMTraffic,LongHorizon,GNNModel]
    Metric      = 'VanillaTrafficMetrics'       # Options [VanillaTrafficMetrics, LongHorizonMetrics]
    Variable    = 'Vanilla_Variables'   # Options [Vanilla_Variables, LSTM_Variables,Long_Horizon_Variables,GNNVariables]
    Buffer      = 'VanillaBuffer'       # Options [VanillaBuffer, LSTMBuffer,Long_HorizonBuffer,GNNBuffer]
    Network     = 'custom_mlp'         # Options [vanilla_fc , vanilla_LSTM, custom_mlp , custom_mlp_LSTM, long_horizon_mlp]


if CLASS_CONFIGURATION == 'SocialLight' :
    Model       = 'SocialLight'
    Metric      = 'ComaTrafficMetricsV2'
    Variable    = 'COMA_VariablesV3'
    Buffer      = 'COMABufferV3'
    Network     = 'coma_mlp'
    ComputeValue = True


if CLASS_CONFIGURATION == 'DecCOMA' :
    Model       = 'DecCOMA'
    Metric      = 'ComaTrafficMetricsV2'
    Variable    = 'COMA_VariablesV3'
    Buffer      = 'COMABufferV3'
    Network     = 'coma_mlpv3'
    ComputeValue = True

