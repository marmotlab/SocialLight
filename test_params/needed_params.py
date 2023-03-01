share_horizon          = False           # Only required for long horizon methods (Share horizon predictions of neighbours as inputs to the network)
share_horizon_LSTM     = False          # Only required for long horizon methods (Share horizon predictions of neighbours as inputs to the LSTM prediction framework)
batch_gradient         = False    
dynamics               = False          # Keep this at false unless using DynamicsTraffic model, this will definitely break the code otherwise
intrinsic_dynamics_rewards = False      # Keep this at false unless using DynamicsTraffic model, this will definitely break the code otherwise
dynamics_LR            = 1.e-5           # Keep this at default, it will not be used until model is Dynamic
Environment            = 'Manhattan'     # Make sure non parameter sharing models work if setting environment to Monaco 
parameter_sharing      = True            
ext_value_head         = False           # To ensure a well defined loss
schedule_weights         = [1,1] 