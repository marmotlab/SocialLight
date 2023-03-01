# SocialLight: 
Public code, baselines and models for SocialLight: Distributed Cooperation Learning towards
Network-Wide Traffic Signal Control accepted to AAMAS 2023 as a full paper.


# Running Code
## Requirements
All the requirements are specified in the requirements.txt file. 

The most important requirements are:
```
tensorflow==1.11.0
traci==1.13.0
scipy
ray==1.13.0
sumolib==1.13.0
gym==0.24.1
h5py4
pandas==1.1.5
neptune-client==0.15.2
numpy
wandb==0.12.21
```

Additionally, setup the cityflow environments and the tensorflow version 2.4.0 as mentioned for the baselines.

## Train
1 . Set parameters within the params folder. The environment, model and the corresponding observation spaces can be modified in the params/Final/important_params.py folder. 

2 .The flow files and urban network for cityflow environments can be changed from the params/Final/Environment_params/CityFlowV2_params.py folder. 
Flow file generation for the artificial manhattan map in sumo can be controlled from the params/Final/Environment_params/Manhattan_params.py

3. Additionally you can modify the network architecture or training hyperparameters through the params/Final/network_params.py and params/Final/training_params.py files.

4. Logging can be done through wandb or neptune, you can specify these parameters in params/Final/logging_params.py

After setting the respective parameters,call the following command to train the model.
```
xvfb-run -a python3 train.py
```

## Test
1. To perform testing refer to the run_commands.txt file for the run commands that we used to test our models. 

2. If you wish to test your own models, it is imperative that you copy your training parameters in the params folder into the the test_params folder.
Under utilities, run the generate_configs.py script to write the testing_params into a .config file which can be found in the configs/test_configs folder. Alternatively, you could choose to copy the config file geenrated during training automatically into the test configs folder.

3. Before testing your model, make sure to note that the important parameters in your test_params/Final/important_params.py match those in your test_config
folder as well as the environment parameters such as urban network or flow files params/Final/Environment_params/ match those in your test config. 

4. Follow the reference commands as specifed in the run_commands.txt to test your models

5. After setting the respective parameters,call the following command to train the model.

# Cite



# Authors
Harsh Goel

Zhang Yifeng

Mehul Damani

Guillaume Sartoretti
