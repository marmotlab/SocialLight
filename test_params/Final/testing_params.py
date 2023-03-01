test_name       = 'A3C-RS-Manhattan' # Modify environment amnd important parmameters accordingly
agent_name      = test_name
model_name      = 'A3C-RS-Manhattan' # Modify environmemnt parameters and important parameters accordingly
EXP_NAME        = model_name
# model_path    = './Trained_Models/intrinsic-pretraining-v016_09_16/model_traffic_intrinsic-pretraining-v016_09_16'
model_path      = 'data/' + model_name + '/model_traffic_{}'.format(model_name)
model_load      = True
TEMP_GIF_FOLDER = 'data/' + test_name +'/TEMPORARY_gifs_traffic_{}'.format(test_name)