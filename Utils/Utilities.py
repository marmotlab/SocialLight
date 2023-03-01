import numpy as np 
#from parameters import Framework
import copy 
import os 
import scipy.signal as signal
import subprocess
import sys 
import json 
import jsonpickle 
import datetime,pathlib 
import neptune.new as neptune 
 
def make_one_hot(self,array,a_size) :
    # Converts a np array into a one-hot np array 
    max_el = np.argmax(array)
    array = np.zeros((1,a_size))
    array[0][max_el] = 1 
    return array 

def process_shape(self,array,a_size) :
    # Converts array of shape (x,) into (x,y) 
    y= a_size
    array = np.stack(array)
    array = np.reshape(array,(-1,y)) 
    return array

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

'''
Returns TD(lamb) return
'''
def lambda_return(rewards,values,gamma,lamb):
    '''

    :param rewards: Rewards (batch,sequence)
    :param values: Values (batch,sequence)
    :param gamma: Discount
    :param lamb: Lamb weight
    :return: Lambda returns (batch,sequence)
    '''

    #shape (batch,T,T)
    rewards = rewards.copy()
    lambret = np.zeros(rewards.shape[-1])
    for j in range(rewards.shape[-1]):
        index = rewards.shape[-1] - j - 1


        if j ==0:
            if len(rewards.shape)==2:
                lambret[:,index] = rewards[:,index] + gamma*values[:,index+1]
            else:
                lambret[index] = rewards[index] + gamma * values[index + 1]
        else:
            if len(rewards.shape) == 2:
                lambret[:,index] = rewards[:,index] + gamma*(1-lamb)*values[:,index+1]\
                                + lamb*gamma*lambret[:,index+1]
            else:
                lambret[index] = rewards[index] + gamma * (1 - lamb) * values[index + 1] \
                                + lamb * gamma * lambret[index + 1]
    return  lambret

def make_gif(env, file_name):
    command = 'ffmpeg -framerate 5 -i "{tempGifFolder}/step%03d.png" {outputFile}'.format(tempGifFolder=env.gif_folder,
                                                                                          outputFile=file_name)

    os.system(command)

    deleteTempImages = "rm {tempGifFolder}/*".format(tempGifFolder=env.gif_folder)
    os.system(deleteTempImages)
    print("wrote gif")

def normalized_columns_initializer(std=1.0):
    import tensorflow as tf 
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

def index_to_ID(index,env,args_dict=None) :
    return env.index_to_id[index] 

def ID_to_index(id,env,args_dict=None) :
    return env.id_to_index[id] 
    
def get_sampled_actions(policy_dict,best = False):
    action_dict = {} 
    for k,v in policy_dict.items() : 
        size = v.shape[1]
        if not best:
            a = np.random.choice(range(size), p=v.ravel())
        else:
            a = np.argmax(v.ravel())
        action_dict[k] = a 
    return action_dict 

def make_list_dict(metrics) :
    tb_dict = {} 
    for k,v in metrics.items() :
        tb_dict[k] = {} 
        for k2,v2 in metrics[k].items() :
            tb_dict[k][k2] = [v2] 
    return tb_dict 

def add_to_dict(tensorboardData,metrics):
    for k,v in metrics.items() :
        for k2,v2 in metrics[k].items() :
            tensorboardData[k][k2].append(v2) 
    return tensorboardData

def get_mean_dict(tensorboardData) :
    for k,v in tensorboardData.items() :
        for k2,v2 in tensorboardData[k].items() :
            tensorboardData[k][k2] = np.nanmean(np.array(v2)) 
    return tensorboardData 

# Methods required for writing to tensorboard 
class Tensorboard() :
    def __init__(self,args_dict,global_summary) :

        self.args_dict = args_dict
        self.window_size = args_dict['SUMMARY_WINDOW']
        self.last_update = 0 
        self.tensorboardData = []
        self.global_summary = global_summary
        self.prev_metrics = None

    def reset(self):
        self.prev_metrics = None
        self.last_update = 0
        self.tensorboardData = []

    def update(self,metrics,currEpisode,run=None,wandb_run=None) :
        import tensorflow as tf

        if 'Traffic' in metrics.keys() and metrics['Traffic']['Ave_speed'] == 0 :
            if self.prev_metrics is not None and 'Traffic' in self.prev_metrics.keys():
                metrics['Traffic'] = self.prev_metrics['Traffic']
        else :
            self.prev_metrics = metrics
        
        if self.last_update ==0 :
            self.tensorboardData = make_list_dict(metrics)  
        else :
            self.tensorboardData = add_to_dict(self.tensorboardData,metrics)

        self.last_update +=1 
        if self.last_update >self.window_size :
            self.writeToTensorBoard(currEpisode)
            self.writeToNeptune(run,currEpisode) 
            self.writeToWandb(wandb_run,currEpisode)
            self.last_update = 0 
            self.tensorboardData = [] 
    
    def writeToTensorBoard(self,currEpisode):
        # each row in tensorboardData represents an episode
        # each column is a specific metric
        import tensorflow as tf
        mean_data = get_mean_dict(self.tensorboardData)
        summary = tf.Summary()
        counter = 0 
        for k,v in mean_data.items() :
            for k2,v2 in mean_data[k].items():
                counter+=1 
                summary.value.add(tag='{}/{}'.format(k,k2), simple_value=v2)

        self.global_summary.add_summary(summary, int(currEpisode -counter))
        self.global_summary.flush()
        return 

    def writeToNeptune(self,run,currEpisode) :
        import tensorflow as tf
        if run is not None : 
            mean_data = get_mean_dict(self.tensorboardData)
            for k,v in mean_data.items() :
                for k2,v2 in mean_data[k].items():
                    run['training/{}/{}'.format(k,k2)].log(value = v2, step =currEpisode )  
        return

    def writeToWandb(self,wandb_run,currEpisode) : 
        import tensorflow as tf
        if wandb_run is not None : 
            mean_data = get_mean_dict(self.tensorboardData)
            for k,v in mean_data.items() :
                for k2,v2 in mean_data[k].items():
                    key = 'training/{}/{}'.format(k,k2) ; val = v2 
                    wandb_run.log({key:val},step=currEpisode)
        return


class TensorboardPytorch():
    def __init__(self, args_dict, global_summary):
        self.args_dict = args_dict
        self.window_size = args_dict['SUMMARY_WINDOW']
        self.last_update = 0
        self.tensorboardData = []
        self.global_summary = global_summary
        self.prev_metrics = None

    def update(self, metrics, currEpisode, run=None):
        #TODO
        if metrics['Traffic']['Ave_speed'] == 0:
            if self.prev_metrics is not None:
                metrics['Traffic'] = self.prev_metrics['Traffic']
        else:
            self.prev_metrics = metrics

        if self.last_update == 0:
            self.tensorboardData = make_list_dict(metrics)
        else:
            self.tensorboardData = add_to_dict(self.tensorboardData, metrics)

        self.last_update += 1
        if self.last_update > self.window_size:
            self.writeToTensorBoard(currEpisode)
            self.writeToNeptune(run, currEpisode)
            self.last_update = 0
            self.tensorboardData = []

    def writeToTensorBoard(self, currEpisode):
        # each row in tensorboardData represents an episode
        # each column is a specific metric
        #TODO
        mean_data = get_mean_dict(self.tensorboardData)
        counter = 0
        for k, v in mean_data.items():
            for k2, v2 in mean_data[k].items():
                self.global_summary.add_scalar(tag='{}/{}'.format(k, k2), scalar_value=v2,global_step = currEpisode)

        return

    def writeToNeptune(self, run, currEpisode):
        if run is not None:
            mean_data = get_mean_dict(self.tensorboardData)
            for k, v in mean_data.items():
                for k2, v2 in mean_data[k].items():
                    run['training/{}/{}'.format(k, k2)].log(value=v2, step=currEpisode)
        return
        # Methods required for logging the stdout
class DebugLogger:
    def __init__(self, out1, out2):
        self.out1 = out1
        self.out2 = out2 
    def write(self, *args, **kwargs):
        self.out1.write(*args, **kwargs)
        self.out2.write(*args, **kwargs)
    def flush(self) :
        pass 

# Set up directories for storing training progress 
def create_logging_utils(args_dict) :
    # Create directories
    if not os.path.exists(args_dict['model_path']):
        os.makedirs(args_dict['model_path'])
    if not os.path.exists(args_dict['gifs_path']):
        os.makedirs(args_dict['gifs_path'])
    # if not os.path.exists(args_dict['trip_path']):
    #     os.makedirs(args_dict['trip_path'])
    if not os.path.exists(args_dict['TEMP_GIF_FOLDER']):
        os.makedirs(args_dict['TEMP_GIF_FOLDER'])
    write_description = open(args_dict['model_path']+  args_dict['description_filename'], 'w+')
    write_description.write(args_dict['description'])
    write_description.close()
    sys.stdout = DebugLogger(open(args_dict['model_path']+'/stdoutlog.txt','w+'), sys.stdout) 
    return 

# Maps index to intersection ID
def convert_to_nt(input_dict,args_dict,env) :
    output_dict = {} 
    for k,v in input_dict.items() :
        output_dict[index_to_ID(k,env)] = v 
    return output_dict

# Description of the training process 
def describe(text_description,policy_weight,value_weight,entropy_weight,prediction_weight,git_head,git_branch) :
    textual = text_description 
    weights = ' \nPolicy Weight: {} \nValue Weight: {} \nEntropy Weight:  {} \nPred_Loss_weight:  {}  \nCurrent git branch: {} \
    \ngit head: {} \n'.format(policy_weight,value_weight,entropy_weight,prediction_weight,git_branch,git_head)
    total = textual + weights 
    return total 

# Obtrain current git branch 
def current_git_branch() :
    out = subprocess.check_output(["git", "branch"]).decode("utf8")
    current = next(line for line in out.split("\n") if line.startswith("*"))
    branch = (current.strip("*").strip())    
    return branch 

# Creates a dictionary of dictionaries 
def get_forward_dict(total,outputs) :
        empty_dict ={} 
        for element in outputs: 
            empty_dict[element] = {} 
            for i in range(total):
                empty_dict[element][i] = 0 
        return empty_dict 

# Creates the config file to store 
def create_config(args_dict) :
    encoded_json = jsonpickle.encode(args_dict)
    if not os.path.exists('configs/train_configs/' + args_dict['EXPERIMENT_NAME']): 
        os.makedirs('configs/train_configs/' + args_dict['EXPERIMENT_NAME'])
    with open('configs/train_configs/' + args_dict['EXPERIMENT_NAME'] + "/config.json", "w") as outfile:
        json.dump(encoded_json, outfile) 
        print('Wrote Configuration')
    with open(args_dict['model_path'] + "/config.json", "w") as outfile:
        json.dump(encoded_json, outfile) 
        print('Wrote Configuration')

def generate_config(var_dict,path) :
    encoded_json = jsonpickle.encode(var_dict)
    with open(path + "/config.json", "w") as outfile:
        json.dump(encoded_json, outfile) 
        print('Wrote Configuration')
    
# Load a configuration file for training 
def load_config(config_path) :
    with open(config_path, "r") as jsonfile:
        data = json.load(jsonfile)
        print("Configuration Read Successful")
        return jsonpickle.decode(data) 

#Updates some variables when running a loaded configuration file 
def update_paths(args_dict) :
    args_dict['EXPERIMENT_NAME'] = args_dict['EXP_NAME'] +  datetime.datetime.now().strftime("%d_%m_%H")
    args_dict['model_path'] =  'data/' + args_dict['EXPERIMENT_NAME'] + '/model_traffic_{}'.format(args_dict['EXPERIMENT_NAME'])
    args_dict['train_path'] = 'data/' + args_dict['EXPERIMENT_NAME'] + '/train_traffic_{}'.format(args_dict['EXPERIMENT_NAME']) 
    args_dict['gifs_path'] = 'data/' + args_dict['EXPERIMENT_NAME'] + '/gifs_traffic_{}'.format(args_dict['EXPERIMENT_NAME'])
    return args_dict

def setup_neptune(args) : 
    global run 
    run = None 
    if int(args['neptune']) : 
        if args['load_model'] :
            if args['neptune_run'] is not None :
                project = args['neptune_project']
                token = args['NEPTUNE_API_TOKEN']
                run = neptune.init(project=project, api_token=token,run=args['neptune_run'] )
                return run 
            else :
                project = args['neptune_project']
                token = args['NEPTUNE_API_TOKEN']
                run = neptune.init(project=project, api_token=token, run=args['neptune_run'])
                raise RuntimeError('Please specify run to resume from in Neptune')  

        project = args['neptune_project']
        token = args['NEPTUNE_API_TOKEN']
        run = neptune.init(project=project, api_token=token)
        run['name'] = args['EXPERIMENT_NAME']
        run['params'] = args 
    return run  

def setup_wandb(args) :
    wandb_args = {} 
    for k,v in args.items():  
        if isinstance(v,(int,str,float,bool,list,dict,bytes)) :
            wandb_args[k] = v 
    global wandb_run 
    wandb_run = None 
    if int(args['wandb']) : 
        import wandb
        wandb_run = wandb.init(project='Traffic',config=wandb_args,name=args['EXPERIMENT_NAME']) 
    return wandb_run 

def setup_test_logging(args_dict) :
    if not os.path.exists(args_dict['trip_dir']):
        os.mkdir(args_dict['trip_dir'])
    if not os.path.exists(args_dict['routes_dir']):
        os.mkdir(args_dict['routes_dir'])
    if not os.path.exists(args_dict['screenshots_dir']):
        os.mkdir(args_dict['screenshots_dir'])
    if not os.path.exists(args_dict['base_dir']):
        os.mkdir(args_dict['base_dir'])
    if not os.path.exists(args_dict['base_dir'] + args_dict['plot_dir']):
        os.mkdir(args_dict['base_dir'] + args_dict['plot_dir'])
    if not os.path.exists(args_dict['base_dir'] + args_dict['eval_dir']):
        os.mkdir(args_dict['base_dir'] + args_dict['eval_dir'])
    if not os.path.exists(args_dict['base_dir'] + args_dict['gifs_dir']):
        os.mkdir(args_dict['base_dir'] + args_dict['gifs_dir'])


def save_as_json(file, data):
    data = json.dumps(data)
    with open(file, 'w+') as file:
        file.write(data)


def load_json(file):
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)

def check_shape_equality(input_shape) :
    item = list(input_shape.values())[0] 
    items = list(input_shape.values()) 
    for element in items :
        if element!= item : 
            return 0 
    return 1 