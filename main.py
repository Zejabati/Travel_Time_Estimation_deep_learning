import torch

import sys
#sys.path.remove('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T5')
sys.path.append('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/TTE final model')

#print(sys.path)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import utils
import DeepTTE
import logger
import inspect
import argparse
import data_loader



parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str , default= 'sample_output')
parser.add_argument('--batch_size', type = int, default = 2)
parser.add_argument('--epochs', type = int, default = 60 )

# evaluation args

parser.add_argument('--weight_file', type = str , default= 'C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/TTE final model/20230127-051216.pt')
parser.add_argument('--result_file', type = str , default='C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/TTE final model/results/deeptte.res')

# cnn args
parser.add_argument('--kernel_size', type = int, default=3)

# rnn args
parser.add_argument('--pooling_method', type = str, default= 'attention')

# multi-task args
parser.add_argument('--alpha', type = float , default= 0.1)

# log file name
parser.add_argument('--log_file', type = str, default= 'run_log')

args = parser.parse_args()
#config = json.load(open('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master/config-7.5file.json', 'r'))
config = json.load(open('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/TTE final model/config.json', 'r'))




def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs
def run():
    # import random
    # import numpy as np
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    #a= data_loader.get_loader(args.batch_size, dict1)
    #a.dataset
    
    # get the model arguments
    kwargs = get_kwargs(DeepTTE.Net)

    # model instance
    model = DeepTTE.Net(**kwargs)

    elogger = logger.Logger(args.log_file)

    if args.task == 'sample_output':

        state_dict = torch.load(args.weight_file,map_location = torch.device('cpu'))
        state_dict.pop("spatio_temporal.GCN.gc1.weight", None)
        state_dict.pop("spatio_temporal.GCN.gc2.weight", None)
        model.load_state_dict(state_dict, strict=False)

        #TTE= evaluate1(model, elogger, dict1=dict1)
        #print('TTE',TTE)
        #return TTE
        return model,elogger

# if __name__ == '__main__':
#    #TTE = run()
#    model,elogger=run()
