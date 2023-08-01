import torch
#torch.autograd.detect_anomaly(True)


torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.is_available()
print(torch.__version__)

import sys
#sys.path.remove('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T5')
sys.path.append('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9')

#print(sys.path)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import time
import utils
import models
import DeepTTE
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str , default= 'train')
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--epochs', type = int, default = 60 )

# evaluation args
parser.add_argument('--weight_file', type = str , default= 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9/saved_weights/20230127-032927.pt')
parser.add_argument('--result_file', type = str , default='C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9/results/deeptte.res')

# cnn args
parser.add_argument('--kernel_size', type = int, default=3)

# rnn args
parser.add_argument('--pooling_method', type = str, default= 'attention')

# multi-task args
parser.add_argument('--alpha', type = float , default= 0.1)

# log file name
parser.add_argument('--log_file', type = str, default= 'run_log')

args = parser.parse_args()
import json
config = json.load(open('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master/config-7.5file.json', 'r'))


for i in config['train_set']:
    #print(i)
    a= data_loader.get_loader(i, args.batch_size)
    a.dataset
    #for idx, (attr, traj) in enumerate(a):
        #print(idx)

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()
    

    if torch.cuda.is_available():
        model.cuda()
        

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    #,weight_decay=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=100)
    lambda1 = lambda epoch: 0.98 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    lrs = []
    
    epoch_loss_saved = []
    valid_loss_saved1 = []
    since = time.time()
    
    for epoch in range(args.epochs):
        
        file_loss_saved = []
        
        model.train()
        
        
        print( 'Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print( 'Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            
            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)
                _, loss = model.eval_on_batch(attr, traj, config)
                torch.cuda.synchronize()
                #with torch.autograd.detect_anomaly():
                # update the model
                    #optimizer.zero_grad()
                    #loss.backward()
                    
                #Replaces pow(2.0) with abs() for L1 regularization
     
                #l2_lambda = 0.0001
                #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                #print(l2_norm)
                #penalty= (l2_lambda/2*args.batch_size)* l2_norm
                #print(penalty, loss)
                #loss = loss + penalty
                
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2 , error_if_nonfinite= True)
                #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                #print('loss',loss.data)
                running_loss += loss.data
                
                #print('??', running_loss)
                #print ('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0))) 
                torch.cuda.synchronize()
                elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss.item() / (idx + 1.0)))
            print('average loss ',(running_loss / (idx + 1)).item())
            
            file_loss_saved.append((running_loss / (idx + 1)).item())
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        #optimizer.param_groups[0]['lr'] = lr
        print('lr-1', epoch, lrs[-1])
        epoch_loss_saved.append(np.mean(file_loss_saved))


        # evaluate the model after each epoch
        valid_loss_saved1.append(evaluate(model, elogger, eval_set, save_result = False))

        # save the weight file after each epoch
        #weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        weight_name = '{}'.format(time.strftime("%Y%m%d-%H%M%S"))

        elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9/saved_weights/{}.pt'.format(str(weight_name)))
    
    return epoch_loss_saved , valid_loss_saved1

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    model.eval()
    idx2 = 0
    valid_loss_saved = []

    if save_result:
        fs = open('%s' % args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        
        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)
            pred_dict, loss = model.eval_on_batch(attr, traj, config)
            torch.cuda.synchronize()
            #l2_lambda = 0.0001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #print(l2_norm)
            #penalty= (l2_lambda/2*args.batch_size)* l2_norm
            #print(penalty, loss)
            #loss = loss + penalty
            
            if save_result: write_result(fs, pred_dict, attr)

            running_loss += loss.data
            idx2 = idx
            #print(idx2)
            
        print ('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx2 + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx2 + 1.0)))
        
        valid_loss_saved.append((running_loss / (idx2 + 1.0)).item())
        
    if save_result: fs.close()
    return valid_loss_saved 

def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs
def run():
    # get the model arguments
    kwargs = get_kwargs(DeepTTE.Net)

    # model instance
    model = DeepTTE.Net(**kwargs)
    total_params = sum(param.numel() for param in model.parameters())
    print('param', total_params)
    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        t0 = time.time()
        torch.cuda.synchronize()
        epoch,valid = train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])
        t1=time.time() - t0
        print('{} seconds'.format(t1))
        print('hour',t1/3600)
        import matplotlib.pyplot as plt
        print(epoch , valid)
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, "-b", label="Training Loss")
        plt.plot(valid, "-r", label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        plt.legend(loc="upper right",fontsize='large')
        plt.show()

        
    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
    run()
    
