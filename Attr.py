import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    embed_dims = [('driverID', 24000, 17), ('weekID', 7,4), ('timeID', 1440,9)]

    def __init__(self):
        super(Net, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        # append total distance
        return sz + 1

    def forward(self, attr):
        #print(attr)
        em_list = []
        #print('attr' , attr)
        for name, dim_in, dim_out in Net.embed_dims:
            embed = getattr(self, name + '_em')
            #print(attr[name])
            attr_t = attr[name].view(-1, 1)
            #print('attr_t1',attr_t)
            attr_t = torch.squeeze(embed(attr_t))
            #print('attr_t2',attr_t)
            #attr_t = torch.squeeze(embed(attr[name]))
            em_list.append(attr_t)
        #print(attr_t)
        dist = utils.normalize(attr['dist'], 'dist')
        
        
        em_list.append(dist.view(-1, 1))

        #em_list.append(dist)
        
        #print ('em',em_list)
        #print('ali',torch.cat(em_list ,dim=1))
        return torch.cat(em_list , dim =1)
