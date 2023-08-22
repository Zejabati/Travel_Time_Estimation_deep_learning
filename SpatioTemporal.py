
import sys
sys.path.append('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9/')
import utils
import torch.cuda.profiler as profiler

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import GeoConv
import numpy as np

from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        print('d',dropout)
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)    
    
class Net(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''
    def __init__(self, attr_size, kernel_size = 3, num_filter = 48, pooling_method = 'attention', rnn = 'Bilstm+TCN', dropout=0.3):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.dropout=dropout

        self.geo_conv = GeoConv.Net(kernel_size = kernel_size, num_filter = num_filter)
    #num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        self.GCN= GeoConv.GCN(nfeat=16, nhid=16, nclass=49 , dropout=0.4)
        
        #self.attention = nn.MultiheadAttention(33, num_heads=3)



        self.build()

        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size = num_filter + 1 + attr_size, \
                                      hidden_size = 128, \
                                      num_layers = 2, \
                                      batch_first = True , bidirectional= False)
            self.packed_sequence = True

        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size = num_filter + 1 + attr_size, \
                              hidden_size = 128, \
                              num_layers = 1, \
                              batch_first = True)
            
        elif rnn== 'TCN':
            print('TCN')
            class TCN(nn.Module):
                def __init__(self, in_features, out_features, kernel_size):
                    super(TCN, self).__init__()
                    self.conv1 = nn.Conv1d(in_features, out_features, kernel_size)
                    self.conv2 = nn.Conv1d(in_features, out_features, kernel_size)
                    self.tanh = nn.Tanh()
                    self.sigmoid = nn.Sigmoid()        
                def forward(self, x):
                    conv1_out = self.conv1(x)
                    conv2_out = self.conv2(x)
                    tanh_out = self.tanh(conv1_out)
                    sigmoid_out = self.sigmoid(conv2_out)
                    output = tanh_out * sigmoid_out
                    return output
            self.rnn=TCN(64,64,3)
            self.packed_sequence=False
            self.MVCNN= False
            
                
        elif rnn =='TransTCN':
            print('TransTCN')
            class TransTCN(nn.Module):
                def __init__(self, in_features, out_features, nhead, num_layers):
                    super(TransTCN, self).__init__()
                    self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4,batch_first=True,norm_first=True)
                    self.enc_norm = nn.LayerNorm(64)
                    self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers,norm=self.enc_norm)

                    #self.transformer = nn.Transformer(in_features, nhead, num_layers)
                    self.fc = nn.Linear(in_features, out_features)
    
                def forward(self, x):

                    output = self.transformer(x)
                    output = self.fc(output)
                    return output

            self.rnn = TransTCN(64, 64, 8, 2)
            self.packed_sequence=False
            self.MVCNN= False   
            
            
            
        elif rnn =='Bilstm+TCN':
            #print('Bilstm+TCN')
            class TCN(nn.Module):
                def __init__(self, in_features, out_features, kernel_size,padding):

                    super(TCN, self).__init__()
                    if padding == 1:
                        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size,padding= 'same')
                        self.conv2 = nn.Conv1d(in_features, out_features, kernel_size,padding= 'same')
                    else:
                        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size)
                        self.conv2 = nn.Conv1d(in_features, out_features, kernel_size)
                    self.tanh = nn.Tanh()
                    self.sigmoid = nn.Sigmoid()        
                def forward(self, x):
                    conv1_out = self.conv1(x)
                    conv2_out = self.conv2(x)
                    tanh_out = self.tanh(conv1_out)
                    sigmoid_out = self.sigmoid(conv2_out)
                    output = tanh_out * sigmoid_out
                    return output
                
            self.rnn0=TCN(96,96,3,1)
            self.rnn1=TCN(96,96,3,1)
            
            
            self.BiLSTM = nn.LSTM(input_size = 96, \
                                      hidden_size = 128, \
                                      num_layers = 2, \
                                      batch_first = True , bidirectional= True)

            self.packed_sequence0=False
            self.MVCNN= False
            self.packed_sequence1 = True
            
            
        if pooling_method == 'attention':
            self.dropoutattent = nn.Dropout(p=dropout)
            self.attr2atten = nn.Linear(attr_size, 352)    


    def out_size(self):
        # return the output size of spatio-temporal component
        return 352

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)

        if torch.cuda.is_available():
            lens = torch.cuda.FloatTensor(lens)
        else:
            lens = lens.type(torch.FloatTensor)
            #lens = torch.FloatTensor(lens)
        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False)
        hiddens = hiddens / lens

        return hiddens


    def attent_pooling(self, hiddens, lens, attr_t):
        attent = F.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

        attent= self.dropoutattent(attent)

    #hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
    
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)
        return hiddens
    
    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        #self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)  

    def forward(self, traj, attr_t, config):
        
        conv_locs = self.geo_conv(traj, config)
        
        lngs = torch.unsqueeze(traj['lngs'], dim = 2)
        lats = torch.unsqueeze(traj['lats'], dim = 2)

        states = self.state_em(traj['states'].long())

        locs = torch.cat((lngs, lats, states), dim = 2)
        l=locs.size()[1]
        locs=locs[:,0:l-2,:]
        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords(locs))
        #locs = locs.permute(0, 2, 1)
        #print('locs',locs.size())
         
        #locs = locs.permute(0, 2, 1)
        GCN=False
        if GCN==True:
            num_nodes = locs.size()[1]
            edge_indices = []
            # Iterate over the nodes in the graph
            for i in range(num_nodes):
                if i > 0:
                    edge_indices.append([i, i-1])
                if i < num_nodes-1:
                    edge_indices.append([i, i+1])


            # Convert the list of edge indices to a tensor
            edge_indices = torch.tensor(edge_indices, dtype=torch.long)
            edge_indices = edge_indices.to("cuda:0")

            edge_indices = edge_indices.permute(1, 0)
        
            GCN_locs=torch.cat((self.GCN(locs,edge_indices),locs),dim = 2)
            conv_locs=GCN_locs 

        #GCN_locs=self.GCN(locs,edge_indices)
        #print('locs',locs.size())
        #print('conv', conv_locs.size())

        conv_locs=torch.cat((conv_locs,locs),dim = 2)

        
        #print('conv', conv_locs.size())
        #local_dist = utils.get_local_seq(traj['dist_gap'], 1, config['dist_gap_mean'], config['dist_gap_std'])
        #local_dist = torch.unsqueeze(local_dist, dim = 2)
        #conv_locs = torch.cat((conv_locs, local_dist), dim = 2)
        
        attr_t = torch.unsqueeze(attr_t, dim = 1)
        #print(attr_t.size())

        if self.MVCNN==True:
                #attr_T=self.attr2atten(attr_t)
                expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))
                conv_locs = conv_locs.permute(0, 2, 1)
                conv_locs = F.tanh(self.norm(self.conv0(conv_locs)))
                #print('conv',conv_locs.size())
                
                expand_attr_t = expand_attr_t.permute(0, 2, 1)
                expand_attr_t = F.tanh(self.norm(self.conv0(expand_attr_t)))
                #print('expand_attr_t',expand_attr_t.size())
                conv_locs = torch.cat((conv_locs, expand_attr_t), dim = 1)
                conv_locs = F.elu(self.norm(self.conv4(conv_locs)))
                #print('cat',conv_locs.size())
                #conv_locs=self.conv264(conv_locs)

        else:
            #print('ex', conv_locs.size()[:2] + (attr_t.size()[-1], ))

            expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))
            #print('exat',expand_attr_t.size() )

            # concat the loc_conv and the attributes
            conv_locs = torch.cat((conv_locs, expand_attr_t), dim = 2)
            #conv_locs=conv_locs[:,0:135,:]
            #print('conv1', conv_locs.size())

            #print('conv_loc',conv_locs.size())
        
        lens = list(map(lambda x: x-self.kernel_size + 1 , traj['lens']))
        #lens1 = []
        #for x in lens:
         #   lens1.append(x - 2)
        #print('0',lens)
        #print('1',lens1)

        if self.packed_sequence0 == False:
            if self.MVCNN==True:
                ## of MVCNN block
                k=4
                X=[]
                conv_locs = conv_locs.permute(0, 2, 1)
                expand_attr_t = expand_attr_t.permute(0, 2, 1)

                X.append(conv_locs)
                for i in range(k):
                    x_pos=self.pos_encoder(X[i])
                    x_pos=self.enc_norm(x_pos)

                    #expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))
                    x_pos = x_pos.permute(0, 2, 1)
                    locs_view1 = F.tanh(self.norm(self.conv1(x_pos)))
                    locs_view2 = F.tanh(self.norm(self.conv2(x_pos)))
                    locs_view3 = F.tanh(self.norm(self.conv3(x_pos)))
                    #print(locs_view1.size())
                
                    view1=F.softmax(torch.matmul(locs_view1,expand_attr_t))
                    view2=F.softmax(torch.matmul(locs_view2,expand_attr_t))
                    view3=F.softmax(torch.matmul(locs_view3,expand_attr_t))
    
                    #x_view_W=nn.Linear(3,1).to('cuda')
                    #sum_=torch.cat((torch.matmul(view1,x_pos),torch.matmul(view2,x_pos),torch.matmul(view3,x_pos)),dim=2)
                    #sum_=sum_.view(x_pos.size(0),x_pos.size(1),x_pos.size(2),3)
                    #x_view=F.elu(torch.squeeze(x_view_W(sum_),3)).to('cuda')

                    sum_= torch.matmul(view1,x_pos)+torch.matmul(view2,x_pos)+torch.matmul(view3,x_pos)
                    #print(sum_.size())
                    x_view_W=nn.Linear(sum_.size(2),sum_.size(2)).to('cuda')
                    x_view=F.elu(x_view_W(sum_)).to('cuda')
                    #print(x_view.size())
                    
                    
                    x_view = x_view.permute(0, 2, 1)
                    x_view=self.enc_norm(x_view)

                    x_pos = x_pos.permute(0, 2, 1)
                    
                    out_s=self.rnn(x_pos)+x_view
                    #out_s=self.enc_norm(out_s)

                    out_t=self.rnn(out_s)
                    #out_t=self.enc_norm(out_t)
                    X.append(out_t+x_pos)
                 
                #print(len(X))
                #print(X[-1].size())
                hiddens= X[-1]
                packed_hiddens = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first = True)

            #only transformer_without CNN
            else:
                #x_pos=self.pos_encoder(conv_locs)
                #packed_inputs = x_pos
                #print('conv_loc1',conv_locs.size())

                #conv_locs= self.gat(conv_locs, edge_indices)
                #print('conv2', conv_locs.size())
                
                #permute for TCN
                conv_locs = conv_locs.permute(0, 2, 1)
                
                #packed_inputs = conv_locs
                #hiddens = self.rnn(packed_inputs)
                
                packed_inputs0 = conv_locs
                hiddens0 = self.rnn0(packed_inputs0)
                self.dropout = nn.Dropout(0.3)
                hiddens0 = self.dropout(hiddens0)
                hiddens_TCN = self.rnn1(hiddens0)
                
                #permute for TCN
                hiddens_TCN = hiddens_TCN.permute(0, 2, 1)
                #print('hidden_T',hiddens_TCN.size())
                
                packed_hiddens_TCN = nn.utils.rnn.pack_padded_sequence(hiddens_TCN, lens, batch_first = True)
                #print('pack_T',packed_hiddens_TCN.size() )
            
        #else:
        if self.packed_sequence1 == True:
            #TCN permute make up
            conv_locs = conv_locs.permute(0, 2, 1)
            #print('conv2', conv_locs.size())

            packed_inputs1 = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first = True)
            packed_hiddens_Bi, (h_n, c_n) = self.BiLSTM(packed_inputs1)
            
            hiddens_Bi, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens_Bi, batch_first = True)
            
            #print('hidden_B',hiddens_Bi.size())
            #print('len',lens)
            #print('pack_B',packed_hiddens_Bi.size() )
            
        hiddens= torch.cat((hiddens_TCN, hiddens_Bi), dim = 2)
        #hiddens=hiddens[:,0:135,:]

        #print('hhhhh', hiddens.size())
        packed_hiddens = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first = True)

        
        #print(hiddens)
        
        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

        elif self.pooling_method == 'attention':
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)


