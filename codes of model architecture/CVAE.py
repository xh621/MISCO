# This file defines the VAE model architecture in MISCO

import torch
from torch import nn
from custom_mlp import MLP,Exp
import pyro
import pyro.distributions as dist
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam
import numpy as np

class VAE(nn.Module):
    
    def __init__(self, y_size = 5, x_size = 5,  # y_size: number of tasks; x_size, size of the morphology
                 hidden_layers_g = [500,], hidden_layers_p = [500,], # number of hidden units and layers in MLPs
                 y_emb = 128, h_emb = 128, hij_emb = 64, # dimension of task embedding, latents and voxel-specific latents
                 heads = 1, att_dim = 64, # number of heads and hidden dimension in attention
                 config_enum = None, use_cuda = False):
        
        super().__init__()
        
        self.y_size = y_size
        self.x_size = x_size
       
        self.hidden_sizes_g = hidden_layers_g
        self.hidden_sizes_p = hidden_layers_p
        
        self.allow_broadcast = config_enum=="parallel"
        self.use_cuda = use_cuda
        
        self.y_emb = y_emb
        self.h_emb = h_emb
        self.hij_emb = hij_emb
        self.x_emb = int(self.h_emb/2)
        
        self.heads = heads
        self.att_dim = att_dim
        
        self.operator_x = torch.zeros(5*self.x_size**2, 5*self.x_size**2)
        for i in range(self.x_size**2):
            self.operator_x[i*5:(i+1)*5,i*5:(i+1)*5] = 1
        
        self.setup_networks()
        
        # one-hot encoded coordinates
        self.W = torch.zeros(self.x_size**2, 10, requires_grad=False)
        for row in range(5): # row index
            for column in range(5): # column index
                ci = [0]*5
                ci[row] = 1
                cj = [0]*5
                cj[column] = 1
                self.W[row*5+column,:] = torch.tensor(ci + cj, requires_grad = False)
        
        # mask used in attention (and neighbors used in NCA)
        self.mask = torch.ones(self.x_size**2, self.x_size**2, requires_grad = False)
        
        self.up = torch.zeros(self.x_size**2, self.x_size**2, requires_grad = False)
        self.down = torch.zeros(self.x_size**2, self.x_size**2, requires_grad = False)
        self.left = torch.zeros(self.x_size**2, self.x_size**2, requires_grad = False)
        self.right = torch.zeros(self.x_size**2, self.x_size**2, requires_grad = False)
        
        for row in range(self.x_size):
            for col in range(self.x_size):
                neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                for flag, (neighbor_row, neighbor_col) in enumerate(neighbors):
                    if neighbor_row>-1 and neighbor_row<5 and neighbor_col>-1 and neighbor_col<5:
                        self.mask[row*self.x_size+col, neighbor_row*self.x_size+neighbor_col] = 0
                        if flag==0:
                            self.up[row*self.x_size+col, neighbor_row*self.x_size+neighbor_col] = 1
                        if flag==1:
                            self.down[row*self.x_size+col, neighbor_row*self.x_size+neighbor_col] = 1
                        if flag==2:
                            self.left[row*self.x_size+col, neighbor_row*self.x_size+neighbor_col] = 1
                        if flag==3:
                            self.right[row*self.x_size+col, neighbor_row*self.x_size+neighbor_col] = 1
                        
        self.mask = self.mask==1
        
    # construct encoders and decoders
    def setup_networks(self):
        
        ###################################################
        # Approximate posterior
        ###################################################
        
        # posterior: voxels & latents -> task (not used, since task type is given during inference)
        self.encoder_y = MLP([5*self.x_size**2+self.h_emb]+self.hidden_sizes_p+[self.y_size],
                            activation=nn.ReLU,
                            output_activation=None,
                            allow_broadcast=self.allow_broadcast,
                            use_cuda=self.use_cuda)
        
        # posterior: voxels+task -> latents
        self.encoder_h = MLP([5*self.x_size**2+self.y_emb]+self.hidden_sizes_p+[[self.h_emb, self.h_emb]],
                            activation=nn.ReLU,
                            output_activation=[None, Exp],
                            allow_broadcast=self.allow_broadcast,
                            use_cuda=self.use_cuda)
        
        ###################################################
        # Generative process
        ###################################################
        
        # embedding layer for tasks
        self.emb_y = nn.Parameter(torch.rand(self.y_size, self.y_emb))
        
        # embedding of voxel location
        self.emb_x = nn.Parameter(torch.rand(5, self.x_emb))
        
        # generator: task embedding -> latents
        self.decoder_h = MLP([self.y_emb]+self.hidden_sizes_g+[[self.h_emb, self.h_emb]],
                            activation=nn.ReLU, output_activation=[None,Exp],
                            allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        
        # generator: latents + coordinates -> voxel-specific latents
        self.decoder_hij = MLP([self.h_emb*2]+self.hidden_sizes_g+[self.hij_emb],
                              activation=nn.ReLU, output_activation = None, 
                              allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        
        
        # generator: NCA among voxel-specific latents
        self.nca = MLP([5*self.hij_emb]+[64]+[self.hij_emb], activation=nn.ReLU, output_activation=None,
                       allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        
        # generator: attented voxel-specific latents -> voxel 
        self.decoder_xij = MLP([self.hij_emb]+self.hidden_sizes_g+[5],
                           activation=nn.ReLU, output_activation=None,
                           allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        
        # enable GPU for faster training
        if self.use_cuda:
            self.cuda()
            
    # define the generative process
    @config_enumerate(default="parallel")
    def model(self, xs, ys=None, hs=None):
        
        # xs and ys are one-hot encoded morphology and task, respectively, 
        #  with shapes (batch_size, self.x_size**2*5) and (batch_size, self.y_size)
        
        pyro.module("ss_vae", self)
        
        if xs == None: # proposing new robot designs
            batch_size = ys.size(0)
        else: # during inference
            batch_size = xs.size(0)
            
        with pyro.plate("data"):
            
            # task
            alpha_prior = torch.ones(batch_size, self.y_size)/(1.0*self.y_size) # uniform prior, unused
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior).to_event(1), obs=ys)
            ys_emb = torch.matmul(ys.float(), self.emb_y)
            # ys_emb = self.emb_y.forward(ys)
            
            # latents
            loc, scale = self.decoder_h.forward(ys_emb)
            hs_emb = pyro.sample("h", dist.Normal(loc, scale).to_event(1))
            
            # voxel-specific latents
            temp_1 = hs_emb.unsqueeze(1).repeat(1, self.x_size**2, 1)
            #temp_2 = self.W.unsqueeze(0).repeat(batch_size, 1, 1)
            temp_2 = torch.matmul(self.W[:,0:5], self.emb_x).unsqueeze(0).repeat(batch_size,1,1)
            temp_3 = torch.matmul(self.W[:,5:10], self.emb_x).unsqueeze(0).repeat(batch_size,1,1)
            hijs_emb = self.decoder_hij.forward(torch.cat((temp_1, temp_2, temp_3), dim=2))
            
            # print(hijs_emb)

            # NCA
            for step in range(30): 
                temp_up = torch.bmm(self.up.unsqueeze(0).repeat(batch_size,1,1), hijs_emb)
                temp_down = torch.bmm(self.down.unsqueeze(0).repeat(batch_size,1,1), hijs_emb)
                temp_left = torch.bmm(self.left.unsqueeze(0).repeat(batch_size,1,1), hijs_emb)
                temp_right = torch.bmm(self.right.unsqueeze(0).repeat(batch_size,1,1), hijs_emb)
                temps = torch.cat((temp_up, temp_down, temp_left, temp_right, hijs_emb), dim=2)
                hijs_emb = self.nca.forward(temps) + hijs_emb
                
            
            # print(hijs_emb)
            
            # voxels
            xs_probs = self.decoder_xij(hijs_emb) # logits of shape: (batch_size, self.x_size**2, 5)
            if xs != None:
                xs = pyro.sample("x", dist.OneHotCategorical(logits=xs_probs).to_event(2), obs=xs.reshape(-1,25,5))
            else:
                xs = pyro.sample("x", dist.OneHotCategorical(logits=xs_probs).to_event(2), obs=None)
        
        return (xs.reshape(-1,self.x_size**2*5), ys, hs_emb, xs_probs)  
        
    # define the approximate posteriors
    @config_enumerate(default="parallel")
    def guide(self, xs, ys=None, hs=None):
        
        with pyro.plate("data"):
            
            # inference of h (x->h)
            if hs is None:
                batch_size = xs.size(0)
                # task
                alpha_prior = torch.ones(batch_size, self.y_size)/(1.0*self.y_size) # uniform prior, unused
                ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior).to_event(1), obs=ys)
                ys_emb = torch.matmul(ys.float(), self.emb_y)
                xs_task = torch.cat((xs, ys_emb), dim=1)#batch×(5*x_size*x_size) concat batch×(y_emb) 
                loc, scale = self.encoder_h.forward(xs_task)
                hs = pyro.sample("h", dist.Normal(loc, scale).to_event(1))
            
            # inference of y (x+h->y), unused
            if ys is None:
                alpha = self.encoder_y.forward([xs,hs])
                ys = pyro.sample("y", dist.OneHotCategorical(logits=alpha).to_event(1)) 
                
        return(xs.reshape(-1,self.x_size**2*5), ys, hs)
