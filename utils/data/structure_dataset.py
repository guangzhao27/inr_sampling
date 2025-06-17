import os
import math
import h5py
import torch
import random
import scipy.io
import numpy as np
import xarray as xr
from itertools import product
from einops import rearrange
from argparse import ArgumentParser



from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

class NavierStokes(Dataset):
    def __init__(
        self, datapath, nx, sub=1, T=20, t_interval=1, n_train=None, n_test=None, missing_rate = 0.75, missing_same = True, train_mask = None):
        self.S = nx // sub
        self.T = T
        self.sub = sub
        self.t_interval = t_interval
        self.n_train = n_train
        self.n_test = n_test
        
        if n_test and missing_same:
            assert train_mask is not None, 'Provide a train mask if missing positions are the same between train, test'
        
        # Missing rate designation
        """
        There are two variants of missing sensors. 
        First, [v1] missing sensors are the same between training and testing
        Second, [v2] missing sensors of training set are different from testing
        Argument: missing_same controls that. [v1] missing_same = True, [v2] missing_same = False
        """
        self.missing_rate = missing_rate
        self.remaining_rate = 1 - missing_rate
        
        
        # Dataloading
        try:
            data = h5py.File(datapath)['u']
        except:
            data = scipy.io.loadmat(datapath)['u']
            data = np.array(data).transpose(3, 1, 2, 0)
        print('Loaded data: {}'.format(data.shape))
        
        if n_train:
            self.a = torch.tensor(data[0 : 1, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            self.u = torch.tensor(data[1 : self.T + 1, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            
            ## Addressing missing rate            
            self.train_support_mask = self.get_mask()
            
            
        if n_test:
            self.a = torch.tensor(data[0 : 1, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)      # channel dimension = 1
            self.u = torch.tensor(data[1 : self.T + 1, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3) # channel dimension = 1
            if missing_same:
                self.test_support_mask = train_mask
            else:
                self.test_support_mask = self.get_mask()
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError
            
        self.get_mesh()
            
    def get_mask(self):
        H, W = self.a.shape[1:3]
        n_support = int(H * W * self.remaining_rate)
        support_mask = torch.zeros(H, W) # get spatial dimension
        loc = list(product(list(range(H)), list(range(W)))) # get all possible combination of H and W
        random.shuffle(loc)
        for h, w in loc[:n_support]:
            support_mask[h,w] = 1
        return support_mask
    
    
    def get_mesh(self):
        # Please use this mesh if need be.
        # geometry locations (x, y)
        mesh1 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh2 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh1 = mesh1.reshape(self.S, 1, 1).repeat([1, self.S, 1])
        mesh2 = mesh2.reshape(1, self.S, 1).repeat([self.S, 1, 1])
        self.mesh = torch.cat((mesh1, mesh2), dim=-1) # (S x S, 2) 
        
    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx].unsqueeze(-2), self.u[idx].unsqueeze(-2)

        # return xout, yout
        
