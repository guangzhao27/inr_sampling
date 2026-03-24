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
from functools import partial
# import apebench

# from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from mmap_ninja import RaggedMmap
from time import time
from collections import defaultdict

# def creat_dataset(train_num, val_num, test_num, train_p_list, val_p_list, test_p_list)

def normalize_fn(x, mean, std):
    assert x.size(0) == mean.size(0)
    # mean = mean.reshape(x.shape)
    return (x - mean) / std

def inv_normalize_fn(x, mean, std):
    # mean = mean.reshape(x.shape)
    # assert x.size(0) == mean.size(0)
    x = x.reshape(-1, *mean.shape)
    # print(x.shape)
    # print(mean.shape)
    mean = mean.to(x.device)
    std = std.to(x.device)
    result = x * std + mean
    result = result.reshape(-1, mean.size(-1))
    # print('result:', result.shape)
    return  result

def my_generator(directory):
    """
    Generator function to yield numpy arrays from .npy files in a directory.

    Args:
        directory (str): Path to the directory containing .npy files.

    Yields:
        ndarray: The contents of each .npy file.
    """
    # List and sort the .npy files
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        print(f"Loading: {file_path}")  # Optional logging
        data = np.load(file_path)
        yield data

class TemporalDatasetWithCode(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim=64, dataset_name=None, data_to_encode=None):
        """
        Args:
            dataset
            v (torch.Tensor): Dataset values, with shape (N Dx Dy C T). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, Dy the size
            of the second spatial dimension, C the number of channels (ususally 1), and T the
            number of timestamps.
            grid (torch.Tensor): Coordinates, with shape (N Dx Dy 2). We suppose that we have
            same grid over time.
            latent_dim (int, optional): Latent dimension of the code. Defaults to 64.
        """
        N = v.shape[0]
        T = v.shape[-1]
        self.v = v
        self.c = grid  # repeat_coordinates(grid, N).clone()
        self.output_dim = self.v.shape[-2]
        self.input_dim = self.c.shape[-2]
        self.z = torch.zeros((N, latent_dim, T))
        self.latent_dim = latent_dim
        self.T = T
        self.dataset_name = dataset_name
        self.set_data_to_encode(data_to_encode)


    def set_data_to_encode(self, data_to_encode):
        self.data_to_encode = data_to_encode
        dataset_name = self.dataset_name
        N = self.v.shape[0]
        T = self.v.shape[-1]

        self.index_value = None
        if (data_to_encode is not None) and (dataset_name is not None):
            self.index_value = KEY_TO_INDEX[dataset_name][data_to_encode]
            self.z = torch.zeros((N, self.latent_dim, T))
            self.output_dim = 1

        if data_to_encode is None:
            c = len(KEY_TO_INDEX[dataset_name].keys())
            # one code for the height / vorticity
            # if c == 1, we squeeze it
            self.z = torch.zeros((N, self.latent_dim, c, T)).squeeze(-2)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.index_value is not None:
            sample_v = self.v[idx, ..., self.index_value, :]
        else:
            sample_v = self.v[idx, ...]

        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        """How to save efficiently the updated codes.

        Args:
            z_values (torch.Tensor): the updated latent code for the whole trajectory.
            idx (int): idx of the trajectory.
        """
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class GraphSomaDataset(Dataset):
    '''
    data path: "/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5"
    hdf5_file.keys(): foward_0 ... foward_99
    data shape: torch.Size([60, 100, 100, 30, 17])
    '''
    def __init__(self, data_path,
                data_num = 10, 
                train_num=20, 
                feature_set = None, 
                space_factor=1,
                time_factor=1,
                initial_step=10, 
                test_ratio=0.1,
                latent_dim=128, 
                split='train', 
                p_transform=None, 
                feature_transform=None,
                data_save_dir=None, 
                mmap_dir=None,
                sub_array_num=1, 
                missing_rate=0.0, 
                ):
        self.name = "SOMA"
        self.data_path =data_path # 
        
        self.inital_step = initial_step
        self.space_factor = space_factor
        self.time_factor = time_factor
        self.feature_set = feature_set
        self.latent_dim = latent_dim
        self.missing_rate = missing_rate
        
        
        self.hdf5_file = h5py.File(self.data_path, 'r')
        self.keys = list(self.hdf5_file.keys()) # keys are 100 forward # data shape is (30, 100)
        self.p_transform = p_transform
        self.feature_transform=feature_transform
        
        if split == 'train':
            self.idx_list = list(range(train_num))
            self.keys = self.keys[:train_num]
        elif split == 'val':
            self.idx_list = list(range(train_num, train_num+data_num))
            self.keys = self.keys[train_num:train_num+data_num]
        else:
            self.idx_list = list(range(-data_num, 0))
            self.keys = self.keys[-data_num:]
            
        if data_save_dir:
            self.data_save_dir = data_save_dir
        else:
            self.data_save_dir = os.path.join(
                '/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_graph_save', 
                f's{self.space_factor}t{self.time_factor}')
            os.makedirs(self.data_save_dir, exist_ok=True)
        
        self.mmap_dir = mmap_dir #/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save
        if self.mmap_dir:
            # RaggedMmap.from_generator(out_dir=self.mmap_dir, 
            #                            sample_generator=my_generator(raw_np_dir), 
            #                            batch_size=2)
            self.mmap_data = RaggedMmap(self.mmap_dir)
            
        self.sub_array_num = sub_array_num
        
            
        # this function generate self.T in its function
        self.cal_cor_and_time_emb() # generate more self properties
        
        if self.sub_array_num > 1:
            base_size = self.T // self.sub_array_num
            extra = self.T % self.sub_array_num
            
            # Array to record the length of each subarray
            self.sub_array_T = [base_size + 1 if i < extra else base_size for i in range(self.sub_array_num)]
        
        # self.dataset = [
        #     self.reduce_resolution(torch.from_numpy(self.hdf5_file[key][:]).permute(1, 2, 3, 0, 4))
        #     for key in self.keys
        # ]
        
        # self.latent_vectors = [
        #     torch.zeros(data.size(3), self.latent_dim)
        #     for key in self.keys
        # ]
        
        self.latent_vectors = []
        for key in self.keys:
            # xx = self.hdf5_file[key][:]
            # reduced_T = len(xx[::time_factor])
            # data = self.reduce_resolution(torch.from_numpy(self.hdf5_file[key][:]).permute(1, 2, 3, 0, 4))
            if self.sub_array_num <= 1:
                self.latent_vectors.append(torch.zeros(self.T, self.latent_dim))
            else:
                for tt in self.sub_array_T:
                    self.latent_vectors.append(torch.zeros(tt, self.latent_dim))
        
    def reduce_resolution(self, _data):
        """
        Apply reduced resolution both in spatial and temporal dimensions.
        """
        # Reduce spatial and temporal dimensions according to reduced_resolution and reduced_resolution_t
        _data = _data[
            ::self.space_factor, 
            ::self.space_factor, 
            ::self.space_factor, 
            ::self.time_factor
        ]
        
        assert len(self.feature_set) == 1
        _data = _data[..., self.feature_set]
        
        # Apply feature set reduction if provided
        # if self.feature_set is not None:
        #     _data = _data[..., self.feature_set]
        
        return _data
    
    def cal_cor_and_time_emb(self):
        # _data = self.hdf5_file[self.keys[0]][:] # just take first data to get data shape information
        _data = self.load_raw_data(0)
        # _data = torch.from_numpy(_data)
        # data = _data[..., :-1]
        sx, sy, sz = _data.shape[0:3]
        # total_T = _data.size(3)
        
        _data = self.reduce_resolution(_data)

        # _data = _data[::self.space_factor, 
        #               ::self.space_factor, 
        #               ::self.space_factor, 
        #               ::self.time_factor
        #               ]

        
        # if self.feature_set is not None:
        #     _data = _data[..., self.feature_set]

        self.mask = _data[..., 0, 0]> -1000
        # self.mu, self.sigma = self.gen_normalize_value(_data) # self.mu and self.sigma is the average over the first single data with index=0, you should not do that!, that is hard to update
        
        
        # generate graph coordinates
        T = _data.size(3)
        self.T = T
        cor = self.mask.nonzero()
        spacial_emb = self.S_embedding(sx, sy, sz, cor)
        time_list = [torch.ones(len(cor), dtype=torch.int)*t for t in range(T)]
        
        self.cor_t = cor.repeat(T, 1)
        self.spacial_emb_t = spacial_emb.repeat(T, 1)
        self.time_t = torch.cat(time_list, dim=0)
        
        # # feature should leave in get function
        # feat_t = _data[self.cor_t[:, 0], self.cor_t[:, 1], self.cor_t[:, 2], self.time_t]

    def S_embedding(self, sx, sy, sz, cor):
        x = torch.arange(sx, dtype=torch.float32)
        y = torch.arange(sy, dtype=torch.float32)
        z = torch.arange(sz, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(x, y, z)
        
        lin_emb = lambda x, y: 2.0*x/(y-1) -1.0
        X = lin_emb(X, sx)
        Y = lin_emb(Y, sy)
        Z = lin_emb(Z, sz)
        
        self.grid = torch.stack((X, Y, Z), dim=-1)
        self.grid = self.grid[::self.space_factor, 
                              ::self.space_factor, 
                              ::self.space_factor, 
                              ]
        
        spatial_embedding = self.grid[cor[:, 0], cor[:, 1], cor[:, 2]]
        
        return spatial_embedding

    def gen_normalize_value(self, data):
        # three dimensional data
        mask_shape = self.mask.shape
        # num_new_dims = len(data.shape) - len(mask_shape) - 1
        expanded_mask = self.mask.view(*mask_shape, 1)
        expanded_mask = expanded_mask.expand(*mask_shape, *data.shape[3:-1])

        mu = torch.zeros(data.shape[-1])
        std = torch.zeros(data.shape[-1])

        for i in range(data.shape[-1]):
            non_zero_data = data[..., i][expanded_mask]
            mu[i] = non_zero_data.mean()
            std[i] = non_zero_data.std()
        
        return mu, std
    
    def create_normalize_from_dataset(self):
        assert self.feature_transform is None, 'feature transform should be none to create new normlaization'
        
        feat_list = []
        p_list = []
        for i in range(len(self.keys)):
            graph = self.getitem(i)
            T = graph.time.max().item()+1
            # p_list.append(graph.ped_para)
            for t in range(T):
                tidx = (graph.time==t)
                feat_list.append(graph.feat[tidx])
            p_list.append(graph.pde_parameter.item())
        feat_tensor = torch.stack(feat_list)
        p_tensor = torch.tensor(p_list)
        
        feat_mean = feat_tensor.mean(dim=0)
        feat_mean = torch.cat([feat_mean]*T)
        feat_std = (feat_tensor - feat_tensor.mean(dim=0)).std()
        
        
        feat_transform = partial(normalize_fn, mean=feat_mean, std=feat_std)
        inv_feat_transform = partial(inv_normalize_fn, mean=feat_mean, std=feat_std)
        
        return feat_transform, inv_feat_transform
        
            
    def update_feat_transform(self, feature_transform):
        self.feature_transform = feature_transform
        
    def update_p_transform(self, p_transform):
        self.p_transform = p_transform

    # def normalize_data(self, data):
    #     # three dimensional data
    #     mask_shape = self.mask.shape
    #     num_new_dims = len(data.shape) - len(mask_shape) - 1
    #     expanded_mask = self.mask.view(*mask_shape, *[1]*num_new_dims)
    #     expanded_mask = expanded_mask.expand(*mask_shape, *data.shape[3:-1])

    #     for i in range(data.shape[-1]):
    #         non_zero_data = data[..., i][expanded_mask]
    #         data[..., i][expanded_mask] = (non_zero_data - self.mu[i])/self.sigma[i]
            
    def __len__(self):
        return len(self.keys)*self.sub_array_num
    
    def load_raw_data(self, idx):
        if self.mmap_dir:
            rawidx = self.idx_list[idx]
            _data = torch.from_numpy(self.mmap_data[rawidx])
        else:        
            key = self.keys[idx]
            _data = torch.from_numpy(self.hdf5_file[key][:])
        
        _data = _data.permute(1, 2, 3, 0, 4)
        return _data
        
    def getitem(self, idx):
        # Y (trajectory) data dimension should be batch*sx*sy*sz*time*D
        """
        graph:
            cor: the cor of each independent point (x, y, z)
            time: time sequence lable for each independent point [0, 0, 0, 1, 1, 2, ...]
                For a batch of data points, the time label will stack together, the next time sequence start with T as [T, T, T, T+1, T+1, ...]
            feat: feature value of each point
            T: total time of each data points (time sequence)
            latent_vector: size of T*hidden_D for each data points
            pde_parameter: size of 1*p_D for each data points
            spacial_emb_t: the cor embedding of each independent point (xe, ye, ze)
        """

        _data = self.load_raw_data(idx)
        pde_parameter = _data[0, 0, 0,0,  -1:]
        
        
        # _data = _data[..., :-1]
        

        if self.p_transform:
            pde_parameter = self.p_transform(pde_parameter)
        
        #reduce datasize
        _data = self.reduce_resolution(_data)
        # _data = _data[
        #             ::self.space_factor, 
        #             ::self.space_factor, 
        #             ::self.space_factor, 
        #             ::self.time_factor,
        #             ]
        # if self.feature_set is not None:
        #     _data = _data[..., self.feature_set]
            
        feat_t_whole = _data[self.cor_t[:, 0], self.cor_t[:, 1], self.cor_t[:, 2], self.time_t]
        
            
        if self.feature_transform:
            feat_t_whole = self.feature_transform(feat_t_whole)
        
        # self.normalize_data(data)
        if self.missing_rate > 0:
            random_mask = torch.rand(self.cor_t.size(0)) > self.missing_rate
            cor_t = self.cor_t[random_mask]
            spacial_emb_t = self.spacial_emb_t[random_mask]
            time_t = self.time_t[random_mask]
            feat_t = feat_t_whole[random_mask]
        else:
            feat_t = feat_t_whole
            cor_t = self.cor_t
            spacial_emb_t = self.spacial_emb_t
            time_t = self.time_t
            
        # feat_t_ori = feat_t.clone()
        

        # change to datapoint and store as a dataset
        graph = Data(
            cor=cor_t, time=time_t, feat=feat_t, 
            T=torch.tensor(_data.size(3)), latent_vector=self.latent_vectors[idx], pde_parameter=pde_parameter,
            space_emb=spacial_emb_t, 
            # feat_ori=feat_t_ori,
            )
        return graph
    
    def __getitem__(self, idx):
        graph = self.getitem(idx)
        # if_feat_transform = 'frame_normalize' if self.feature_transform else ''
        # graph_path = os.path.join(self.data_save_dir, 
        #                           self.keys[idx]+if_feat_transform+str(self.latent_dim)+'.pt'
        #                           )
        # if os.path.exists(graph_path):
        #     graph = torch.load(graph_path)
        # else:
        #     graph = self.getitem(idx)
        #     torch.save(graph, graph_path)
        return graph
            
    def __del__(self):
        # Ensure the file is closed when the dataset object is deleted
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
            
    
    # trainset.update_latent_vector(i, z0[latent_idx: latent_idx+tempT])
    def update_latent_vector(self, idx, tensor):
        self.latent_vectors[idx] = tensor



class GraphBurgers(Dataset):
    def __init__(self, 
                 datapath_dict,
                 latent_dim,
                 missing_rate,
                 space_factor=1, time_factor=1,
                 p_transform=None,
                 ):
        # datapath_dict includes datapath and the index_list of corresponding datapath, and parameter value of this datapath
        # dataset include ['t-coordinate', 'tensor', 'x-coordinate']
        # tensor shape (N, tdim, xdim)
        # tdim: 201+1, xdim:1024

        super().__init__()
        
        # self.datapath_dict = datapath_dict
        self.missing_rate = missing_rate
        self.latent_dim = latent_dim
        self.space_factor = space_factor
        self.time_factor = time_factor
        self.p_transform = p_transform
        
        # assert self.space_factor is None
        # assert self.time_factor is None
        
        feature_list = []
        
        dataset = {}
        i = 0
        
        for datapath, (index_list, p) in datapath_dict.items():
            tensor = torch.from_numpy(h5py.File(datapath)['tensor'][index_list, ::time_factor, ::space_factor])
            tc = torch.from_numpy(h5py.File(datapath)['t-coordinate'][::time_factor])
            xc = torch.from_numpy(h5py.File(datapath)['x-coordinate'][::space_factor])
            
            if self.missing_rate>0:
                mask = torch.rand(tensor.shape)>self.missing_rate
            else:
                mask = torch.ones(tensor.shape, dtype=torch.bool)
            
            
            
            for index in range(tensor.size(0)):
                # every index every t, generate cordinates and features
                cor_list = []
                feat_list = []
                time_list = []
                SE_list = []       
                # TE_list = []     
                p_list = []
                
                T = mask.size(1)
                for t in range(T):
                    tmpmask = mask[index, t, :]
                    tmptensor = tensor[index, t, :]
                    cor = tmpmask.nonzero()
                    spacial_embedding = xc[tmpmask].reshape(-1, 1)
                    # time_embedding = torch.ones(len(cor))*tc[t]
                    cor_list.append(cor)
                    feat_list.append(tmptensor[tmpmask])
                    time_list.append(torch.ones(len(cor), dtype=torch.int)*t)
                    SE_list.append(spacial_embedding)
                    # TE_list.append(time_embedding)
                
                cor_t = torch.cat(cor_list, dim=0)
                feat_t = torch.cat(feat_list, dim=0)
                feat_t = feat_t.reshape(-1, 1)
                time_t = torch.cat(time_list, dim=0)
                SE_t = torch.cat(SE_list, dim=0)
                # TE_t = torch.cat(TE_list, dim=0)
                pde_parameter=torch.tensor(p)
                if self.p_transform:
                    pde_parameter = self.p_transform(pde_parameter)
                datapoint = Data(cor=cor_t, time=time_t, 
                                 feat=feat_t, T=torch.tensor(T), latent_vector=torch.zeros(T, self.latent_dim), 
                                 space_emb=SE_t, time_emb=tc, pde_parameter=pde_parameter)
                dataset[f"{i}"] = datapoint
                i+=1
                
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        graph = self.dataset[f"{key}"]

        return graph



class GraphBurgers2D(Dataset):
    """
    2D viscous Burgers equation dataset.

    Loads from HDF5 files produced by `utils/data/generate_burgers2d.py`.

    HDF5 layout (one file per viscosity nu):
        tensor       : float32 (N, T, H, W)  -- scalar field u(x,y,t)
        t-coordinate : float32 (T,)
        x-coordinate : float32 (H,)
        y-coordinate : float32 (W,)

    The graph data contract follows GraphBurgers (1D) closely:
        cor       : (N_pts, 2)  -- integer (i, j) grid indices
        space_emb : (N_pts, 2)  -- normalized coords in [-1, 1]^2
        feat      : (N_pts, 1)  -- scalar field values
        time      : (N_pts,)    -- integer time index
        T         : scalar      -- total time frames in this sample
        latent_vector : (T, latent_dim)
        pde_parameter : scalar  -- viscosity (normalized via p_transform)
    """

    def __init__(
        self,
        datapath_dict: dict,
        latent_dim: int,
        missing_rate: float,
        space_factor: int = 1,
        time_factor: int = 1,
        p_transform=None,
    ):
        super().__init__()

        self.missing_rate = missing_rate
        self.latent_dim = latent_dim
        self.space_factor = space_factor
        self.time_factor = time_factor
        self.p_transform = p_transform

        dataset = {}
        i = 0

        for datapath, (index_list, p) in datapath_dict.items():
            with h5py.File(datapath, "r") as f:
                tensor = torch.from_numpy(
                    f["tensor"][index_list, ::time_factor, ::space_factor, ::space_factor]
                ).float()  # (N_sub, T, H, W)
                tc = torch.from_numpy(f["t-coordinate"][::time_factor]).float()

            N_sub, T, H, W = tensor.shape

            # Normalized spatial embedding grid: (H, W, 2)
            xs = 2.0 * torch.arange(H, dtype=torch.float32) / max(H - 1, 1) - 1.0
            ys = 2.0 * torch.arange(W, dtype=torch.float32) / max(W - 1, 1) - 1.0
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            spatial_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

            if self.missing_rate > 0:
                mask = torch.rand(tensor.shape) > self.missing_rate
            else:
                mask = torch.ones(tensor.shape, dtype=torch.bool)

            pde_param_val = torch.tensor(p)
            if self.p_transform is not None:
                pde_param_val = self.p_transform(pde_param_val)

            for n in range(N_sub):
                cor_list, feat_list, time_list, se_list = [], [], [], []

                for t in range(T):
                    tmpmask = mask[n, t]                        # (H, W)
                    tmptensor = tensor[n, t]                    # (H, W)
                    cor = tmpmask.nonzero(as_tuple=False)       # (K, 2)
                    se = spatial_grid[cor[:, 0], cor[:, 1]]     # (K, 2)
                    cor_list.append(cor)
                    feat_list.append(tmptensor[tmpmask].reshape(-1, 1))
                    time_list.append(
                        torch.full((len(cor),), t, dtype=torch.int)
                    )
                    se_list.append(se)

                datapoint = Data(
                    cor=torch.cat(cor_list, dim=0),
                    feat=torch.cat(feat_list, dim=0),
                    time=torch.cat(time_list, dim=0),
                    space_emb=torch.cat(se_list, dim=0),
                    T=torch.tensor(T),
                    latent_vector=torch.zeros(T, self.latent_dim),
                    pde_parameter=pde_param_val,
                )
                dataset[f"{i}"] = datapoint
                i += 1

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[f"{key}"]


def create_burgers2d_dataset(
    missing_rate: float = 0.5,
    space_factor: int = 1,
    time_factor: int = 1,
    train_num: int = 100,
    data_dir: str = "/pscratch/sd/g/gzhao27/INR/data",
    latent_dim: int = 128,
):
    """
    Build train / val / test GraphBurgers2D datasets.

    Viscosity splits (mirror the 1-D Burgers convention):
        train : nu = 0.001, 0.002, 0.005, 0.02, 0.05
        val   : nu = 0.01   (held-out viscosity)
        test  : nu = 0.002  (seen viscosity, held-out samples)

    Data files must be pre-generated with:
        python utils/data/generate_burgers2d.py --out_dir <data_dir>

    Returns:
        trainset, valset, testset, p_mean, p_std
    """
    train_p_list = (0.001, 0.002, 0.005, 0.02, 0.05)
    val_p_list   = (0.01,)
    test_p_list  = (0.002,)

    p_mean = float(np.mean(train_p_list))
    p_std  = float(np.std(train_p_list))
    p_transform = lambda t: (t - p_mean) / p_std

    path_fmt = os.path.join(data_dir, "2D_Burgers_Sols_Nu{}.hdf5")

    valnum  = 100
    testnum = 100

    def _make_dict(p_list, start, n):
        return {
            path_fmt.format(p): (list(range(start, start + n)), p)
            for p in p_list
        }

    traindict = _make_dict(train_p_list, 0,                       train_num)
    valdict   = _make_dict(val_p_list,   train_num,                valnum)
    testdict  = _make_dict(test_p_list,  train_num + valnum,       testnum)

    common = dict(
        latent_dim=latent_dim,
        space_factor=space_factor,
        time_factor=time_factor,
        p_transform=p_transform,
    )

    trainset = GraphBurgers2D(datapath_dict=traindict, missing_rate=missing_rate, **common)
    valset   = GraphBurgers2D(datapath_dict=valdict,   missing_rate=0.0,          **common)
    testset  = GraphBurgers2D(datapath_dict=testdict,  missing_rate=0.0,          **common)

    return trainset, valset, testset, p_mean, p_std


class GraphNavierStokes(Dataset):
    # properties included in grpah of all time frame:
    # graph.cor: coordinate collections, concatenate all time frame coordinate together
    # graph.time: all time stamples of each coordinate
    # T: time duration of this episode
    # latent_vector: encoded latent vector, initally zero and will be updated
    # space_emb: spatical embedding of graph.cor
    # all other dataset should also includes all these properties to run inr.py and train.py
    
    def __init__(
            self, 
            ssub=1, 
            datapath='/pscratch/sd/g/gzhao27/INR/data/ns_V1e-3_N5000_T50.mat', 
            latent_dim=128, 
            split='train', 
            datanum = 1000,
            trainnum = 1000, 
            missing_rate=0,
            ):
        
        super().__init__()

        self.path = datapath
        self.split = split
        self.missing_rate = missing_rate
        self.latent_dim = latent_dim
        self.ssub = ssub
        
        # Dataloading
        datashape = h5py.File(datapath)['u'].shape
        if split == 'train':
            
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub,  :datanum]
        elif split == 'val':
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub, trainnum:trainnum+datanum]  #data shape (50, 64, 64, 100)
        elif split == 'test':
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub, -datanum:]
        tensor = torch.from_numpy(data)
        tensor = tensor.permute(3, 1, 2, 0)

        self.height, self.width = datashape[1], datashape[2]
        self.dataset, self.mask = self.noisy_data_processing(tensor)
        # print("mask->\n",str(self.mask))
        # print("mask->\n",str(len(self.mask)))
        
    def noisy_data_processing(self, tensor):
        if self.missing_rate>0:
            mask = torch.rand(tensor.shape) > self.missing_rate
        else:
            mask = torch.ones(tensor.shape, dtype=torch.bool)
        dataset = {}
        for index in range(len(mask)):
            cor_list = []
            feat_list = []
            time_list = []
            SE_list = []

            datapoint = {}
            T = mask.size(-1)
            for t in range(T):
                tmpmask = mask[index, ..., t]
                tmptensor = tensor[index, ..., t]
                cor =tmpmask.nonzero()
                spacial_embedding = self.S_embedding(self.height, self.width, cor)
                cor_list.append(cor)
                feat_list.append(tmptensor[tmpmask])
                time_list.append(torch.ones(len(cor), dtype=torch.int)*t)
                SE_list.append(spacial_embedding)

            cor_t = torch.cat(cor_list, dim=0)
            feat_t = torch.cat(feat_list, dim=0)
            feat_t = feat_t.reshape(-1, 1)
            time_t = torch.cat(time_list, dim=0)
            spacial_emb_t = torch.cat(SE_list, dim=0)
            datapoint = Data(cor=cor_t, time=time_t, feat=feat_t, T=torch.tensor(T), latent_vector=torch.zeros(T, self.latent_dim), space_emb=spacial_emb_t)
            dataset[f"{index}"] = datapoint

        return dataset, mask

    def S_embedding(self, image_height, image_width, cor):
        x_coords, y_coords = torch.meshgrid(
                torch.arange(image_height, dtype=torch.float32),
                torch.arange(image_width, dtype=torch.float32),
                indexing='ij'
            )

        x_coords = 2.0 * x_coords / (image_width - 1) - 1.0
        y_coords = 2.0 * y_coords / (image_height - 1) - 1.0

        spatial_grid = torch.stack([x_coords, y_coords], dim=-1)
        spatial_embedding = spatial_grid[cor[:, 0]*self.ssub, cor[:, 1]*self.ssub]

        return spatial_embedding

    def len(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        graph = self.dataset[f"{key}"]
        
        return graph
    
    
    
def collate_graph_inr(data_list):
    data_list_new = []
    all_cluster_sets = []
    
    time_bias = 0 # the bias used to differentiate different batch of time, so time can act as index
    for i, data in enumerate(data_list):
        datatmp = data.clone()
        datatmp.time = data.time + time_bias
        time_bias += data.T
        data_list_new.append(datatmp)
        
        if hasattr(data, 'cluster_set') and data.cluster_set is not None:
            cluster_set = data.cluster_set
        else:
            # no clusters: create placeholders
            cluster_set = [defaultdict(dict)]
        all_cluster_sets.extend(cluster_set)
        
    batched = Batch.from_data_list(data_list_new)

    # 4) Attach the stitched‐together cluster list
    batched.cluster_set = all_cluster_sets
    return batched


class GraphNavierStokesSampling(GraphNavierStokes):
    def __init__(
            self, 
            raw, 
            # datapath,
            # splits, 
            ssub=1, 
            missing_rate = 0,
            # use_mmap = True
        ):
        
        super(GraphNavierStokes, self).__init__()
        
        # self.path = datapath
        # self.splits = splits
        self.missing_rate = missing_rate
        self.ssub = ssub
        
        
        # if use_mmap:
        #     images_mmap = RaggedMmap(datapath)
        #     raw = images_mmap[2] # (50, 64, 64, 5000)
        # else:
        #     with h5py.File(datapath, 'r') as f:
        #         raw = f['u']
        
        # TODO: Implement data selector using cfg
        data = raw[..., ::ssub, ::ssub, :]  # (T, H', W', N)

        # data = raw_subsampled[..., splits]
        tensor = torch.from_numpy(data.copy()).float()
        tensor = tensor.permute(3, 1, 2, 0) # (N, H, W, T)
        # print("tensor shape\n", tensor.shape)

        # print("tensor new shape\n", tensor.shape)
        is_all_zeros = not (tensor != 0).any().item()
        # print("is all zeros: \n", str(is_all_zeros))
        self.height = tensor.size(1)
        self.width = tensor.size(2)
        self.T = tensor.size(3)
        
        # Precompute the spatial grid for fast lookup.
        x_coords, y_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing='ij'
        )
        x_coords = 2.0 * x_coords / (self.width - 1) - 1.0
        y_coords = 2.0 * y_coords / (self.height - 1) - 1.0
        self.spatial_grid = torch.stack([x_coords, y_coords], dim=-1)  # shape: (H, W, 2)

        self.dataset, self.mask = self.noisy_data_processing(tensor)

    
    def noisy_data_processing(self, tensor):
        """
        Vectorized processing for a tensor of shape (N, H, W, T):
          - Create a binary mask for missing values.
          - For each sample, extract the (h, w, t) indices where data is present.
          - Use these indices to gather feature values and spatial embeddings.
        """
        tensor = tensor.permute(0, 3, 1, 2)
        if self.missing_rate > 0:
            mask = torch.rand(tensor.shape) > self.missing_rate
        else:
            mask = torch.ones(tensor.shape, dtype=torch.bool)

        dataset = {}
        N, T, H, W = tensor.shape

        # Process each sample (avoid inner Python loops over time)
        # print("N is", str(N))
        for idx in range(N):
            # print("idx is", str(idx))
            sample_mask = mask[idx]      # shape: (H, W, T)
            sample_tensor = tensor[idx]  # shape: (H, W, T)

            # Get all nonzero indices at once: returns (num_points, 3) with columns [h, w, t]
            # print("sample mask shape\n", sample_mask.shape)
            nonzero_idx = sample_mask.nonzero(as_tuple=False)
            if nonzero_idx.numel() == 0:
                # print("no data present @", str(idx))
                # If no data is present, create empty tensors.
                cor = torch.empty((0, 2), dtype=torch.long)
                feat = torch.empty((0, 1), dtype=torch.float32)
                time = torch.empty((0,), dtype=torch.int)
                space_emb = torch.empty((0, 2), dtype=torch.float32)
            else:
                # Optional: sort indices by time (column 2) for consistency.
                # nonzero_idx = nonzero_idx[nonzero_idx[:, 2].argsort()]

                # Coordinates from the nonzero indices.
                cor = nonzero_idx[:, 1:]  # (num_points, 2)

                # Gather feature values using the boolean mask.
                feat = sample_tensor[sample_mask].reshape(-1, 1)

                # Time labels are the third column of nonzero indices.
                time = nonzero_idx[:, 0]

                # Look up spatial embeddings using the precomputed grid.
                cor_long = cor.long()
                space_emb = self.spatial_grid[cor_long[:, 0], cor_long[:, 1]]

            # Create a data object (using torch_geometric.data.Data)
            datapoint = Data(cor=cor, space_emb=space_emb, time=time, feat=feat, T=torch.tensor(T),)
            dataset[f"{idx}"] = datapoint

        return dataset, mask
    
    def initial_latent_vector(self, latent_dim):
        for key, datapoint in self.dataset.items():
            T = datapoint.T
            latent_vector = torch.zeros(T, latent_dim).float()
            datapoint.latent_vector = latent_vector
    
        
# Navier Stokes
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
        
def create_burgers_dataset(missing_rate=0.5, space_factor=1, time_factor=1, train_num=100):
    trainnum = train_num
    valnum = 100
    testnum = 100
    train_p_list=(0.001, 0.002, 0.004, 0.02, 0.04, 0.1)
    p_mean = np.array(train_p_list).mean()
    p_std = np.array(train_p_list).std()
    p_transform = lambda tensor: (tensor - p_mean) / p_std
    p_invtransform = lambda tensor: (tensor * p_std) + p_mean
    
    val_p_list = (0.01, )
    test_p_list = (0.002, )
    path_format = "/pscratch/sd/g/gzhao27/INR/data/1D_Burgers_Sols_Nu{}.hdf5"
    def create_dict(p_list, start, Dnum):
        Ddict = {}
        for p in p_list:
            temppath = path_format.format(p)
            Ddict[temppath] = (list(range(start, start+Dnum)), p)
        return Ddict
    
    traindict = create_dict(train_p_list, 0, trainnum)
    valdict = create_dict(val_p_list, trainnum, valnum)
    testdict = create_dict(test_p_list, trainnum+valnum, testnum)
    
    trainset = GraphBurgers(
        datapath_dict=traindict,
        latent_dim=128,
        missing_rate=missing_rate,
        space_factor=space_factor,
        time_factor=time_factor,
        p_transform=p_transform,
    )
    valset = GraphBurgers(
        datapath_dict=valdict,
        latent_dim=128,
        missing_rate=0.,
        space_factor=space_factor,
        time_factor=time_factor,
        p_transform=p_transform,
    )
    testset = GraphBurgers(
        datapath_dict=testdict,
        latent_dim=128,
        missing_rate=0.,
        p_transform=p_transform
    )
    return trainset, valset, testset, p_mean, p_std


def soma_claculate_p_normalize(data_path):
    p_list = []
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            _data = f[key][:]
            p_list.append(_data[0, 0, 0, 0, -1])
    
    p_array = np.array(p_list)
    return p_array.mean(), p_array.std()

def create_soma_dataset(ntrain, mmap_dir, space_factor, time_factor, latent_dim, missing_rate, val_missing_rate, feature_set, data_path, ):
    print('for soma thedataset-impliciBottomDrag p_mean and p_std pre-calculated: 0.005342233, 0.002689823')
    p_mean, p_std = 0.005342233, 0.002689823 # for 
    p_transform = lambda tensor: (tensor - p_mean) / p_std
    p_invtransform = lambda tensor: (tensor * p_std) + p_mean
    
    trainset0 = GraphSomaDataset(
        data_path=data_path,
        train_num=ntrain, 
        feature_set=feature_set,
        space_factor=space_factor,
        time_factor=time_factor, 
        latent_dim=latent_dim,
        mmap_dir=mmap_dir,
        missing_rate=0.0,
    )
    # create a separate create normlize transform function, avoid it to be related to the dataset sampling strategy
    feat_transform, feat_inv_transform = trainset0.create_normalize_from_dataset() 
    trainset = GraphSomaDataset(
        data_path=data_path,
        train_num=ntrain, 
        feature_set=feature_set,
        space_factor=space_factor,
        time_factor=time_factor, 
        latent_dim=latent_dim,
        p_transform=p_transform,
        mmap_dir=mmap_dir,
        missing_rate=missing_rate,
    )
    trainset.update_feat_transform(feat_transform)
    valset = GraphSomaDataset(
        data_path=data_path,
        train_num=ntrain, 
        data_num=10,
        feature_set=feature_set,
        space_factor=space_factor,
        time_factor=time_factor, 
        latent_dim=latent_dim,
        split='val',
        p_transform=p_transform,
        feature_transform=feat_transform,
        mmap_dir=mmap_dir,
        missing_rate=val_missing_rate,
    )
    testset = GraphSomaDataset(
        data_path=data_path,
        train_num=ntrain, 
        data_num=10,
        feature_set=[10],
        space_factor=space_factor,
        time_factor=time_factor, 
        latent_dim=latent_dim,
        split='test',
        p_transform=p_transform,
        feature_transform=feat_transform,
        mmap_dir=mmap_dir,
    )
    
    return trainset, valset, testset, feat_transform, feat_inv_transform
# Original split ratios: 0.7, 0.15, 0.15
def create_ns_dataset(datapath, latent_dim=256, space_factor=1, split_ratios=(0.5, 0.25, 0.25), seed=42, data_type='mmap', single_image=False):
    """
    Randomly split indices into train/val/test based on given ratios.
    Optionally saves to JSON so splits can be reused.
    default datapath: "/pscratch/sd/g/gzhao27/INR/data/NS2d/ns_mmap"
    scent datapth: /pscratch/sd/g/gzhao27/INR/data/scent_data/1k
    
    The original data is in the shape of (N, T, H, W), where T is time, H is height, W is width, and N is the number of samples.
    And the function transposes the data to (T, H, W, N) for further processing.
    In the dataset GraphNavierStokesSampling, the data is transformed back to  (N, H, W, T).
    
    npy datapah: 
    """
    
    # transform to raw, raw is numpy array with shape (T = 50, 64, 64, N = 5000)
    if data_type == 'mmap':
        images_mmap = RaggedMmap(datapath)
        if datapath.endswith('ns_mmap'):
            raw = images_mmap[2] # (50, 64, 64, 5000) for NS
            # total_samples = images_mmap[2].shape[-1] # (50, 64, 64, 5000)
        elif datapath.endswith('1k'):
            raw = np.array(images_mmap)
            raw = np.transpose(raw, (1, 2, 3, 0))
            # total_samples = len(raw)
        else:
            raise NotImplementedError()
    elif datapath.endswith('hdft'):
        with h5py.File(datapath, 'r') as f:
            f = h5py.File(datapath, 'r')
            raw = f['particles']
            total_samples = raw.shape[0]
            raw = np.squeeze(raw, axis=-1)
            raw = np.transpose(raw, (1, 2, 3, 0))
            
        #     total_samples = f['particles'].shape[-1]    # original was u
        #     raw = f['particles']    # original was u
            # print("Particles shape: " + str(f['particles'].shape[-1]))
            # print("Particles raw: " + str(f['particles']))
        # f = h5py.File(datapath, 'r')
        
        # print("Compression codec:", str(raw.compression))
        # print("Filter pipeline:", str(raw.compression_opts))
        # print("First two time-slices shape:", str(raw[..., :2].shape))
        # print("Particles shape: " + str(f['particles'].shape[-1]))
        # print("Particles raw: " + str(f['particles']))
    elif datapath.endswith('npy'):
        raw = np.load(datapath)
        # raw = raw.squeeze()
        raw = np.transpose(raw, (1, 2, 3, 0))
        # total_samples = raw.shape[3]
    else:
        raise NotImplementedError()
    
    if single_image:
        chosen_N = 0    # Currently 0-3
        sel_array = raw[..., slice(chosen_N,chosen_N + 1)]
        trainset = GraphNavierStokesSampling(
            raw = sel_array,
            ssub=space_factor, 
            )
        return trainset

    total_samples = raw.shape[-1]
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    indices = np.arange(total_samples)

    train_end = int(split_ratios[0] * total_samples)
    val_end = train_end + int(split_ratios[1] * total_samples)
    test_end = train_end+val_end +int(split_ratios[2] * total_samples)

    # splits = {
    #     'train': indices[:train_end].tolist(),
    #     'val': indices[train_end:val_end].tolist(),
    #     'test': indices[val_end:test_end].tolist()
    # }
    
    splits = {
        'train': slice(0, train_end),
        'val': slice(train_end, val_end),
        'test': slice(val_end, test_end)
    }
    
    # Can use sel_array instead of train_array to specify
    # desired image for single image inr
    chosen_N = 3    # Currently 0-3
    sel_array = raw[..., slice(chosen_N,chosen_N + 1)]
    train_array = raw[..., splits['train']]
    val_array = raw[..., splits['val']]
    test_array = raw[..., splits['test']]
    
    
    # raw shape (T, H, W, N)
    trainset = GraphNavierStokesSampling(
        raw = sel_array,
        ssub=space_factor, 
        )
    
    valset = GraphNavierStokesSampling(
        raw = val_array,
        ssub=space_factor, 
        )
    
    testset = GraphNavierStokesSampling(
        raw = test_array,
        ssub=space_factor, 
        )
    
    trainset.initial_latent_vector(latent_dim)
    valset.initial_latent_vector(latent_dim)
    testset.initial_latent_vector(latent_dim)
    # print("Trainset Total:\n", str(trainset))
    # print("Trainset First:\n", str(trainset[0]))
    # print("Trainset Array:\n", str(train_array))
    # print("Trainset Array First:\n", str(train_array[999,:,:,0].shape))

    return trainset, valset, testset

# def create_ks_dataset(datapath, latent_dim, space_factor=1, split_ratios=(0.7, 0.15, 0.15), seed=42):
#     # datapath = /pscratch/sd/g/gzhao27/INR/data/KuramotoSivashinsky/sample300_time101_space160.npy
#     raw = np.load(datapath)
    
    
#     train_end = int(split_ratios[0] * total_samples)
#     val_end = train_end + int(split_ratios[1] * total_samples)
#     test_end = train_end+val_end +int(split_ratios[2] * total_samples)
    

def piecewise_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 10.0,
    beta: float = 1.0,
    eps: float = None,
) -> torch.Tensor:
    """
    Analytically-defined piecewise function on [0, 1]^2:

        f(x, y) = sin(2πx) + β * I(x > 0.5) * sin(2π·α·y)

    The domain is split at x = 0.5: the left half contains only a smooth
    low-frequency signal while the right half also carries a high-frequency
    oscillation in y.  This creates a sharp spectral discontinuity that
    challenges uniform-sampling INR baselines.

    Args:
        x, y : tensors with identical shape, values in [0, 1].
        alpha : frequency multiplier for the high-frequency y-term.
        beta  : amplitude of the high-frequency term.
        eps   : if None, use a hard step indicator at x = 0.5;
                otherwise use a smooth sigmoid with temperature eps.

    Returns:
        f : tensor with the same shape as x / y.
    """
    base = torch.sin(2.0 * math.pi * x)
    indicator = (x > 0.5).float() if eps is None else torch.sigmoid((x - 0.5) / eps)
    high_freq = torch.sin(2.0 * math.pi * alpha * y)
    return base + beta * indicator * high_freq


class GraphPiecewise2D(Dataset):
    """
    Synthetic 2D dataset based on an analytically-defined piecewise function.

        f(x, y) = sin(2πx) + β * I(x > 0.5) * sin(2π·α·y)

    Why on-the-fly generation is correct here (unlike Burgers 2D):
        Evaluating a closed-form expression on a (H, W) grid is pure tensor
        math and takes O(H·W) microseconds regardless of resolution.  There
        is no PDE to solve, no disk I/O, and no pre-generation step needed.
        The image is created once inside __init__ and stored in memory, just
        as GraphNavierStokesSampling holds its data in a dict after loading.

    The dataset exposes a single sample with T = 1 time frame, following the
    single-image INR training flow.  Set data.single_time_frame = 0 in the
    config (or omit it; the default works).

    Graph contract (identical to other 2D datasets):
        cor       : (K, 2)  -- integer (i, j) pixel indices
        space_emb : (K, 2)  -- normalised coords in [-1, 1]^2
        feat      : (K, 1)  -- function values f(x, y)
        time      : (K,)    -- all-zeros (single frame)
        T         : scalar  -- 1
        latent_vector : (1, latent_dim)
    """

    def __init__(
        self,
        resolution: int = 256,
        alpha: float = 10.0,
        beta: float = 1.0,
        eps: float = None,
        latent_dim: int = 128,
        missing_rate: float = 0.0,
    ):
        super().__init__()
        H = W = resolution

        # Physical coordinates in [0, 1]
        xs = torch.linspace(0.0, 1.0, W)
        ys = torch.linspace(0.0, 1.0, H)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")  # (H, W)

        # Evaluate piecewise function
        image = piecewise_fn(grid_x, grid_y, alpha=alpha, beta=beta, eps=eps)  # (H, W)

        # Normalised spatial embedding in [-1, 1]^2  (matches NS/Burgers2D convention)
        se_x = 2.0 * grid_x - 1.0
        se_y = 2.0 * grid_y - 1.0
        spatial_grid = torch.stack([se_x, se_y], dim=-1)  # (H, W, 2)

        T = 1
        mask = (
            torch.rand(H, W) > missing_rate
            if missing_rate > 0
            else torch.ones(H, W, dtype=torch.bool)
        )

        cor  = mask.nonzero(as_tuple=False)          # (K, 2)
        se   = spatial_grid[cor[:, 0], cor[:, 1]]    # (K, 2)
        feat = image[mask].reshape(-1, 1)             # (K, 1)
        time = torch.zeros(len(cor), dtype=torch.int)

        datapoint = Data(
            cor=cor,
            feat=feat,
            time=time,
            space_emb=se,
            T=torch.tensor(T),
            latent_vector=torch.zeros(T, latent_dim),
        )
        self.dataset = {"0": datapoint}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key):
        return self.dataset["0"]


def create_piecewise_dataset(
    resolution: int = 256,
    alpha: float = 10.0,
    beta: float = 1.0,
    eps: float = None,
    latent_dim: int = 128,
) -> GraphPiecewise2D:
    """
    Build a GraphPiecewise2D dataset (single image, in-memory, no disk I/O).

    The full dense image is stored; coordinate sub-sampling is performed by
    the INR sampler at training time (matching the single-image flow for NS).

    Returns:
        dataset : GraphPiecewise2D — use directly as trainset.
    """
    return GraphPiecewise2D(
        resolution=resolution,
        alpha=alpha,
        beta=beta,
        eps=eps,
        latent_dim=latent_dim,
        missing_rate=0.0,
    )


def get_graph_t_idx(graph, t) -> torch.Tensor:
    indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
    # print("graph time in get_graph_t_idx\n", str(t))
    # print("indices\n", str(indices_t.shape))
    return indices_t