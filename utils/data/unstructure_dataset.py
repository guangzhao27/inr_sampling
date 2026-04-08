import os
import math
import glob
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


class GraphBurgers2D(Dataset):
    """
    Graph dataset for generated 2D Burgers trajectories.

    Expected HDF5 layout per file:
      - tensor       : (N, T, H, W)
      - t-coordinate : (T,)
      - x-coordinate : (W,) optional for loading
      - y-coordinate : (H,) optional for loading
    """

    def __init__(
        self,
        file_path,
        latent_dim=256,
        ssub=1,
        missing_rate=0.0,
        sample_idx=0,
    ):
        super().__init__()
        self.file_path = file_path
        self.latent_dim = latent_dim
        self.ssub = ssub
        self.missing_rate = missing_rate

        with h5py.File(file_path, "r") as f:
            if "tensor" not in f:
                raise KeyError(f"Missing key 'tensor' in {file_path}")
            raw = f["tensor"][:]  # (N, T, H, W)

        if raw.ndim != 4:
            raise ValueError(
                f"Expected tensor with 4 dims (N,T,H,W), got shape {raw.shape} from {file_path}"
            )

        if sample_idx is not None:
            if sample_idx < 0 or sample_idx >= raw.shape[0]:
                raise IndexError(
                    f"sample_idx={sample_idx} out of range for {file_path} with N={raw.shape[0]}"
                )
            raw = raw[sample_idx : sample_idx + 1]

        # Optional spatial subsampling.
        raw = raw[:, :, ::ssub, ::ssub]
        tensor = torch.from_numpy(raw.copy()).float()  # (N, T, H, W)

        self.height = tensor.size(2)
        self.width = tensor.size(3)
        self.T = tensor.size(1)

        x_coords, y_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij",
        )
        x_coords = 2.0 * x_coords / (self.width - 1) - 1.0
        y_coords = 2.0 * y_coords / (self.height - 1) - 1.0
        self.spatial_grid = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

        self.dataset = self._build_dataset(tensor)

    def _build_dataset(self, tensor):
        dataset = {}
        N, T, H, W = tensor.shape

        for n in range(N):
            cor_list = []
            feat_list = []
            time_list = []
            emb_list = []

            for t in range(T):
                if self.missing_rate > 0:
                    mask_t = torch.rand(H, W) > self.missing_rate
                else:
                    mask_t = torch.ones(H, W, dtype=torch.bool)

                cor_t = mask_t.nonzero(as_tuple=False)  # (K_t, 2)
                feat_t = tensor[n, t][mask_t].reshape(-1, 1)
                time_t = torch.full((len(cor_t),), t, dtype=torch.long)
                emb_t = self.spatial_grid[cor_t[:, 0], cor_t[:, 1]]

                cor_list.append(cor_t)
                feat_list.append(feat_t)
                time_list.append(time_t)
                emb_list.append(emb_t)

            datapoint = Data(
                cor=torch.cat(cor_list, dim=0),
                feat=torch.cat(feat_list, dim=0),
                time=torch.cat(time_list, dim=0),
                space_emb=torch.cat(emb_list, dim=0),
                T=torch.tensor(T),
                latent_vector=torch.zeros(T, self.latent_dim),
            )
            dataset[str(n)] = datapoint

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[str(key)]


def create_burgers2d_dataset(
    data_dir,
    nu=0.01,
    latent_dim=256,
    space_factor=1,
    seed=42,
    single_image=True,
    sample_idx=0,
):
    """
    Create a 2D Burgers graph dataset from generated HDF5 files.

    Args:
        data_dir: directory containing 2D_Burgers_Sols_Nu*.hdf5, or a specific file path
        nu: viscosity value used in file naming (e.g. 0.01)
        latent_dim: latent vector width
        space_factor: spatial subsampling stride
        seed: reserved for future split behavior
        single_image: only True is supported in current single-image INR flow
        sample_idx: which trajectory index to load from the selected HDF5 file
    """
    del seed  # reserved for future split support

    if os.path.isfile(data_dir):
        file_path = data_dir
    else:
        file_path = os.path.join(data_dir, f"2D_Burgers_Sols_Nu{nu}.hdf5")
        if not os.path.exists(file_path):
            candidates = sorted(glob.glob(os.path.join(data_dir, "2D_Burgers_Sols_Nu*.hdf5")))
            if not candidates:
                raise FileNotFoundError(
                    f"No Burgers2D files found in {data_dir}. Expected 2D_Burgers_Sols_Nu*.hdf5"
                )
            raise FileNotFoundError(
                f"Could not find {file_path}. Available files: {[os.path.basename(c) for c in candidates]}"
            )

    if not single_image:
        raise NotImplementedError(
            "create_burgers2d_dataset currently supports single_image=True only"
        )

    return GraphBurgers2D(
        file_path=file_path,
        latent_dim=latent_dim,
        ssub=space_factor,
        missing_rate=0.0,
        sample_idx=sample_idx,
    )


class GraphPoolBoiling2D(Dataset):
    """
    Graph dataset for PoolBoiling 2D trajectories stored in HDF5 files.

    Expected per-file layout:
      - temperature: (T, H, W) preferred
      - or an equivalent 4D tensor (N, T, H, W)
    """

    def __init__(
        self,
        file_path,
        field_key="temperature",
        latent_dim=256,
        ssub=1,
        missing_rate=0.0,
        sample_idx=0,
    ):
        super().__init__()
        self.file_path = file_path
        self.field_key = field_key
        self.latent_dim = latent_dim
        self.ssub = ssub
        self.missing_rate = missing_rate

        with h5py.File(file_path, "r") as f:
            if field_key not in f:
                raise KeyError(
                    f"Missing key '{field_key}' in {file_path}. Available keys: {list(f.keys())}"
                )
            raw = f[field_key][:]

        if raw.ndim == 3:
            raw = raw[None, ...]  # (N=1, T, H, W)
        elif raw.ndim != 4:
            raise ValueError(
                f"Expected '{field_key}' with 3 or 4 dims, got shape {raw.shape} from {file_path}"
            )

        if sample_idx is not None:
            if sample_idx < 0 or sample_idx >= raw.shape[0]:
                raise IndexError(
                    f"sample_idx={sample_idx} out of range for {file_path} with N={raw.shape[0]}"
                )
            raw = raw[sample_idx : sample_idx + 1]

        raw = raw[:, :, ::ssub, ::ssub]
        tensor = torch.from_numpy(raw.copy()).float()  # (N, T, H, W)

        self.height = tensor.size(2)
        self.width = tensor.size(3)
        self.T = tensor.size(1)

        x_coords, y_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij",
        )
        x_coords = 2.0 * x_coords / (self.width - 1) - 1.0
        y_coords = 2.0 * y_coords / (self.height - 1) - 1.0
        self.spatial_grid = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

        self.dataset = self._build_dataset(tensor)

    def _build_dataset(self, tensor):
        dataset = {}
        N, T, H, W = tensor.shape

        for n in range(N):
            cor_list = []
            feat_list = []
            time_list = []
            emb_list = []

            for t in range(T):
                if self.missing_rate > 0:
                    mask_t = torch.rand(H, W) > self.missing_rate
                else:
                    mask_t = torch.ones(H, W, dtype=torch.bool)

                cor_t = mask_t.nonzero(as_tuple=False)
                feat_t = tensor[n, t][mask_t].reshape(-1, 1)
                time_t = torch.full((len(cor_t),), t, dtype=torch.long)
                emb_t = self.spatial_grid[cor_t[:, 0], cor_t[:, 1]]

                cor_list.append(cor_t)
                feat_list.append(feat_t)
                time_list.append(time_t)
                emb_list.append(emb_t)

            datapoint = Data(
                cor=torch.cat(cor_list, dim=0),
                feat=torch.cat(feat_list, dim=0),
                time=torch.cat(time_list, dim=0),
                space_emb=torch.cat(emb_list, dim=0),
                T=torch.tensor(T),
                latent_vector=torch.zeros(T, self.latent_dim),
            )
            dataset[str(n)] = datapoint

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[str(key)]


def create_poolboiling2d_dataset(
    data_dir,
    latent_dim=256,
    space_factor=1,
    seed=42,
    single_image=True,
    sample_idx=0,
    field_key="temperature",
    condition=100,
    file_name=None,
):
    """
    Create a PoolBoiling 2D graph dataset from Twall-*.hdf5 files.

    Args:
        data_dir: directory containing Twall-*.hdf5 files, or a specific file path.
        latent_dim: latent vector width.
        space_factor: spatial subsampling stride.
        seed: reserved for future split behavior.
        single_image: only True is supported in current single-image INR flow.
        sample_idx: sample index when source tensor is 4D (N, T, H, W).
        field_key: HDF5 key to read, e.g. "temperature".
        condition: wall temperature suffix used when selecting default file.
        file_name: optional explicit filename (e.g. "Twall-100.hdf5").
    """
    del seed  # reserved for future split support

    if os.path.isfile(data_dir):
        file_path = data_dir
    else:
        if file_name is not None:
            file_path = os.path.join(data_dir, file_name)
        else:
            file_path = os.path.join(data_dir, f"Twall-{condition}.hdf5")

        if not os.path.exists(file_path):
            candidates = sorted(glob.glob(os.path.join(data_dir, "Twall-*.hdf5")))
            if not candidates:
                raise FileNotFoundError(
                    f"No PoolBoiling files found in {data_dir}. Expected Twall-*.hdf5"
                )
            raise FileNotFoundError(
                f"Could not find {file_path}. Available files: {[os.path.basename(c) for c in candidates]}"
            )

    if not single_image:
        raise NotImplementedError(
            "create_poolboiling2d_dataset currently supports single_image=True only"
        )

    return GraphPoolBoiling2D(
        file_path=file_path,
        field_key=field_key,
        latent_dim=latent_dim,
        ssub=space_factor,
        missing_rate=0.0,
        sample_idx=sample_idx,
    )


def get_graph_t_idx(graph, t) -> torch.Tensor:
    indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
    # print("graph time in get_graph_t_idx\n", str(t))
    # print("indices\n", str(indices_t.shape))
    return indices_t