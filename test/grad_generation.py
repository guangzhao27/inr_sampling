import torch
import os
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path
# sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))

# Set working directory
# os.chdir("/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/")

# Verify
# print("Current working directory:", os.getcwd())
print('import set')
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.data import Data

from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader

from utils.data.unstructure_dataset import (
    create_ns_dataset,
    )
from utils.data.unstructure_dataset import (
    collate_graph_inr, 
    get_graph_t_idx,
    )
# sys.path.append(str(Path(__file__).parents[1]))
# sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
from utils.load_inr import create_inr_instance, load_inr_model

from torchdiffeq import odeint
from torch_geometric.data import DataLoader as GDataLoader
import numpy as np
from utils.quadtree import HierarchicalImageGrid
# from train_utility_sampling.SamplerWrapper import InrSamplerWrapper, graph_3d_cluster, graph_2d_cluster, add_cluster_label, sample_random_node_indices_per_cluster
# from mmap_ninja import RaggedMmap
# from hydra import initialize, compose

# NS_inr_save_name = 'NS_keep_for_test_file'
# NS_inr_save_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/best_result/'
device = torch.device('cuda')
print('set device')
def load_inr(i, model_dir):
    inr_save_path = os.path.join(model_dir, f"{i}.pt")
    inr_results = torch.load(inr_save_path, weights_only=False)
    cfg = inr_results['cfg']

    # create & load weights
    torch.set_default_dtype(torch.float32)
    inr = create_inr_instance(cfg, input_dim=2, output_dim=1, device=device)
    inr.load_state_dict(inr_results["inr"])
    inr.to(device).eval()
    return inr 

def grad_coor(grad1, grad2):
    return torch.dot(grad1, grad2)/torch.norm(grad1)/torch.norm(grad2)

def loss_function(features_recon, features):
    loss = features_recon
    loss = ((features_recon - features)**2)
    return loss


model_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/inr_sampling/2025-08-15-14-20-15NS1024_single_null_0.001_lr_5e-4_depth_6_end_128_t100'
model_path = os.path.join(model_dir, '0.pt')
save_results = torch.load(model_path, weights_only = False, map_location=device)

print('test single image inr')
cfg = save_results['cfg']
data_path = cfg.data.data_path
data_type = cfg.data.data_type  
seed = cfg.data.seed
trainset = create_ns_dataset(
            datapath = data_path, 
            data_type=data_type, 
            seed=seed,
            single_image=True  # If True, only use one image from the dataset
        )

train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=1, shuffle=False, )
graph = next(iter(train_loader))

inr = create_inr_instance(cfg, input_dim=2, output_dim=1, device=device)
t = cfg.data.single_time_frame 
indices_t = get_graph_t_idx(graph, t)

graph_ori = Data(
    cor=graph.cor[indices_t],
    feat=graph.feat[indices_t],
    time=torch.zeros(len(indices_t)),  # set time to 0 tensor
    space_emb=graph.space_emb[indices_t],
    T=torch.tensor(1),
)
print('load data')
torch.cuda.empty_cache()
graph = graph_ori
graph = graph.to(device)
H = graph.cor.max().item()+1

features = graph.feat.view(H, H, 1)
coords = graph.space_emb.detach().view(H, H, 2)

rate = 1

center = (341, 341)

features = features[::rate, ::rate]
coords = coords[::rate, ::rate]

for t in [100]:
    inr = load_inr(t, model_dir)
    inr.to(device)
    reconfeature = inr(coords)

    params = list(inr.parameters())
    per_pix_losses = loss_function(reconfeature, features)
    
    all_pix_grads = []
    
    for a in range(4):
        for b in range(4):
            for i in range(256*a//rate, 256*(a+1)//rate):
                for j in range(256*b//rate, 256*(b+1)//rate):
                    tempcor = coords[i, j]
                    tempf = features[i, j]
                    reconf = inr(tempcor)
                    loss = loss_function(reconf, tempf)
                    point_grad = torch.autograd.grad(
                        loss, params, retain_graph=False, allow_unused=True
                    )
                    
                    # loss = per_pix_losses[i, j]
                    # point_grad = torch.autograd.grad(
                    #     loss, params, retain_graph=True, allow_unused=True
                    # )
                    grads_all = [g.view(-1) for g in point_grad if g is not None]
                    grad_vec = torch.cat(grads_all)
                    all_pix_grads.append(grad_vec.cpu())
                    print(grad_vec)
            batch_grad = torch.stack(all_pix_grads)
            np.save(f'all_pixel_grad_t{t}_rate{rate}_row{a}_colume{b}.npy',batch_grad.cpu().numpy())
        