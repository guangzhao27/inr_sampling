import torch
import os
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path
# sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))
import numpy as np
device = torch.device('cuda')
seg_num = 32

from train_utility_sampling.taylor_estimation import estimate_frobenius_norm_corrected, grad_estimation, grad_estimation_fully_batched
from utils.quadtree import HierarchicalImageGrid

def value_function(crop_grad, seg_number, value_type='variance'):
    if value_type == 'variance':
        crop_grad = crop_grad.reshape(256//seg_number*256//seg_number, -1)
        grad_var = crop_grad.var(axis = (0)).sum()
        return grad_var.detach().cpu()*seg_number**2

t = 100
rate = 1



for g in [16, 32, 64, 128]:
    grad_GT_list = []
    grid = HierarchicalImageGrid(1024, 1024, initial_grid_size=g)
    center, dimension = grid.get_leaf_centers_tensor()
    bounds = grid.get_leaf_properties_tensor()
    
    print(bounds)
    seg_number = g//4
    seg_index = [(x, x+256//seg_number) for x in range(0, 256, 256//seg_number)]
    for a in range(4):
        for b in range(4):
            torch.cuda.empty_cache()
            batch_grad = torch.from_numpy(
                np.load(f'all_pixel_grad_t{t}_rate{rate}_row{a}_colume{b}.npy')
            )
            print(f'all_pixel_grad_t{t}_rate{rate}_row{a}_colume{b}.npy')
            batch_grad = batch_grad.reshape(256, 256, -1)
            for r_range in seg_index:
                for c_range in seg_index:
                    
                    crop_grad = batch_grad[r_range[0]:r_range[1], c_range[0]:c_range[1], ]
                    grad_var = value_function(crop_grad, seg_number, 'variance')
                    grad_GT_list.append(grad_var)
    grad_GT_array = np.array(grad_GT_list)
    np.save(f't_100_grid_{g}_grad_GT.npy', grad_GT_array)

    # for a in range(4):
    #     for b in range(4):
    #         for i in range(256*a//rate, 256*(a+1)//rate):
    #             for j in range(256*b//rate, 256*(b+1)//rate):
    #                 loss = per_pix_losses[i, j]
    #                 point_grad = torch.autograd.grad(
    #                     loss, params, retain_graph=True, allow_unused=True
    #                 )
    #                 grads_all = [g.view(-1) for g in point_grad if g is not None]
    #                 grad_vec = torch.cat(grads_all)
    #                 all_pix_grads.append(grad_vec.cpu())
    #         batch_grad = torch.stack(all_pix_grads)
    #         np.save(f'all_pixel_grad_t{t}_rate{rate}_row{a}_colume{b}.npy',batch_grad.cpu().numpy())