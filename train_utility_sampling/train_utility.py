import torch
import torch.nn as nn
from torch_geometric.data import Data
from .metalearning_sampling import graph_outer_step as outer_step
from .metalearning_sampling import graph_inner_loop, single_image_step
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
def divide_array_indexes(N, k):
    # Base size and extra elements
    base_size = N // k
    extra = N % k
    
    # Create the index ranges
    indexes = []
    start = 0
    for i in range(k):
        end = start + base_size + (1 if i < extra else 0)
        indexes.append(torch.tensor(list(range(start, end))))
        start = end
    
    return indexes

def split_graph_by_time(graph: Data, N: int) -> list[Data]:
    """
    Split a torch_geometric Data object into N subgraphs along the time dimension.
    
    The graph is assumed to have the following attributes:
      - time: a 1D tensor of shape [num_nodes] with integer time indices (range 0 to T_total-1)
      - cor: tensor of coordinates (shape [num_nodes, ...])
      - feat: tensor of features (shape [num_nodes, ...])
      - space_emb: tensor of spatial embeddings (shape [num_nodes, ...])
      - latent_vector: a tensor of shape [T_total, latent_dim]
      
    Args:
        graph (Data): The full graph data.
        N (int): Number of subgraphs (iterations) to split the time range into.
    
    Returns:
        list: A list of Data objects (subgraphs) corresponding to each time segment.
    """
    
    if N == 1:
        return [graph]
    
    if N >1:
        raise NotImplementedError()
    
    # Determine total number of time steps from the 'time' attribute.
    T_total = graph.T.sum().item() # assumes time indices start at 0

    # Compute boundaries to split [0, T_total-1] into N segments.
    # Using linspace ensures nearly equal-sized segments.
    boundaries = torch.linspace(0, T_total, steps=N+1)
    subgraphs = []
    
    for i in range(N):
        t_start = int(boundaries[i].item())
        t_end = int(boundaries[i+1].item())
        # For segments except the last, include times in [t_start, t_end)
        # For the last segment, include the upper boundary (t_end).
        if i < N - 1:
            mask = (graph.time >= t_start) & (graph.time < t_end)
        else:
            mask = (graph.time >= t_start) & (graph.time <= t_end)
        
        # Index the nodes corresponding to the time slice.
        sub_cor = graph.cor[mask]
        sub_feat = graph.feat[mask]
        sub_space_emb = graph.space_emb[mask]
        sub_time = graph.time[mask]
        # time is the index for frames, it should start with 0
        sub_time = sub_time - sub_time.min() 
        
        # Slice the latent_vector corresponding to the time segment.
        # (Assuming latent_vector is ordered by time.)
        if i < N - 1:
            sub_latent_vector = graph.latent_vector[t_start:t_end]
        else:
            sub_latent_vector = graph.latent_vector[t_start:]
        sub_T = sub_latent_vector.shape[0]
        
        # Create the subgraph with updated T and latent_vector.
        sub_data = Data(
            cor=sub_cor,
            time=sub_time,
            feat=sub_feat,
            space_emb=sub_space_emb,
            T=torch.tensor(sub_T),
            latent_vector=sub_latent_vector, 
        )
        subgraphs.append(sub_data)
    
    return subgraphs

def train_step(step, train_loader, inr, sub_array_num, device, 
               inner_steps, alpha, use_rel_loss, optimizer, 
               sample_params, sampler=None,
               ):
    
    rel_train_mse = 0
    fit_train_mse = 0
    ntrain = len(train_loader.dataset)
    
    for substep, graph_ori in enumerate(train_loader):
        # graph include: cor, time, feat, space_emb; T, latent_vector;
        # sampled_graph = inr.sample(graph_ori, sample_params)
        sub_graph_list = split_graph_by_time(graph_ori, sub_array_num)
        total_time = graph_ori.T.sum()
            
        # torch.cuda.empty_cache()
        inr.train()
        
        # when set missing rate, then not using sub_array_num for training procedure
        # sub_sample_loader for the has down sampled graph
        
        
        for graph in sub_graph_list:
            graph_time = graph.T.sum()

            # graph = inr.sample(graph, sample_params)
            graph.to(device)
            outputs = outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                iter=step,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=True,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
                sampler=sampler, 
            )
            
            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            optimizer.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse += loss.item() * graph_time / total_time
            
            if use_rel_loss:
                rel_train_mse += outputs["rel_loss"].item() * graph_time / total_time
    
    train_loss = fit_train_mse / ntrain
    
    if use_rel_loss:
        rel_train_loss = rel_train_mse / ntrain
        print('rel train loss:', rel_train_loss)
    else:
        rel_train_loss = None
        print('train loss:', train_loss)
    
    return train_loss, rel_train_loss

def save_sampling_result(step, val_loader, inr, sub_array_num, device,
                        inner_steps, alpha, sampler,
                        ):
    graph_ori = next(iter(val_loader))
    
    graph_ori = graph_ori[0]
    inr.eval()
    inr.zero_grad()
    
    if isinstance(inr, DDP):
        inr = inr.module
    
    graph_ori.to(device)
    inr.to(device)
    
    graph_inner_loop(
        inr, graph_ori, inner_steps, alpha, 
        is_train=False, gradient_checkpointing=False, 
        loss_type='mse', sampler=sampler, 
        outer_step=step, save_image=True,
    )

def save_sampling_result_single_image(
    step, graph, inr, device,
    inner_steps, sampler,
):
    inr.eval()
    inr.zero_grad()
    
    if isinstance(inr, DDP):
        inr = inr.module
    
    graph.to(device)
    inr.to(device)
    sampler.sample(
        inner_step=step,
        graph=graph,
        save_image=True, 
    )
    return
    
    
    

def validation_step(step, val_loader, inr, sub_array_num, device, 
                    inner_steps, alpha, use_rel_loss, 
                    sample_params, sampler=None,
                    ):
    
    fit_test_mse = 0
    rel_test_mse = 0
    print('need to check if the loss is still the same with different sub array num')
    # https://vscode.dev/github/LouisSerrano/coral/blob/main/coral/metalearning.py#L294
    ntest = len(val_loader.dataset)
    
    print('todo')
    for substep, graph_ori in enumerate(val_loader):
        
        # save sampled images in validation steps. ..
        
        
        inr.eval()
        # sampled_graph = inr.sample(graph_ori, sample_params)
        sub_graph_list = split_graph_by_time(graph_ori, sub_array_num)
        total_time = graph_ori.T.sum()
        
        for graph in sub_graph_list:
            graph_time = graph.T.sum()

            
            graph.images = graph.feat.to(device)
            graph.pos = graph.space_emb.to(device)
            graph.batch = graph.time.to(device)
            graph.modulations = graph.latent_vector.to(device) #torch.zeros_like(graph.latent_vector)
            # graph.modulations = torch.zeros_like(graph.latent_vector)
            graph.to(device)

            outputs = outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                iter=step,
                is_train=False,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
                sampler=sampler, 
            )

            loss = outputs["loss"]
            fit_test_mse += loss.item() * graph_time / total_time

            if use_rel_loss:
                rel_test_mse += outputs["rel_loss"].item() * graph_time / total_time

    test_loss = fit_test_mse.item() / ntest

    if use_rel_loss:
        rel_test_loss = rel_test_mse.item() / ntest
        print(f'{step}, rel test loss:', rel_test_loss)
    else:
        rel_test_loss = 0
        print(f'{step}, train loss:', test_loss)
    
    # print(f'{step}, test loss')
    # print(test_loss)
    
    return test_loss, rel_test_loss

def train_step_single_image(step, graph, inr, device, 
               use_rel_loss, optimizer, 
               sampler=None,
               cfg=None
               ):
    
    inr.train()
    graph.to(device)
    inr.to(device)
    
    outputs = single_image_step(
        inr, 
        graph, 
        iter=step,
        is_train=True,
        return_reconstructions=False,
        use_rel_loss=use_rel_loss,
        sampler=sampler, 
        cfg=cfg,
    )
    
    # t0 = time()
    # recon = outputs['reconstructions']

    optimizer.zero_grad()
    loss = outputs['loss']
    loss.backward()

    # Attain grad vector
    grad_norm = get_grad_norm(inr)

    nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0) # remove values over 1
    optimizer.step()
    
    train_loss = outputs['loss'].cpu().detach().item()
    
    if use_rel_loss:
        rel_train_loss = outputs['rel_loss'].cpu().detach().item()
    else:
        rel_train_loss = None
    
    psnr_score = 0
    ssim_score = 0
    # print('train step time:', time() - t0)
        
    return train_loss, rel_train_loss, grad_norm

def validation_step_single_image(step, graph, inr, device, 
               use_rel_loss, optimizer, 
               sampler=None, cfg=None
               ):
    
    inr.eval()
    graph.to(device)
    inr.to(device)
    
    outputs = single_image_step(
        inr, 
        graph, 
        iter=step,
        is_train=False,
        return_reconstructions=False,
        use_rel_loss=use_rel_loss,
        sampler=None,
        cfg = cfg
    )
    
    
    val_loss = outputs['loss'].cpu().detach().item()
    if use_rel_loss:
        rel_val_loss = outputs['rel_loss'].cpu().detach().item()
    else:
        rel_val_loss = None
    
    psnr_score = outputs['psnr']
    ssim_score = outputs['ssim']

    if sampler is not None:
        _save_validation_sampling_dynamics(
            step=step,
            graph=graph,
            inr=inr,
            sampler=sampler,
            cfg=cfg,
        )
        
    return val_loss, rel_val_loss, psnr_score, ssim_score


def _sample_with_current_sampler(step: int, graph: Data, sampler, cfg):
    """Sample points with the current sampler using the same branch logic as training."""
    with torch.no_grad():
        if cfg is not None and cfg.sampling.type == "EVOS":
            return sampler.sample(graph, step)
        return sampler.sample(
            inner_step=step,
            graph=graph,
            save_image=False,
        )


def _save_validation_sampling_dynamics(step: int, graph: Data, inr, sampler, cfg=None):
    """Save per-pixel loss heatmap with sampled points and adaptive grid overlays."""
    if not hasattr(graph, "cor") or not hasattr(graph, "space_emb") or not hasattr(graph, "feat"):
        return

    with torch.no_grad():
        pred = inr(graph.space_emb)
        pixel_loss = (pred - graph.feat).pow(2).flatten()

    coords = graph.cor.long().detach().cpu()
    losses = pixel_loss.detach().cpu().numpy()
    sampled_graph = _sample_with_current_sampler(step, graph, sampler, cfg)
    sampled_coords = sampled_graph.cor.detach().cpu().numpy() if hasattr(sampled_graph, "cor") else np.empty((0, 2))

    # Single-image path is expected to have one frame and square grids.
    h = int(coords[:, 0].max().item() + 1)
    w = int(coords[:, 1].max().item() + 1)
    loss_image = np.zeros((h, w), dtype=np.float32)
    loss_image[coords[:, 0].numpy(), coords[:, 1].numpy()] = losses

    save_root = Path(getattr(sampler, "save_samples_path", "./sampled_frames"))
    save_dir = save_root / f"validation_i{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    #save one loss image
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(loss_image, cmap="hot", origin="lower")
    ax.set_title(f"Validation Loss Heatmap (step={step})")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(str(save_dir / "loss_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save sampled points overlaid on the loss image, with grid overlays when available.
    fig, ax = plt.subplots(figsize=(7, 6))

    # Use the quadtree draw helper for adaptive sampling.
    is_adaptive = getattr(sampler, "sample_type", None) == "2d_grid_adaptive"
    cached_grid = getattr(sampler, "cached_grid", None)
    if is_adaptive and cached_grid is not None:
        cached_grid.draw_with_image(loss_image, ax=ax, show_cells=True, cell_alpha=0.3)
    else:
        ax.imshow(loss_image, cmap="hot", origin="lower")

    # For linear-grid samplers, draw the current linearly scheduled grid.
    if getattr(sampler, "sample_type", None) in ("2d_grid_linear", "2d_grid_linear_weighted"):
        _draw_linear_grid_overlay(ax, sampler)

    if sampled_coords.size > 0:
        ax.scatter(sampled_coords[:, 1], sampled_coords[:, 0], c="cyan", s=2, alpha=0.85)

    ax.set_title(f"Validation Sampling Dynamics (step={step})")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(str(save_dir / "loss_with_samples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_linear_grid_overlay(ax, sampler):
    """Draw linear-grid cell rectangles from sampler cached bounds."""
    bounds = getattr(sampler, "cached_linear_bounds", None)
    if bounds is None or bounds.numel() == 0:
        return

    import matplotlib.patches as patches

    bounds_np = bounds.cpu().numpy()
    for x_low, x_high, y_low, y_high in bounds_np:
        rect = patches.Rectangle(
            (x_low, y_low),
            x_high - x_low + 1,
            y_high - y_low + 1,
            linewidth=0.2,
            edgecolor="lime",
            facecolor="none",
            alpha=0.3,
        )
        ax.add_patch(rect)

def get_grad_norm(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().view(-1))
    concat_grads = torch.cat(grads)
    total_norm = torch.norm(concat_grads)
    return total_norm