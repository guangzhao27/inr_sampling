from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.parallel import DistributedDataParallel as DDP
from . import losses
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import os

def get_grad_norm(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().view(-1))
    concat_grads = torch.cat(grads)
    total_norm = torch.norm(concat_grads)
    return total_norm

def grad_norm_per_pixel(model, per_pix_losses, optimizer):
    pix_norms = []
    all_pix_grads = []
    losses = per_pix_losses.view(-1)
    len_losses = len(losses)

    for i, pix_loss in enumerate(losses):
        pix_grads = []
        optimizer.zero_grad()
        pix_loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                pix_grads.append(param.grad.detach().view(-1))
        concat_grads = torch.cat(pix_grads)
        all_pix_grads.append(concat_grads)
        final_pix_norm = torch.norm(concat_grads)
        pix_norms.append(final_pix_norm.item())

    return pix_norms, all_pix_grads

def grad_norm_pixel_image(pix_norms, step, cfg):
    norms_matrix = np.array(pix_norms).reshape(64, 64)
    plt.figure(figsize=(12,12))
    sns.heatmap(
        norms_matrix,
        linewidths=0.5,
        norm=mcolors.LogNorm(),
        xticklabels=False,
        yticklabels=False)
    depth = cfg.inr.depth
    title = f'Per-Pixel Grad Norms Step {step} Depth {depth}'
    plt.title(title)
    parent_dir = "/sdcc/u/smccue/projects/inr_sampling/visuals/norms"
    path = os.path.join(parent_dir, f"depth_{depth}")
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory created")
    except OSError as error:
        print("Directory can not be created")

    save_path = f"/sdcc/u/smccue/projects/inr_sampling/visuals/norms/depth_{depth}/pixel_grad_norms_depth_{depth}_step_{step}.png"
    plt.savefig(save_path)
    plt.close()
    
def gradient_similarity(pix_norms, pix_grads, step, cfg, loss, optimizer, inr):
    depth = cfg.inr.depth
    print("pix grads shape: " + str(len(pix_grads)))
    sel = pix_grads[2080].view(-1)
    similarities = []
    cos = nn.CosineSimilarity(dim=0)
    for grad in pix_grads:
        similarities.append(abs(cos(sel, grad.view(-1)).item()))
    similarity_matrix = np.array(similarities).reshape(64, 64)
    print(similarity_matrix)
    plt.figure(figsize=(12,12))
    sns.heatmap(
        similarity_matrix,
        cmap="hot",
        linewidths=0.0,
        # norm=mcolors.LogNorm(),
        xticklabels=False,
        yticklabels=False)
    depth = cfg.inr.depth
    title = f'Gradient Correlation Step {step} Depth {depth} {cfg.sampling.type}'
    plt.title(title)
    parent_dir = "/sdcc/u/smccue/projects/inr_sampling/visuals/norms"
    path = os.path.join(parent_dir, f"depth_{depth}/correlation/{cfg.sampling.type}")
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory created")
    except OSError as error:
        print("Directory can not be created")

    save_path = f"/sdcc/u/smccue/projects/inr_sampling/visuals/norms/depth_{depth}/correlation/{cfg.sampling.type}/gradient_correlation_depth_{depth}_step_{step}.png"
    plt.savefig(save_path)
    plt.close()

def graph_inner_loop(
    func_rep,
    graph_ori,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    sampler=None,
    outer_step=0,
    save_image=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    
    
    fitted_modulations = torch.zeros_like(graph_ori.latent_vector).requires_grad_()
    
    for inner_step in range(inner_steps):
        # print("In the inner step")
        if sampler is not None:
            # print("Reached the sample step")
            graph = sampler.sample(
                outer_step=outer_step, 
                inner_step=inner_step, 
                graph=graph_ori, 
                modulations=fitted_modulations, 
                save_image=save_image)
        else:
            graph = graph_ori
        coords = graph.space_emb
        features = graph.feat
        batch_index = graph.time
        # TODO: graph = inr.sample(graph, sample_params)
        # coords = graph.space_emb
        # features = graph.feat
        # graph = sampler.sample(step, graph, sample_params)
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                graph_inner_loop_step,
                func_rep,
                fitted_modulations,
                coords,
                features,
                batch_index,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
                loss_type,
            )
        else:
            fitted_modulations = graph_inner_loop_step(
                func_rep,
                fitted_modulations,
                coords,
                features,
                batch_index,
                inner_lr,
                is_train,
                gradient_checkpointing,
                loss_type,
            )
    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    last_element=False,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = modulations.shape[0]
    if loss_type == "mse":
        element_loss_fn = losses.per_element_mse_fn
    elif loss_type == "nll":
        element_loss_fn = losses.per_element_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        element_loss_fn = partial(
            losses.per_element_multi_scale_fn,
            loss_name=loss_name,
            last_element=last_element,
        )

    loss = 0
    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch

        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        # features = features.reshape(-1, 1)
        assert features_recon.shape == features.shape, 'two matrix should have same shape'
        loss = ((features_recon - features) ** 2).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    return modulations - inner_lr * grad

def mse_points_image(per_pix_losses, step, cfg):
    plt.figure(figsize=(12,12))
    sns.heatmap(
        per_pix_losses,
        cmap="YlOrBr",
        linewidths=0.5,
        norm=mcolors.LogNorm(),
        xticklabels=False,
        yticklabels=False)
    depth = cfg.inr.depth
    title = f'Per-Pixel Losses Step {step} Depth {depth}'
    plt.title(title)
    parent_dir = "/sdcc/u/smccue/projects/inr_sampling/visuals/norms"
    path = os.path.join(parent_dir, f"depth_{depth}")
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory created")
    except OSError as error:
        print("Directory can not be created")

    save_path = f"/sdcc/u/smccue/projects/inr_sampling/visuals/norms/depth_{depth}/pixel_mse_depth_{depth}_step_{step}.png"
    plt.savefig(save_path)
    plt.close()

def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    iter=0,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    use_rel_loss=False,
    loss_type="mse",
    detach_modulations=False,
    feat_inv_transform=None,
    sampler=None,
):
    """
    graph.time is actually graph.time, it's used to distinguish different time frame, 
    so that modulated_forward can handdle a batch of data together with a single tensor calculation
    
    

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
            
    Return:
        modulation: Shape (batch_size, hidden_dim). Note that with sub_array_num>1, the batch_size is the sub_batch_size,  
        and the graph.time is adjust to start with 0, i.e. (3, 4, 5) -> (0, 1, 2)
    """
    
    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(graph)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    # modulations = torch.zeros_like(graph.latent_vector).requires_grad_()
    coords = graph.space_emb
    features = graph.feat

    # Run inner loop
    modulations = graph_inner_loop(
        func_rep,
        graph,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
        sampler=sampler,
        outer_step=iter,
    )

    if detach_modulations:
        modulations = modulations.detach()  # 1er ordre

    loss = 0
    batch_size = modulations.shape[0]

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[graph.time])
        if feat_inv_transform:
            features_recon = feat_inv_transform(features_recon)
            features = feat_inv_transform(features)
        loss = ((features_recon - features) ** 2).mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = losses.relative_rmse(features_recon, features)
        # eps = 1e-8
        # rel_loss = (((features_recon - features) ** 2).mean() /
        #                 ((features ** 2).mean() + eps))
        outputs["rel_loss"] = rel_loss

    return outputs

def single_image_step(
    inr, 
    graph_ori, 
    iter,
    is_train=True,
    return_reconstructions=False,
    use_rel_loss=False,
    sampler=None, 
    cfg=None,
    optimizer=None
):
    step = iter
    if sampler is not None:
        if cfg is not None and cfg.sampling.type == "EVOS":
            graph = sampler.sample(graph_ori, step)
        else:
            graph = sampler.sample(
                inner_step=0, 
                graph=graph_ori, 
                save_image=False
            )
    else:
        graph = graph_ori
    features = graph.feat
    coords = graph.space_emb
    features_recon = inr(coords)

    # print("---graph---\n" + str(graph) + "\n ---features---\n" + str(features) + "\n ---coords---\n" + str(coords) + "\n ---recon---\n" + str(features_recon))

       


    # print("---FR---")
    # print(features_recon)
    # print("---PPL---")
    
    if is_train and cfg.sampling.type == "EVOS":
        # print("===p recon===\n" + str(features_recon) + "\n===p features ===\n" + str(features) + "\n===step===\n" + str(step))
        # loss = sampler._sampler_compute_loss(features_recon, features, step)
        sampler._sampler_compute_loss(features_recon, features, step)
    # else:
        # loss = ((features_recon - graph.feat)**2).mean()
        # if not is_train:
            # print("---features recon---\n" + str(features_recon.shape))
            # print("---gt features---\n" + str(graph.feat.shape))
    loss = F.mse_loss(features_recon, graph.feat)

    # if iter % 100 == 0 and cfg is not None and optimizer is not None:
    #     per_pix_losses = ((features_recon - graph.feat)**2)
    # #     pix_norms, pix_grads = grad_norm_per_pixel(inr, per_pix_losses, optimizer)
    #     # grad_norm_pixel_image(pix_norms, step, cfg)
    #     # gradient_similarity(pix_norms, pix_grads, step, cfg, loss, optimizer, inr)
    #     # print(per_pix_losses)
    #     # print("---PLL---")
    #     pix_losses_list = per_pix_losses.mean(dim=1).cpu().detach().tolist()
    #     # print(pix_losses_list)
    #     # print("---MATRIX---")
    #     matrix = np.array(pix_losses_list).reshape(512, 512)
    #     # print(matrix)
    #     mse_points_image(matrix, step, cfg)

    outputs = {
        "loss": loss,
        "reconstructions": features_recon if return_reconstructions else None,
        "rel_loss": losses.relative_rmse(features_recon, features) if use_rel_loss else None,
    }
    
    return outputs