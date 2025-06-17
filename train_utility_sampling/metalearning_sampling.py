from functools import partial
import torch
import torch.utils.checkpoint as cp
from torch.nn.parallel import DistributedDataParallel as DDP
from . import losses

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
        if sampler is not None:
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
):
    if sampler is not None:
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
    loss = ((features_recon - graph.feat)**2).mean()
    
    outputs = {
        "loss": loss,
        "reconstructions": features_recon if return_reconstructions else None,
        "rel_loss": losses.relative_rmse(features_recon, features) if use_rel_loss else None,
    }
    
    
    
    return outputs