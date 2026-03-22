"""
Taylor Estimation Utilities for Implicit Neural Representation (INR) Training

This module provides various methods for estimating gradient variance and computing
Frobenius norms in the context of INR training with utility-based sampling.

Main Components:
1. Loss function utilities
2. Frobenius norm estimation methods
3. Gradient variance estimation methods (single and batched)
4. Window-based gradient variance calculation
"""

import torch
from torch.func import functional_call, jacrev, vmap
from collections import OrderedDict


# ============================================================================
# Section 1: Loss Function Utilities
# ============================================================================

def loss_function(features_recon, features):
    """
    Compute squared L2 loss between reconstructed and ground truth features.
    
    Args:
        features_recon: Reconstructed features from the INR model
        features: Ground truth features
        
    Returns:
        Squared L2 loss: (features_recon - features)^2
    """
    loss = ((features_recon - features)**2)
    return loss


# ============================================================================
# Section 2: Frobenius Norm Estimation Methods
# ============================================================================

def estimate_frobenius_norm_corrected(
    loss,
    model_output, 
    params,
    input_x,
    y_x,               
    n_probes: int = 10,
):
    """
    Estimate Frobenius norm of corrected Hessian using Hutchinson's trick.
    
    This function estimates the Frobenius norm of H' = H_orig - 2 y_x f_theta^T
    using random probes (Hutchinson estimator). This is specifically designed
    for squared loss functions.

    Args:
        loss: Scalar tensor representing the computed loss
        model_output: Model output with require_grad=True
        params: Iterable of model parameters (θ)
        input_x: Input tensor with requires_grad=True
        y_x: Gradient ∂y/∂x (shape matches input_x when flattened)
        n_probes: Number of random probes for estimation (default: 10)
        
    Returns:
        Scalar tensor approximating ‖H'‖_F (Frobenius norm)
    """
    # Step 1: Compute ∂L/∂θ (gradient w.r.t. parameters, shape = P)
    grad_theta = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    flat_grad = torch.cat([g.reshape(-1) for g in grad_theta if g is not None])  # [P]

    # Step 2: Generate random probe vectors
    P = flat_grad.numel()
    V = torch.randn(n_probes, P, device=flat_grad.device)  # [N, P]

    # Step 3: Compute <f_theta, v_i> for every probe (shape [N])
    # f_theta = ∂y/∂θ (model output gradient w.r.t. parameters)
    grad1 = torch.autograd.grad(
        model_output, params, retain_graph=True, allow_unused=True, create_graph=True
    )
    f_theta = torch.cat([g.reshape(-1) for g in grad1 if g is not None])
    s = V.matmul(f_theta)  # [N]

    # Step 4: Compute <grad_theta, v_i> (shape [N])
    dots = V.matmul(flat_grad)  # [N]

    # Step 5: Apply Hessian-vector products with correction
    grads_x = []
    for i in range(n_probes):
        # Compute H_orig @ v_i
        g_x = torch.autograd.grad(
            dots[i], input_x, retain_graph=True, allow_unused=True
        )[0]
        if g_x is None:
            g_x = torch.zeros_like(input_x)

        # Apply correction: H' @ v_i = H_orig @ v_i - 2 * <f_theta, v_i> * y_x
        hv_corr = g_x - 2.0 * s[i] * y_x
        grads_x.append(hv_corr.reshape(-1))  # Flatten to 1-D

    # Step 6: Estimate Frobenius norm
    stacked = torch.stack(grads_x)  # [N, D]
    frob_sq = (stacked ** 2).sum() / n_probes  # Unbiased estimator of ‖H'‖_F²
    return frob_sq.sqrt()


def estimate_frobenius_norm_swaped(
    loss_grad_x_tensor, 
    params,
    device,
):
    """
    Estimate Frobenius norm for batched input gradients.
    
    This function computes the Frobenius norm of parameter gradients
    for batched coordinate gradients (typically x and y directions).
    
    Args:
        loss_grad_x_tensor: Tensor of shape [batch_size, 2] containing gradients 
                          w.r.t. x/y coordinates
        params: Model parameters (iterable)
        device: Computation device
    
    Returns:
        Tensor of shape [batch_size] containing Frobenius norm estimates
        
    Note:
        Coordinates are assumed to be in the range [-1, 1]
    """
    batch_size = loss_grad_x_tensor.shape[0]
    grad_norms = torch.zeros(batch_size, device=device)
    
    # Compute gradient norm for each sample
    for b in range(batch_size):
        total_grad_norm = 0
        # Iterate over coordinate dimensions (x and y)
        for i in range(2):
            # Compute parameter gradients for this coordinate dimension
            grad = torch.autograd.grad(
                loss_grad_x_tensor[b, i], 
                params, 
                retain_graph=True, 
                allow_unused=True, 
                create_graph=False
            )
            # Flatten and concatenate all parameter gradients
            f_theta = torch.cat([g.reshape(-1) for g in grad if g is not None])
            total_grad_norm += (f_theta.detach()**2).sum()
        
        grad_norms[b] = total_grad_norm
        
    return grad_norms.sqrt()


def frobenius_norm_via_jacrev(inr, params, neighbor_coords, neighbor_targets, coords_step):
    """
    Compute Frobenius norms of d([dx,dy])/d(theta) using jacrev + vmap.
    
    This function uses functional differentiation (jacrev and vmap) to efficiently
    compute Frobenius norms across a batch of spatial derivatives.
    
    Args:
        inr: Implicit Neural Representation model
        params: Model parameters (dict/OrderedDict or list of tensors)
        neighbor_coords: Coordinates of 4 neighbors per center, shape [4B, 2]
                        (4 neighbors per center, stacked)
        neighbor_targets: Target values at neighbor coordinates, shape [4B, 1]
        coords_step: Step size in coordinate space (scalar float or 0-dim tensor)
        
    Returns:
        Tensor of shape [B] containing Frobenius norm for each center point
        
    Note:
        Neighbor mapping: [0:(r+1,c), 1:(r-1,c), 2:(r,c+1), 3:(r,c-1)]
    """
    # Step 1: Prepare parameters and buffers for functional_call
    if not isinstance(params, (dict, OrderedDict)):
        # Convert list/tuple of tensors to named dict from the module
        params = OrderedDict((n, p) for n, p in inr.named_parameters())
    else:
        params = OrderedDict(params)
    buffers = OrderedDict((n, b) for n, b in inr.named_buffers())
    
    # Step 2: Ensure proper grouping by centers: [B, 4, ...]
    assert neighbor_coords.dim() == 2 and neighbor_coords.size(-1) == 2
    assert neighbor_targets.dim() == 2 and neighbor_targets.size(-1) == 1
    B4 = neighbor_coords.shape[0]
    assert B4 % 4 == 0, "neighbor_* should stack 4 neighbors per center"
    B = B4 // 4
    coords_B42 = neighbor_coords.view(B, 4, 2)
    targs_B41  = neighbor_targets.view(B, 4, 1)

    device = coords_B42.device
    dtype  = coords_B42.dtype
    if not torch.is_tensor(coords_step):
        coords_step = torch.tensor(coords_step, device=device, dtype=dtype)
    else:
        coords_step = coords_step.to(device=device, dtype=dtype)

    # Step 3: Define per-sample function: params + 4 neighbors -> [dx, dy]
    def per_sample_fn(p, coords4, targets4):
        """
        Compute finite difference derivatives for one center point.
        
        Args:
            p: Parameters dict
            coords4: Coordinates of 4 neighbors [4, 2]
            targets4: Target values at 4 neighbors [4, 1]
            
        Returns:
            Tensor [2] containing [dx, dy] derivatives
        """
        # Build state for functional_call (params + buffers)
        state = OrderedDict({**p, **buffers})
        recon4 = functional_call(inr, state, (coords4,))  # Forward on 4 coords -> [4, C]
        losses4 = loss_function(recon4, targets4).reshape(4)  # [4]

        # Compute central differences
        dx = (losses4[0] - losses4[1]) / (2 * coords_step)  # (r+1,c) - (r-1,c)
        dy = (losses4[2] - losses4[3]) / (2 * coords_step)  # (r,c+1) - (r,c-1)
        dy = (losses4[2] - losses4[3]) / (2 * coords_step)  # (r,c+1) - (r,c-1)
        return torch.stack([dx, dy])  # [2]

    # Step 4: Compute Jacobian per sample w.r.t. params (argnums=0), batched over B samples
    # Result is a PyTree of tensors: for each param, shape is [B, 2, *param.shape]
    per_sample_jac = vmap(jacrev(per_sample_fn, argnums=0), in_dims=(None, 0, 0))(
        params, coords_B42, targs_B41
    )

    # Step 5: Compute Frobenius norm per sample
    # Sum over output dimensions (2) and all parameter dimensions
    frob_sq = torch.zeros(B, device=device, dtype=dtype)
    for name, J in per_sample_jac.items():
        # J has shape [B, 2, *param.shape]
        frob_sq += (J ** 2).reshape(B, -1).sum(dim=1)
    
    return frob_sq.sqrt()  # [B]


# ============================================================================
# Section 3: Gradient Variance Estimation - Main Functions
# ============================================================================

def cell_grad_variance_estimate_with_jacrev(cell_cor_range, graph, inr, device, approx_last_layer=False) -> torch.Tensor:
    """
    Calculate total gradient variance within boxes using jacrev-based approach.
    
    This function estimates the gradient variance for specified regions (boxes)
    in the spatial domain using the Frobenius norm of spatial derivatives.
    
    Args:
        xy_list: Centers of the boxes, shape [N, 2] where first column is x, 
                second column is y (image rows are y, columns are x)
        width_list: Widths of the boxes; if box starts at [0, 0] and width = n,
                   then center is at (n/2, n/2). Note: x_end and y_end are exclusive.
        graph: Graph object containing spatial embeddings and features
        inr: Implicit Neural Representation model
        device: Computation device
        approx_last_layer: If True, only compute gradients for last layer parameters
        
    Returns:
        Tensor containing gradient variance estimates for each box
        
    Note:
        Center calculation: center = ((x_start + x_end) // 2, (y_start + y_end) // 2)
    """
    # Prepare data on CPU
    graph = graph.cpu()
    H = graph.cor.max().item() + 1
    features = graph.feat.view(H, H, 1)
    coords = graph.space_emb.view(H, H, 2)
    
    # Select parameters (all or last layer only)
    params = list(inr.parameters())
    if approx_last_layer:
        params = list(inr.last_layer.parameters())
    
    # Calculate coordinate step size
    coords_range = coords.max() - coords.min()
    coords_step = coords_range / (H - 1)
    
    inr.to(device)
    
    # change xy_list and width_list to cell_cor_range tensor
    # the cell_cor_range is of right shape [N, 4], with (r_start, r_end, c_start, c_end)
    r_indices = ((cell_cor_range[:, 0] + cell_cor_range[:, 1]) // 2).long()  # y coordinates -> row indices
    c_indices = ((cell_cor_range[:, 2] + cell_cor_range[:, 3]) // 2).long()  # x coordinates -> column indices
    batch_size = cell_cor_range.size(0)
    width_tensor = cell_cor_range[:, 1] - cell_cor_range[:, 0]
    # # Convert xy_list to tensor indices
    # r_indices = xy_list[:, 1].long()  # y coordinates -> row indices
    # c_indices = xy_list[:, 0].long()  # x coordinates -> column indices
    # batch_size = len(r_indices)
    
    # Collect 4 neighbors per center point
    neighbor_coords = []
    neighbor_targets = []
    i = 0
    for r, c in zip(r_indices, c_indices):
        neighbors = [
            (r + 1, c),  # Below
            (r - 1, c),  # Above
            (r, c + 1),  # Right
            (r, c - 1),  # Left
        ]
        for rr, cc in neighbors:
            try:
                neighbor_coords.append(coords[rr, cc])
                neighbor_targets.append(features[rr, cc])
            except IndexError:
                print("IndexError for neighbor")
                print(cell_cor_range[i])
        i += 1

    neighbor_coords = torch.stack(neighbor_coords, dim=0)    # [4B, 2]
    neighbor_targets = torch.stack(neighbor_targets, dim=0)  # [4B, 1]
    
    # Move to GPU
    neighbor_coords = neighbor_coords.to(device)
    neighbor_targets = neighbor_targets.to(device)

    # Calculate Frobenius norms using jacrev
    frobenius_norm = frobenius_norm_via_jacrev(
        inr, params, neighbor_coords, neighbor_targets, coords_step
    )

    # Compute variance scaling based on box width
    # assert isinstance(width_list, torch.Tensor), "width_list should be a tensor"
    # width_tensor = width_list
    width = coords_step * width_tensor
    n = width_tensor + 1
    sigma_square = (width**2 / 12 * (n + 1) / (n - 1)).to(device)

    # print(sigma_square)

    grad_total_var_tensor = sigma_square * frobenius_norm**2
    return grad_total_var_tensor

def cell_grad_variance_estimate_with_norm_corrected(cell_cor_range:torch.Tensor, graph, inr, device, probes=500)-> torch.Tensor:
    """
    Partially batched gradient estimation.
    
    For even number of cor window width, the center plus 1/2, make the approximation less accurate. 
    
    Processes multiple samples with some batching optimizations while still
    requiring individual autograd calls for Frobenius norm computation.
    
    Args:
        rc_list: List of (r, c) tuples or tensor of shape [B, 2]
        width_list: List of widths or tensor of shape [B]
        graph: Graph object containing spatial embeddings and features
        inr: Implicit Neural Representation model
        device: Computation device
        probes: Number of random probes for estimation (default: 500)
        
    Returns:
        List of gradient standard deviation estimates for each sample
    """
    
    
    
    graph = graph.to(device)
    features = graph.feat.view(1024, 1024, 1)
    emb = graph.space_emb.view(1024, 1024, 2).requires_grad_(True)
    params = list(inr.parameters())
    
    # Convert inputs to tensors if needed
    # if isinstance(rc_list, list):
    #     rc_tensor = torch.tensor(rc_list, device=device)
    # else:
    #     rc_tensor = rc_list.to(device)
    
    # if isinstance(width_list, list):
    #     width_tensor = torch.tensor(width_list, device=device)
    # else:
    #     width_tensor = width_list.to(device)
    
    batch_size = cell_cor_range.size(0)
    
    # width_list = 
    # rc_list
    
    
    
    # Calculate coordinate step size
    coords_range = emb.max() - emb.min()
    H = graph.cor.max().item()
    coords_step = coords_range / H
    
    # Prepare batch data
    y_x_batch = []
    input_x_batch = []
    target_batch = []
    
    for i in range(batch_size):
        r = (cell_cor_range[i][0] + cell_cor_range[i][1]) // 2
        c = (cell_cor_range[i][2] + cell_cor_range[i][3]) // 2
        
        
        # r, c = rc_tensor[i]
        # r, c = int(r), int(c)
        
        # Compute spatial gradient (y_x) using finite differences
        y_x = torch.tensor([
            (features[r+1, c] - features[r-1, c]) / coords_step / 2,
            (features[r, c+1] - features[r, c-1]) / coords_step / 2
        ], device=device)
        y_x_batch.append(y_x)
        
        # Input coordinates
        input_x = emb[r, c].clone().detach().requires_grad_(True)
        input_x_batch.append(input_x)
        
        # Target value
        target_batch.append(features[r, c])
    
    y_x_batch = torch.stack(y_x_batch)  # [B, 2]
    target_batch = torch.stack(target_batch)  # [B, 1]
    
    # Forward pass and compute losses (individual calls required for autograd)
    grad_var_list = []
    for i in range(batch_size):
        cor_width = cell_cor_range[i][1] - cell_cor_range[i][0]
        n = cor_width + 1
        r = (cell_cor_range[i][0] + cell_cor_range[i][1]) // 2
        c = (cell_cor_range[i][2] + cell_cor_range[i][3]) // 2
        input_x = input_x_batch[i]
        out = inr(input_x)
        loss = loss_function(out, target_batch[i])
        
        # Estimate Frobenius norm for this sample
        fro_norm = estimate_frobenius_norm_corrected(
            loss, out, params, input_x, y_x_batch[i], probes
        )
        
        # Compute variance scaling
        # n = width_tensor[i]
        width = coords_step.double() * cor_width
        # n = 2 * h + 1
        sigma0 = torch.sqrt(width**2 / 12 * (n + 1) / (n - 1))
        print(sigma0**2)
        grad_var= (sigma0 * fro_norm)**2
        grad_var_list.append(grad_var.item())

    return grad_var_list

def grad_variance_ground_truth(cell_cor_range, loss_per_pixel, params, graph):
    """
    r = 16, c = 16, cell_range = 8, window = [12:20, 12:20] include 12~19
    Compute gradient variance within a window of [c - cell_range/2, c + cell_range/2].

    This function calculates the variance of parameter gradients within a spatial
    window by comparing individual pixel gradients to the mean gradient over the window.
    
    Args:
        r: Row index of window center
        c: Column index of window center
        loss_per_pixel: Pre-computed per-pixel loss tensor, shape [H, W, ...]
        params: Model parameters
        cell_range: Window width (total range, not half-width)
        
    Returns:
        Gradient variance scaled by window area (variance * cell_range^2)
        
    Note:
        Variance is computed as: E[||∇θ_i||²] - ||E[∇θ_i]||²
        where i indexes pixels within the window.
    """
    r_lower, r_upper, c_lower, c_upper = cell_cor_range.tolist()

    pix_grad_list = []

    # Per-pixel gradients
    for rr in range(r_lower, r_upper + 1):
        for cc in range(c_lower, c_upper + 1):
            pix_loss = loss_per_pixel[rr, cc]
            pix_grads = torch.autograd.grad(
                pix_loss, params, retain_graph=True, allow_unused=True
            )
            pix_grad_vec = torch.cat(
                [g.reshape(-1) for g in pix_grads if g is not None]
            ).double()
            pix_grad_list.append(pix_grad_vec)

    # Stack: [num_pixels, num_params]
    G = torch.stack(pix_grad_list)

    # Mean gradient
    G_mean = G.mean(dim=0, keepdim=True)

    # Variance: E[||g_i - E[g]||^2]
    var = ((G - G_mean) ** 2).sum(dim=1).mean()

    embbb = graph.space_emb.reshape(1024, 1024, 2).double()
    all_emb = embbb[r_lower:r_upper + 1, c_lower:c_upper + 1]
    emb_var = all_emb.var(dim=(0, 1))


    return var.cpu().item(), emb_var.cpu()


def loss_variance_ground_truth(cell_cor_ranges, graph, inr, device) -> torch.Tensor:
    loss_var_list = []
    for cell_cor_range in cell_cor_ranges:
        r_lower, r_upper, c_lower, c_upper = cell_cor_range.tolist()
        graph = graph.to(device)
        H = graph.cor.max().item() + 1
        features = graph.feat.view(H, H, 1)
        coords = graph.space_emb.detach().view(H, H, 2)
        
        features = features[r_lower:r_upper + 1, c_lower:c_upper + 1]
        coords = coords[r_lower:r_upper + 1, c_lower:c_upper + 1]
        
        reconfeatures = inr(coords)
        per_pixel_loss = loss_function(reconfeatures, features)
        
        lvar = per_pixel_loss.var().item()
        loss_var_list.append(lvar)
    loss_var = torch.tensor(loss_var_list).to(device)
    return loss_var


# ============================================================================
# Section 4: evaluation function define
# ============================================================================

def total_variance_estimation(cell_cor_range, graph, inr, device):
    H = graph.cor.max().item() + 1
    cell_area = (cell_cor_range[:,1] - cell_cor_range[:,0]) * (cell_cor_range[:,3] - cell_cor_range[:,2])
    grad_var_tensor = cell_grad_variance_estimate_with_jacrev(
        cell_cor_range, graph, inr, device
    )
    return grad_var_tensor * cell_area

def grad_estimation(xy_list, width_list, graph, inr, device, probes=500):
    """
    Single-sample gradient estimation using Hutchinson's estimator.
    
    Estimates gradient variance for each point individually using random probes
    and the corrected Frobenius norm estimator.
    
    Args:
        xy_list: List of (x, y) coordinate pairs or tensor of shape [N, 2]
        width_list: List or tensor of box widths for each point
        graph: Graph object containing spatial embeddings and features
        inr: Implicit Neural Representation model
        device: Computation device
        probes: Number of random probes for Frobenius norm estimation (default: 500)
        
    Returns:
        List of gradient variance estimates (variance * box_area)
        
    Note:
        This is a single-sample (non-batched) implementation. For batched processing,
        use grad_estimation_batch or grad_estimation_fully_batched.
        Box size calculation: actual box is (n+1)×(n+1) with center at (n/2, n/2)
    """
    graph = graph.to(device)
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2).requires_grad_(True)
    params = list(inr.parameters())
    
    # Calculate coordinate step size
    coords_range = coords.max() - coords.min()
    H = coords.size(0)
    coords_step = coords_range / (H - 1)
    grad_std_list = []
    
    # Process each point individually
    for (c, r), n in zip(xy_list, width_list):
        # Compute spatial gradient using finite differences
        y_x = torch.tensor(
            [(features[r+1, c] - features[r-1, c]) / coords_step / 2, 
             (features[r, c+1] - features[r, c-1]) / coords_step / 2]
        ).to(device)

        # Prepare input and compute loss
        input_x = coords[r, c].clone().detach().to(device).requires_grad_() 
        out = inr(input_x)
        loss = loss_function(out, features[r, c])
        
        # Estimate Frobenius norm
        fro_norm = estimate_frobenius_norm_corrected(
            loss, out, params, input_x, y_x, probes
        )
        
        # Compute variance scaling
        width = coords_step * n
        # Note: n = 2*h + 1, so box is (n+1)×(n+1) with center at (n/2, n/2)
        sigma0 = torch.sqrt(width**2 / 12 * (n + 2) / n)
        
        # Store variance * area
        grad_std = sigma0 * fro_norm
        grad_std_list.append(grad_std.item()**2 * n**2)
    
    return grad_std_list


# ============================================================================
# Section 4: Batched Gradient Variance Estimation
# ============================================================================


def cell_width_sigma(cor_width_tensor, coords_step):
    """
    Compute sigma values for given cell widths.
    
    Args:
        cor_width_tensor: Tensor of cell widths
        coords_step: Step size in coordinate space (scalar float or 0-dim tensor)   
    Returns:
        Tensor of sigma values corresponding to each cell width
    """
    width = coords_step * cor_width_tensor
    sigma_square = width**2 / 12 * (cor_width_tensor + 1) / (cor_width_tensor - 1)
    sigma = torch.sqrt(sigma_square)
    return sigma
    

def grad_estimation_fully_batched(rc_tensor, cor_width_tensor, graph, inr, device, probes=500):
    """
    Fully batched gradient estimation with maximum parallelization.
    
    Processes all samples simultaneously where possible, though autograd
    constraints still require individual Frobenius norm computations.
    
    Args:
        rc_tensor: Tensor of row/column coordinates, shape [B, 2]
        width_tensor: Tensor of box widths, shape [B]
        graph: Graph object containing spatial embeddings and features
        inr: Implicit Neural Representation model
        device: Computation device
        probes: Number of random probes for estimation (default: 500)
        
    Returns:
        List of gradient standard deviation estimates for each sample
    """
    graph = graph.to(device)
    rc_tensor = rc_tensor.to(device)
    cor_width_tensor = cor_width_tensor.to(device)
    
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2)
    
    batch_size = rc_tensor.size(0)
    coords_range = coords.max() - coords.min()
    H = coords.size(0)
    coords_step = coords_range / H
    
    # Extract row/column indices
    r_indices = rc_tensor[:, 0].long()
    c_indices = rc_tensor[:, 1].long()
    
    # Compute spatial gradients (y_x) for all samples using vectorized operations
    y_x_r = (features[r_indices + 1, c_indices] - features[r_indices - 1, c_indices]) / coords_step / 2
    y_x_c = (features[r_indices, c_indices + 1] - features[r_indices, c_indices - 1]) / coords_step / 2
    y_x_batch = torch.stack([y_x_r.squeeze(), y_x_c.squeeze()], dim=1)  # [B, 2]
    
    # Get input coordinates and targets
    input_coords_batch = coords[r_indices, c_indices]  # [B, 2]
    target_batch = features[r_indices, c_indices]  # [B, 1]
    
    # Compute Frobenius norms (still requires individual autograd calls)
    grad_std_list = []
    for i in range(batch_size):
        input_x = input_coords_batch[i].clone().detach().requires_grad_(True)
        out = inr(input_x)
        loss = loss_function(out, target_batch[i])
        
        fro_norm = estimate_frobenius_norm_corrected(
            loss, out, list(inr.parameters()), input_x, y_x_batch[i], probes
        )
        
        # Compute variance scaling
        n = cor_width_tensor[i]
        width = coords_step * (n-1)
        # n = 2 * h + 1
        sigma0 = torch.sqrt(width**2 / 12 * (n + 1) / (n - 1))
        grad_std = sigma0 * fro_norm
        grad_std_list.append(grad_std.item())
    
    return grad_std_list


# ============================================================================
# Section 5: Window-Based Gradient Variance Estimation
# ============================================================================

def grad_win_batched(rc_tensor, width_tensor, graph, inr, device):
    """
    Batched window-based gradient variance estimation.
    
    Computes gradient variance by comparing per-pixel gradients within a window
    to the average gradient over the entire window.
    
    Args:
        rc_tensor: Tensor of center coordinates, shape [B, 2]
        width_tensor: Tensor of window widths, shape [B]
        graph: Graph object containing spatial embeddings and features
        inr: Implicit Neural Representation model
        device: Computation device
        
    Returns:
        List of gradient variances for each window
    """

    graph = graph.to(device)
    rc_tensor = rc_tensor.to(device)
    width_tensor = width_tensor.to(device)
    
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2)
    
    # Reconstruct all features and compute per-pixel loss
    features_recon = inr(coords)
    loss_per_pixel = loss_function(features_recon, features)
    params = list(inr.parameters())
    
    batch_size = rc_tensor.size(0)
    
    # Extract center coordinates
    r_indices = rc_tensor[:, 0].long()
    c_indices = rc_tensor[:, 1].long()
    
    # Compute variance for each window
    grad_std_list = []
    for i in range(batch_size):
        r = r_indices[i]
        c = c_indices[i]
        width = width_tensor[i]
        grad_std = grad_variance_ground_truth(r, c, loss_per_pixel, params, width)
        grad_std_list.append(grad_std)
    
    return grad_std_list




# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pass