import torch

def estimate_frobenius_norm_corrected(
    loss,
    out, 
    params,
    input_x,
    y_x,               
    n_probes: int = 10,
):
    """
    Frobenius-norm estimator for H' = H_orig - 2 y_x f_theta^T
    using Hutchinson / random-probe trick.
    
    this is for squared loss

    Args:
        loss:       scalar tensor (already computed)
        params:     iterable of model parameters (θ)
        input_x:    input tensor with requires_grad = True
        y_x:        ∂y/∂x (shape = input_x.shape flattened)           
        f_theta:    ∂y/∂θ (concatenated over params; shape = P)       
        n_probes:   number of random probes
    Returns:
        scalar tensor ≈ ‖H'‖_F
    """
    # ∂L/∂θ  (shape = P)
    grad_theta = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    flat_grad = torch.cat([g.reshape(-1) for g in grad_theta if g is not None])  # P

    P = flat_grad.numel()
    V = torch.randn(n_probes, P, device=flat_grad.device)                       # [N, P]

    # <f_theta , v_i> for every probe  –  shape [N]
    grad1 = torch.autograd.grad(
    out, params, retain_graph=True, allow_unused=True, create_graph=True
    )
    f_theta = torch.cat([
        g.reshape(-1) for g in grad1 if g is not None
    ])
    s = V.matmul(f_theta)                        

    # <grad_theta , v_i>  –  shape [N]
    dots = V.matmul(flat_grad)                        

    grads_x = []
    for i in range(n_probes):
        # H_orig v_i
        g_x = torch.autograd.grad(
            dots[i], input_x, retain_graph=True, allow_unused=True
        )[0]
        if g_x is None:
            g_x = torch.zeros_like(input_x)

        hv_corr = g_x - 2.0 * s[i] * y_x

        grads_x.append(hv_corr.reshape(-1))           # flatten to 1-D

    stacked = torch.stack(grads_x)                    # [N, D]
    frob_sq = (stacked ** 2).sum() / n_probes         # unbiased estimator of ‖H'‖_F²
    return frob_sq.sqrt()

# coords range in [-1, 1]

def grad_estimation(rc_list, width_list, graph, inr, device, probes=500):
    # probes: number of random probes for frobenius norm estimation
    graph = graph.to(device)
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2).requires_grad_(True)
    params = list(inr.parameters())
    
    # coords shape is (H, H, 2)
    # features shape is (H, H, 1)
    coords_range = coords.max() - coords.min()
    H = coords.size(0)
    coords_step = coords_range / H
    grad_std_list = []
    
    #TODO: try make the whole function handle the batch of points
    for (r, c), h in zip(rc_list, width_list):
        y_x = torch.tensor(
            [(features[r+1, c] - features[r-1, c])/coords_step/2, 
             (features[r, c+1] - features[r, c-1])/coords_step/2]
        ).to(device)

        input_x = torch.tensor(
            coords[r, c],
            requires_grad=True,
            device=device
        ) 
        out = inr(input_x)
        loss = loss_function(out, features[r, c])
        fro_norm = estimate_frobenius_norm_corrected(loss, out, params, input_x, y_x, probes)
        
        width = coords_step *(2*h)
        n = 2*h + 1
        sigma0 = torch.sqrt( width**2/12*(n+1)/(n-1))
        
        grad_std = sigma0*fro_norm
        grad_std_list.append(grad_std.item())
    return grad_std_list

def estimate_frobenius_norm_corrected_batch(
    loss_batch, out_batch, params, input_x_batch, y_x_batch, n_probes: int = 10
):
    """
    Batched Frobenius-norm estimator for H' = H_orig - 2 y_x f_theta^T
    
    Args:
        loss_batch: batch of scalar losses [B]
        out_batch: batch of model outputs [B, ...]
        params: iterable of model parameters (θ)
        input_x_batch: batch of input tensors [B, ...] with requires_grad = True
        y_x_batch: batch of ∂y/∂x [B, D] where D is flattened input dimension
        n_probes: number of random probes
        
    Returns:
        batch of frobenius norms [B]
    """
    batch_size = loss_batch.size(0)
    
    # Compute gradients for the entire batch
    grad_theta_batch = []
    f_theta_batch = []
    
    for i in range(batch_size):
        # ∂L/∂θ for sample i
        grad_theta = torch.autograd.grad(
            loss_batch[i], params, create_graph=True, retain_graph=True, allow_unused=True
        )
        flat_grad = torch.cat([g.reshape(-1) for g in grad_theta if g is not None])
        grad_theta_batch.append(flat_grad)
        
        # f_theta for sample i
        grad1 = torch.autograd.grad(
            out_batch[i], params, retain_graph=True, allow_unused=True, create_graph=True
        )
        f_theta = torch.cat([g.reshape(-1) for g in grad1 if g is not None])
        f_theta_batch.append(f_theta)
    
    # Stack gradients
    grad_theta_stacked = torch.stack(grad_theta_batch)  # [B, P]
    f_theta_stacked = torch.stack(f_theta_batch)  # [B, P]
    
    P = grad_theta_stacked.size(1)
    V = torch.randn(n_probes, P, device=grad_theta_stacked.device)  # [N, P]
    
    # Compute dot products for all samples and probes
    s_batch = torch.matmul(f_theta_stacked, V.T)  # [B, N]
    dots_batch = torch.matmul(grad_theta_stacked, V.T)  # [B, N]
    
    frob_norms = []
    
    for i in range(batch_size):
        grads_x = []
        for j in range(n_probes):
            # H_orig v_j for sample i
            g_x = torch.autograd.grad(
                dots_batch[i, j], input_x_batch[i], retain_graph=True, allow_unused=True
            )[0]
            if g_x is None:
                g_x = torch.zeros_like(input_x_batch[i])
            
            hv_corr = g_x - 2.0 * s_batch[i, j] * y_x_batch[i]
            grads_x.append(hv_corr.reshape(-1))
        
        stacked = torch.stack(grads_x)  # [N, D]
        frob_sq = (stacked ** 2).sum() / n_probes
        frob_norms.append(frob_sq.sqrt())
    
    return torch.stack(frob_norms)


def grad_estimation_batch(rc_list, width_list, graph, inr, device, probes=500):
    """
    Batched version of gradient estimation
    
    Args:
        rc_list: list of (r, c) tuples or tensor of shape [B, 2]
        width_list: list of widths or tensor of shape [B]
        graph: graph object
        inr: implicit neural representation model
        device: computation device
        probes: number of random probes
        
    Returns:
        list of gradient standard deviations
    """
    graph = graph.to(device)
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2).requires_grad_(True)
    params = list(inr.parameters())
    
    # Convert to tensors if needed
    if isinstance(rc_list, list):
        rc_tensor = torch.tensor(rc_list, device=device)
    else:
        rc_tensor = rc_list.to(device)
    
    if isinstance(width_list, list):
        width_tensor = torch.tensor(width_list, device=device)
    else:
        width_tensor = width_list.to(device)
    
    batch_size = rc_tensor.size(0)
    
    coords_range = coords.max() - coords.min()
    H = coords.size(0)
    coords_step = coords_range / H
    
    # Prepare batch data
    y_x_batch = []
    input_x_batch = []
    target_batch = []
    
    for i in range(batch_size):
        r, c = rc_tensor[i]
        r, c = int(r), int(c)
        
        # Compute y_x (gradient of features)
        y_x = torch.tensor([
            (features[r+1, c] - features[r-1, c]) / coords_step / 2,
            (features[r, c+1] - features[r, c-1]) / coords_step / 2
        ], device=device)
        y_x_batch.append(y_x)
        
        # Input coordinates
        input_x = coords[r, c].clone().detach().requires_grad_(True)
        input_x_batch.append(input_x)
        
        # Target value
        target_batch.append(features[r, c])
    
    y_x_batch = torch.stack(y_x_batch)  # [B, 2]
    target_batch = torch.stack(target_batch)  # [B, 1]
    
    # Forward pass for entire batch
    out_batch = []
    for input_x in input_x_batch:
        out = inr(input_x)
        out_batch.append(out)
    out_batch = torch.stack(out_batch)  # [B, ...]
    
    # Compute losses
    loss_batch = []
    for i in range(batch_size):
        loss = loss_function(out_batch[i], target_batch[i])
        loss_batch.append(loss)
    loss_batch = torch.stack(loss_batch)  # [B]
    
    # Estimate Frobenius norms
    fro_norms = estimate_frobenius_norm_corrected_batch(
        loss_batch, out_batch, params, input_x_batch, y_x_batch, probes
    )
    
    # Compute gradient standard deviations
    grad_std_list = []
    for i in range(batch_size):
        h = width_tensor[i]
        width = coords_step * (2 * h)
        n = 2 * h + 1
        sigma0 = torch.sqrt(width**2 / 12 * (n + 1) / (n - 1))
        grad_std = sigma0 * fro_norms[i]
        grad_std_list.append(grad_std.item())
    
    return grad_std_list


# Alternative more efficient version that processes everything in true batch mode
def grad_estimation_fully_batched(rc_tensor, width_tensor, graph, inr, device, probes=500):
    """
    Fully batched version - processes all samples simultaneously where possible
    
    Args:
        rc_tensor: tensor of coordinates [B, 2]
        width_tensor: tensor of widths [B]
        graph: graph object
        inr: implicit neural representation model
        device: computation device
        probes: number of random probes
        
    Returns:
        tensor of gradient standard deviations [B]
    """
    graph = graph.to(device)
    rc_tensor = rc_tensor.to(device)
    width_tensor = width_tensor.to(device)
    
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2)
    
    batch_size = rc_tensor.size(0)
    coords_range = coords.max() - coords.min()
    H = coords.size(0)
    coords_step = coords_range / H
    
    # Extract coordinates and compute gradients
    r_indices = rc_tensor[:, 0].long()
    c_indices = rc_tensor[:, 1].long()
    
    # Compute y_x for all samples
    y_x_r = (features[r_indices + 1, c_indices] - features[r_indices - 1, c_indices]) / coords_step / 2
    y_x_c = (features[r_indices, c_indices + 1] - features[r_indices, c_indices - 1]) / coords_step / 2
    y_x_batch = torch.stack([y_x_r.squeeze(), y_x_c.squeeze()], dim=1)  # [B, 2]
    
    # Get input coordinates and targets
    input_coords_batch = coords[r_indices, c_indices]  # [B, 2]
    target_batch = features[r_indices, c_indices]  # [B, 1]
    
    # For the Frobenius norm computation, we still need individual autograd calls
    # This is a limitation of the current approach due to how autograd works
    grad_std_list = []
    
    for i in range(batch_size):
        input_x = input_coords_batch[i].clone().detach().requires_grad_(True)
        out = inr(input_x)
        loss = loss_function(out, target_batch[i])
        
        fro_norm = estimate_frobenius_norm_corrected(
            loss, out, list(inr.parameters()), input_x, y_x_batch[i], probes
        )
        
        h = width_tensor[i]
        width = coords_step * (2 * h)
        n = 2 * h + 1
        sigma0 = torch.sqrt(width**2 / 12 * (n + 1) / (n - 1))
        grad_std = sigma0 * fro_norm
        grad_std_list.append(grad_std.item())
    
    return grad_std_list

def grad_win_batched(rc_tensor, width_tensor, graph, inr, device):
    graph = graph.to(device)
    rc_tensor = rc_tensor.to(device)
    width_tensor = width_tensor.to(device)
    
    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.view(1024, 1024, 2)
    
    features_recon = inr(coords)
    loss_per_pixel = loss_function(features_recon, features)
    params = list(inr.parameters())
    
    
    batch_size = rc_tensor.size(0)
    # coords_range = coords.max() - coords.min()
    # H = coords.size(0)
    # coords_step = coords_range / H
    
    # # Get input coordinates and targets
    # input_coords_batch = coords[r_indices, c_indices]  # [B, 2]
    # target_batch = features[r_indices, c_indices]  # [B, 1]
    
    # Extract coordinates and compute gradients
    r_indices = rc_tensor[:, 0].long()
    c_indices = rc_tensor[:, 1].long()
    grad_std_list = []
    for i in range(batch_size):
        r = r_indices[i]
        c = c_indices[i]
        width = width_tensor[i]//2
        grad_std = grad_win_variance(r, c, loss_per_pixel, params, width)
        grad_std_list.append(grad_std)
    return grad_std_list

def grad_win_variance(r, c, loss_per_pixel, params, width):
    pix_grad_list = []
    loss_batch = loss_per_pixel[r-width:r+width, c-width:c+width].mean()
    batch_grads = torch.autograd.grad(loss_batch, params,retain_graph=True, allow_unused=True,)
    batch_grad_vec = torch.cat([g.reshape(-1) for g in batch_grads if g is not None])
    grad_sq_batch = batch_grad_vec.pow(2).sum()
    window_grad_sqs = []
    for dr in range(-width, width, 1):
        for dc in range(-width, width, 1):
            rr, cc = r + dr, c + dc
            # loss = inr(coords[rr, cc])
            pix_loss = loss_per_pixel[rr, cc]
            
            pix_grads = torch.autograd.grad(
                pix_loss, params, retain_graph=True, allow_unused=True
            )
            pix_grad_vec = torch.cat([g.reshape(-1) for g in pix_grads if g is not None])
            pix_grad_list.append(pix_grad_vec)
            # total_diff += (pix_grad_vec - batch_grad_vec).pow(2).sum().cpu().item()
            window_grad_sqs.append(pix_grad_vec.pow(2).sum())
            
    avg_grad_sq_win = torch.stack(window_grad_sqs).mean()
    
    return (avg_grad_sq_win - grad_sq_batch).sqrt().cpu().item()

def loss_function(features_recon, features):
    loss = features_recon
    loss = ((features_recon - features)**2)
    return loss

if __name__ == "__main__":

    pass
    grad_norm_list = []
    fro_list = []
    r, c = 48, 23
    grad_list = []
    grad_sq_diff_sqrt_list = []
    width = 10
    graph = graph.to(device)


    features = graph.feat.view(1024, 1024, 1)
    coords = graph.space_emb.detach().view(1024, 1024, 2).requires_grad_(True)
    s_step = 2/1024

    y_x = torch.tensor([(features[r+1, c] - features[r-1, c])/s_step, (features[r, c+1] - features[r, c-1])/s_step]).to(device)/2

    # for i in range(0, 1000, 20):
    for i in [100]:
        
        inr = load_inr(i, model_dir)
        inr.to(device)
        features_recon = inr(coords)
        loss_per_pixel = loss_function(features_recon, features)
        params = list(inr.parameters())
        # total_grad = torch.autograd.grad(loss_per_pixel.mean(), params, retain_graph=True , allow_unused = True)
        # total_grad_vec = torch.cat([g.reshape(-1) for g in total_grad if g is not None])
        # grad_list.append(total_grad_vec.norm().detach().cpu().item())
        
        grad_sq_dif_sqrt, batch_norm, batch_vector, pix_list= grad_win_variance(r, c, loss_per_pixel, params, width)
        grad_sq_diff_sqrt_list.append(grad_sq_dif_sqrt)
        
        grad_list.append(batch_norm)
        
        input_x = torch.tensor(
            coords[r, c],
            requires_grad=True,
            device=device
        ) 
        out = inr(input_x)
        loss = loss_function(out, features[r, c])
        
        
        fro_norm = estimate_frobenius_norm_corrected(loss, out, params, input_x, y_x, 500)
        # fro_norm = estimate_frobenius_norm_jacobian_param_input(loss_per_pixel[r, c], params, coords[r, c], n_probes=500)
        
        # point_grad = torch.autograd.grad(
        #     loss, params, retain_graph=True, allow_unused=True
        # )
        # grads_all = [g.view(-1) for g in point_grad if g is not None]
        # grad_vec = torch.cat(grads_all)
        # grad_list.append(grad_vec)
        # grad_norm = grad_vec.norm()
        # grad_norm_list.append(grad_norm.item())
        fro_list.append(fro_norm.item())
        
    coords = graph.space_emb.view(1024, 1024, 2)
    cor = torch.zeros((2, 2))
    i = 0
    for dr in range(-width, width+1, 1):
        for dc in range(-width, width+1, 1):
            rr, cc = r + dr, c + dc
            dif  =( coords[rr, cc] - coords[r, c]).view(-1, 1).cpu()
            cor += dif @ dif.T
            i += 1
    cor = cor / i

    print(grad_sq_diff_sqrt_list[0])