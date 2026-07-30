import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from torch_geometric.data import Data

from components.laplacian import compute_laplacian_loss as compute_laplacian
from components.nmt import mt_scheduler_factory
from components.transform import Transform
from train_utility_sampling.taylor_estimation import (
    build_3d_index_grid,
    cell_grad_variance_estimate_with_jacrev,
    cell_loss_variance_batched,
    cell_loss_variance_batched_3d,
    cell_loss_variance_estimate_with_random_sampling,
    cell_loss_variance_estimate_with_random_sampling_3d,
    cell_sqrt_loss_variance_estimate_with_random_sampling,
    cell_sqrt_loss_variance_estimate_with_random_sampling_3d,
    loss_variance_ground_truth,
)
from util.misc import fix_seed
from utils.data.unstructure_dataset import get_graph_t_idx
from utils.octree import HierarchicalVoxelGrid
from utils.quadtree import HierarchicalImageGrid


def normalize_and_clip_cell_weights(per_cell_weight: torch.Tensor, weight_clip_ratio: float = 10.0) -> torch.Tensor:
    """Clamp invalid/extreme cell weights, then normalize by mean for stability."""
    w = per_cell_weight.to(torch.float32)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = w.clamp_min(0.0)

    if weight_clip_ratio > 0:
        positive = w[w > 0]
        if positive.numel() > 0:
            med = positive.median()
            if torch.isfinite(med) and med.item() > 0:
                cap = med * weight_clip_ratio
                w = torch.clamp(w, max=cap)

    mean_w = w.mean()
    if torch.isfinite(mean_w) and mean_w.item() > 0:
        return w / mean_w
    return torch.ones_like(w)


class InrSamplerWrapper:
    """
    Wrapper class for coordinate sampling algorithms in INR training.

    Args:
        model (torch.nn.Module): The INR model to be trained.
        iters (int): Number of training iterations.
        n_clusters_2d_start (int): Starting number of 2D clusters. Defaults to 100.
        n_clusters_2d_end (int): Ending number of 2D clusters. Defaults to 100.
        epochs (int): Total number of training epochs. Defaults to 5000.
        device (str): Device to run sampling on. Defaults to "cuda:0".
        sample_type (str): Type of sampling strategy ("random", "NMT", "3d_cluster", "2d_grid_linear", "2d_grid_linear_weighted"). Defaults to "random".
        use_weight_function (bool): Whether to apply per-cell loss-based weights when `sample_type` is `2d_grid_linear_weighted`. Defaults to True.
        sample_rate (float): Fraction of nodes to sample. Defaults to 0.5.
        save_samples_path (Path): Directory to save sampled images. Defaults to Path("logs/sampling").
        save_interval (int): Interval for saving samples. Defaults to 100.
        image_width (int): Width of the image grid. Defaults to 512.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        iters: int,
        n_clusters_2d_start: int = 100,
        n_clusters_2d_end: int = 100,
        epochs: int = 5000,
        device: str = "cuda:0",
        sample_type: str = "random",
        use_weight_function: bool = True,
        sample_rate: float = 0.5,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        image_width: int = 512,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.sample_type = sample_type
        self.use_weight_function = use_weight_function
        self.sample_rate = sample_rate
        self.iters = iters

        if sample_type in ("2d_grid_linear", "2d_grid_linear_weighted"):
            self.n_clusters_2d_start = n_clusters_2d_start
            self.n_clusters_2d_end = n_clusters_2d_end
            self.epochs = epochs

        self.save_interval = save_interval
        self.save_samples_path = save_samples_path
        self.image_width = image_width
        # Optional visualization caches populated by specific samplers.
        self.cached_linear_bounds: Optional[torch.Tensor] = None
        self.cached_linear_n_bins: Optional[int] = None


    def _get_T(self, graph: Data) -> int:
        """Extract total number of time frames from graph.T."""
        if hasattr(graph.T, "sum"):
            return graph.T.sum()
        return graph.T

    def _sample_random(self, graph: Data, T: int) -> torch.Tensor:
        """Randomly sample nodes from each time frame."""
        sampled_indices = []
        for t in range(T):
            indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
            n_t = indices_t.numel()
            n_samples = max(int(n_t * self.sample_rate), 1)
            perm = torch.randperm(n_t, device=self.device)[:n_samples]
            sampled_indices.append(indices_t[perm])
        return torch.cat(sampled_indices, dim=0)

    def _sample_nmt(self, graph: Data, modulations: torch.Tensor, T: int) -> torch.Tensor:
        """Sample using Non-parametric Machine Teaching (NMT) - select high-error nodes."""
        with torch.no_grad():
            graph = graph.to(self.device)
            modulations = modulations.to(self.device)
            preds = self.model.modulated_forward(graph.space_emb, modulations[graph.time.cpu()])
            dif = torch.sum(torch.abs(graph.feat - preds), 1)

            sampled_indices = []
            for t in range(T):
                indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
                n_t = indices_t.numel()
                _, top_idx = torch.topk(dif[indices_t], int(self.sample_rate * n_t))
                sampled_indices.append(indices_t[top_idx])

            return torch.cat(sampled_indices, dim=0)

    def _sample_3d_cluster(self, graph: Data, modulations: torch.Tensor, T: int) -> torch.Tensor:
        """Sample using 3D clustering - cluster-based sampling with error-based selection."""
        W, H = graph.cor.max(axis=0)[0] + 1
        n_samples = max(1, int(W * H * T * self.sample_rate))
        num_per_cluster = max(1, math.ceil(n_samples / len(graph.cluster_set[0])))

        # Get rough sample from clusters
        rough_idx = sample_random_node_indices_per_cluster(
            graph, cluster_dim='3d', num_per_cluster=num_per_cluster
        )

        # Compute errors on rough sample
        times = graph.time[rough_idx]
        space_emb = graph.space_emb[rough_idx].to(self.device)
        feats = graph.feat[rough_idx].to(self.device)
        mod = modulations.to(self.device)

        with torch.no_grad():
            preds = self.model.modulated_forward(space_emb, mod[times.cpu()])
            diffs = torch.sum((feats - preds).abs(), dim=1)

        # Select top-k from each time frame
        sampled_per_t = []
        for t in range(T):
            local_mask = (times == t).nonzero(as_tuple=True)[0]
            if local_mask.numel() == 0:
                continue

            count = min(int(W * H * self.sample_rate), local_mask.numel())
            _, topk_local = torch.topk(diffs[local_mask], count)
            selected_global = rough_idx[local_mask[topk_local.cpu()]]
            sampled_per_t.append(selected_global)

        return torch.cat(sampled_per_t, dim=0)

    def sample(
        self,
        outer_step: int,
        inner_step: int,
        graph: Data,
        modulations: torch.Tensor = None,
        save_image: bool = False
    ) -> Data:
        """
        Perform coordinate sampling on the graph.

        Args:
            outer_step (int): Current meta outer step (for saving images).
            inner_step (int): Current sampling iteration (for saving images).
            graph (Data): Input graph with coordinates, features, and time information.
            modulations (torch.Tensor, optional): Modulation vectors for NMT/3d_cluster sampling.
            save_image (bool): Whether to save visualization of sampled points.

        Returns:
            Data: Sampled graph with subset of nodes.
        """
        T = self._get_T(graph)

        # Select sampling method
        if self.sample_type == "random":
            sampled_idx = self._sample_random(graph, T)
            dif = None
        elif self.sample_type == "NMT":
            assert modulations is not None, "Modulations required for NMT sampling."
            sampled_idx = self._sample_nmt(graph, modulations, T)
            dif = None  # Could extract from _sample_nmt if needed
        elif self.sample_type == "3d_cluster":
            assert modulations is not None, "Modulations required for 3d_cluster sampling."
            sampled_idx = self._sample_3d_cluster(graph, modulations, T)
            dif = None
        else:
            raise NotImplementedError(f"Sampling type {self.sample_type} is not implemented.")

        # Create sampled graph
        sampled_graph = Data(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
            T=graph.T,
            latent_vector=graph.latent_vector
        )

        if save_image:
            self.save_image_path = os.path.join(
                self.save_samples_path,
                f"{self.sample_type}_o{outer_step}_i{inner_step}"
            )
            self._save_sample_images(graph, sampled_graph, dif=dif)

        return sampled_graph

    def _save_sample_images(self, graph: Data, sampled_graph: Data, dif: torch.Tensor = None):
        """
        Save visualization images for each time frame showing sampled positions.
        
        Args:
            graph: Full graph with all nodes.
            sampled_graph: Graph containing only sampled nodes.
            dif: Optional difference/error map for visualization.
        """
        os.makedirs(self.save_image_path, exist_ok=True)

        T_show = self._get_T(graph)
        W = int(graph.cor[:, 0].max() + 1)
        H = int(graph.cor[:, 1].max() + 1)

        for t in range(T_show):
            # Get data for current time frame
            frame_mask = (graph.time == t)
            values = graph.feat[frame_mask].cpu().numpy()

            sampled_frame_mask = (sampled_graph.time == t)
            sampled_coords = sampled_graph.cor[sampled_frame_mask].cpu().numpy()

            # Plot field with sampled points
            plt.figure()
            field = values.reshape(H, W)
            plt.imshow(field, cmap='viridis', origin='lower')
            plt.axis('off')
            plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0], c='red', s=0.015625)
            plt.title(f'Time Frame {t}')

            filename = Path(self.save_image_path) / f'frame_{t:03d}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            # Optionally plot difference map
            if dif is not None:
                plt.figure()
                dif_frame = dif[frame_mask].cpu().numpy().reshape(H, W)
                plt.imshow(dif_frame, cmap='hot', alpha=0.5, origin='lower')
                plt.axis('off')
                plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0], c='red', s=10)
                plt.title(f'Time Frame {t}')

                filename = Path(self.save_image_path) / f'frame_{t:03d}_dif.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()


def _extract_T_info(T_raw):
    """Extract time and graph information from graph.T."""
    if isinstance(T_raw, int):
        return T_raw, T_raw, 1

    if torch.is_tensor(T_raw):
        if T_raw.dim() == 0:
            T_val = T_raw.item()
            return T_val, T_val, 1
        elif T_raw.dim() == 1:
            if not torch.all(T_raw == T_raw[0]):
                raise ValueError("All entries of graph.T must be equal when it's a 1-D tensor")
            return T_raw.sum().item(), T_raw[0].item(), T_raw.size(0)

    raise TypeError(f"Unexpected type for graph.T: {type(T_raw)}")


def sample_random_node_indices_per_cluster(
    graph: Data,
    cluster_dim: str = '2d',
    num_per_cluster: int = 1,
) -> torch.Tensor:
    """Sample random node indices from each cluster in a graph."""
    T_total, T, graph_num = _extract_T_info(graph.T)
    W, H = graph.cor.max(axis=0)[0] + 1
    nodes_per_graph = T * W * H
    device = torch.device("cuda")

    def extract_samples(cluster_dict, graph_idx):
        """Extract samples from a cluster dictionary."""
        samples = []
        offset = graph_idx * nodes_per_graph
        for idx_tensor in cluster_dict.values():
            n = idx_tensor.numel()
            if n < num_per_cluster:
                raise AssertionError(
                    f"Cluster has {n} nodes, but requested {num_per_cluster} samples. "
                    "Reduce sampling rate or number of clusters."
                )
            perm = torch.randperm(n)[:num_per_cluster]
            chosen = idx_tensor[perm].to(device) + offset
            samples.append(chosen)
        return samples

    all_samples = []
    if cluster_dim == '2d':
        for t in range(T_total):
            frame_cluster_dict = graph.cluster_set[t]
            graph_idx = t // T
            all_samples.extend(extract_samples(frame_cluster_dict, graph_idx))
    elif cluster_dim == '3d':
        for graph_idx in range(graph_num):
            cluster_dict = (graph.cluster_set[graph_idx] if isinstance(graph.cluster_set, list)
                           else graph.cluster_set)
            all_samples.extend(extract_samples(cluster_dict, graph_idx))
    else:
        raise ValueError(f"cluster_dim must be '2d' or '3d', got '{cluster_dim}'")

    return torch.cat(all_samples)


def graph_3d_cluster(graph: Data, n_segments: int, compactness: float, cluster_type: str = 'slic'):
    """Apply 3D clustering to graph data."""
    T = graph.T.sum()
    W, H = graph.cor.max(axis=0)[0] + 1
    graph.cluster_set = [defaultdict(dict)]
    vol = graph.feat.reshape(T, W, H)

    if cluster_type == 'slic':
        segments = slic(vol, n_segments=n_segments, compactness=compactness,
                       start_label=0, channel_axis=None)
    else:
        raise NotImplementedError(f"Unknown cluster_type: {cluster_type}")

    segments_flat = torch.tensor(segments).reshape(-1)
    for i in range(segments.max() + 1):
        mask = segments_flat == i
        graph.cluster_set[0][i] = torch.where(mask)[0]
    graph.segments = segments


def graph_2d_cluster_old(graph, n_segments, compactness, cluster_dim='2d', num_per_cluster: int=1) -> None:
    """OLD VERSION - TO BE REMOVED
    Sample random node indices from each cluster in a graph.
    
    """

    T_raw = graph.T

    if isinstance(T_raw, int):
        # pure Python int → single graph
        T_total = T_raw
        T = T_raw
        graph_num = 1

    elif torch.is_tensor(T_raw):
        if T_raw.dim() == 0:
            # 0-D tensor → single graph
            T_total = T_raw.item()
            T = T_total
            graph_num = 1

        elif T_raw.dim() == 1:
            # 1-D tensor → potentially multiple graphs
            # ensure every entry is the same
            if not torch.all(T_raw == T_raw[0]):
                raise ValueError("All entries of graph.T must be equal when it’s a 1-D tensor")
            T_total = T_raw.sum().item()      # total over all graphs
            T = T_raw[0].item()              # common value per graph
            graph_num = T_raw.size(0)        # number of graphs

    W, H = graph.cor.max(axis=0)[0]+1
    nodes_per_graph = T * W * H

    def extract_sample_from_cluster_dict(
        cluster_dict,
        graph_idx,
    ) -> list[torch.Tensor]:
        device = torch.device("cuda")
        samples = []
        offset = graph_idx * nodes_per_graph
        for cluster_num, idx_tensor in cluster_dict.items():
            n = idx_tensor.numel()
            if n < num_per_cluster:
                raise AssertionError(
                    f"Cluster {cluster_num} has {n} nodes, but requested {num_per_cluster} samples, "
                    "reduce sampling rates or reduce number of clusters."
                )
            # k = min(n, num_per_cluster)
            k = num_per_cluster
            perm = torch.randperm(n)[:k]
            chosen = idx_tensor[perm].to(device)
            samples.append(chosen+offset)
        return samples

    samples = []
    if cluster_dim == '2d':
        for t in range(T_total):
            # cluster_dict = getattr(graph, 'cluster_set', None)
            # if not cluster_dict:
            #     continue
            frame_cluster_dict = graph.cluster_set[t]
            graph_idx = t // T

            frame_samples = extract_sample_from_cluster_dict(frame_cluster_dict, graph_idx)

            samples.extend(frame_samples)
    elif cluster_dim == '3d':
        for graph_idx in range(graph_num):
            if isinstance(graph.cluster_set, list):
                graph_cluster_dict = graph.cluster_set[graph_idx]
            elif isinstance(graph.cluster_set, dict):
                graph_cluster_dict = graph.cluster_set
            else:
                raise TypeError(f"graph.cluster_set must be a list or dict, got {type(graph.cluster_set)}")
            graph_samples = extract_sample_from_cluster_dict(graph_cluster_dict, graph_idx)
            samples.extend(graph_samples)

    #     return torch.empty(0, dtype=torch.long)

    all_samples = torch.cat(samples)
    return all_samples


def graph_3d_cluster(graph, n_segments, compactness, cluster_type='slic'):
    T = graph.T.sum()
    W, H = graph.cor.max(axis=0)[0] +1
    graph.cluster_set = [defaultdict(dict)]
    vol = graph.feat.reshape(T, W, H)

    if cluster_type == 'slic':
        segments = slic(
            vol,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=None
        )
    else:
        raise NotImplementedError()

    segments_flat = torch.tensor(segments).reshape(-1)
    for i in range(segments.max() + 1):
        mask = segments_flat == i
        graph.cluster_set[0][i] = torch.where(mask)[0]
    graph.segments = segments


def graph_2d_cluster(graph, n_segments, compactness, cluster_type='slic'):
    T = graph.T.sum()

    W, H = graph.cor.max(axis=0)[0] +1

    graph.cluster_set = [defaultdict(dict) for _ in range(graph.T)]

    for t in range(T):
        indices_t = get_graph_t_idx(graph, t)
        image = graph.feat[indices_t].reshape(W, H)

        if cluster_type == 'slic':
            segments = slic(image,
                            n_segments=n_segments,
                            compactness=compactness,
                            start_label=0,
                            channel_axis=None)
        else:
            raise NotImplementedError()

        segments_flat = torch.tensor(segments).reshape(-1)
        for i in range(segments.max() + 1):
            mask = segments_flat == i
            graph.cluster_set[t][i] = indices_t[mask]

        # graph.cluster_label[indices_t] = torch.tensor(segments).reshape(-1, 1)


def add_cluster_label(data_loader, n_segments, compactness, cluster_type='slic', cluster_dim='2d'):
    '''
    Due to preprocessing the cluster_set including the index that correspond to the single graph
    
    for a batch of graph, we need to add the time label in each graph manually, add graph.time[mask].min
    https://vscode.dev/github/guangzhao27/SOMA_INR/blob/master/train_utility_sampling/train_utility.py#L68
    '''
    for i, graph in enumerate(data_loader.dataset):
        if cluster_dim == '2d':
            graph_2d_cluster(graph, n_segments, compactness, cluster_type)
        elif cluster_dim == '3d':
            graph_3d_cluster(graph, n_segments, compactness, cluster_type)
        else:
            raise NotImplementedError()


class INRSingle2dSamplerWrapper(InrSamplerWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        iters: int,
        n_clusters_2d_start: int = 100,
        n_clusters_2d_end: int = 100,
        epochs: int = 5000,
        device: str = "cuda:0",
        sample_type: str = "random",
        use_weight_function: bool = True,
        sample_rate: float = 0.5,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        image_width: int = 512,
        cell_size: int = 32,
        k_per_cell: int = 1,
        stratified_allocation: str = "neyman",
        stratified_n_bins: int = 16,
        stratified_min_alloc_frac: float = 0.1,
        stratified_update_interval: int = 500,
        stratified_pilot_per_cell: int = 16,
    ):
        super().__init__(
            model=model,
            iters=iters,
            n_clusters_2d_start=n_clusters_2d_start,
            n_clusters_2d_end=n_clusters_2d_end,
            epochs=epochs,
            device=device,
            sample_type=sample_type,
            use_weight_function=use_weight_function,
            sample_rate=sample_rate,
            save_samples_path=save_samples_path,
            save_interval=save_interval,
            image_width=image_width,
        )
        if sample_type == "2d_grid_fixed":
            self.cell_size = cell_size
            self.k_per_cell = k_per_cell

        if sample_type == "2d_grid_stratified":
            if stratified_allocation not in ("proportional", "neyman"):
                raise ValueError(
                    f"Unknown stratified_allocation '{stratified_allocation}'. "
                    "Expected 'proportional' or 'neyman'."
                )
            self.stratified_allocation = stratified_allocation
            self.stratified_n_bins = int(stratified_n_bins)
            self.stratified_min_alloc_frac = float(stratified_min_alloc_frac)
            self.stratified_update_interval = max(1, int(stratified_update_interval))
            self.stratified_pilot_per_cell = int(stratified_pilot_per_cell)
            # Fixed partition: built once, never rebuilt.
            self._strat_bounds: Optional[torch.Tensor] = None
            self._strat_cell_n: Optional[torch.Tensor] = None   # N_h, points per cell
            self._strat_sigma: Optional[torch.Tensor] = None    # sigma_h, refreshed on interval
            self._strat_last_update: Optional[int] = None

    def _build_stratified_partition(self) -> None:
        """Build the fixed uniform grid once and cache bounds plus per-cell point counts N_h."""
        bounds = build_uniform_grid_bounds_2d(
            self.image_width, self.stratified_n_bins, device='cpu'
        )
        widths = bounds[:, 1] - bounds[:, 0] + 1
        heights = bounds[:, 3] - bounds[:, 2] + 1
        self._strat_bounds = bounds
        self._strat_cell_n = (widths * heights).to(torch.float32).to(self.device)

    def _refresh_stratified_sigma(self, graph: Data, inner_step: int) -> None:
        """Re-estimate sigma_h from pilot samples if the refresh interval has elapsed.

        sigma_h is the within-cell standard deviation of the per-point squared error --
        the quantity the training objective averages -- which is what textbook Neyman
        allocation calls for. Pilot points are used for scoring only, never for training.
        """
        due = (
            self._strat_sigma is None
            or self._strat_last_update is None
            or (inner_step - self._strat_last_update) >= self.stratified_update_interval
        )
        if not due:
            return

        # The estimator indexes rows/cols, so hand it (r=y, c=x) ordered bounds.
        cell_rc = self._strat_bounds[:, [2, 3, 0, 1]]
        loss_var = cell_loss_variance_batched(
            cell_rc, graph, self.model, self.device,
            max_samples_per_cell=self.stratified_pilot_per_cell, use_sqrt=False,
        )
        self._strat_sigma = loss_var.clamp_min(0.0).sqrt().to(self.device)
        self._strat_last_update = inner_step

    def sample(
        self,
        inner_step: int,
        graph: Data,
        save_image: bool = False,
        ) -> Data:
        """ Sample random coordinates from a single 2D graph.

        Args:
            inner_step (int): The current inner step, used for saving images.
            graph (Data): The input graph data.
            save_image (bool): Whether to save the sampled image.

        Returns:
            Data: A new Data object containing the sampled nodes.
        """
        n_t = graph.cor.shape[0]  # total number of corrdinates in the graph
        n_samples = max(int(n_t * self.sample_rate), 1)
        sampling_weight = None
        if self.sample_type == "random":
            sampled_idx = torch.randperm(n_t, device=self.device)[:n_samples]

        elif self.sample_type == "NMT":
            with torch.no_grad():
                graph = graph.to(self.device)
                preds = self.model(graph.space_emb)
                features = graph.feat
                dif = torch.sum(torch.abs(features - preds), 1)
                _, sampled_idx = torch.topk(dif, n_samples)

        elif self.sample_type == '2d_grid_linear':
            # t0 = time()
            _start = self.n_clusters_2d_start
            _end = self.n_clusters_2d_end
            n_bins= np.round(_start + ((_end - _start) / self.epochs) * inner_step).astype(int)
            n_per_cell = max(1, math.ceil(n_samples / (n_bins * n_bins)))
            bounds_1d = generate_equal_bins(0, self.image_width-1, n_bins, device='cpu')
            x_bounds = bounds_1d  # (n_bins, 2)
            y_bounds = bounds_1d  # assuming square grid, reuse same bounds for y
            # Create all combinations of x and y bounds
            x_low = x_bounds[:, 0].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)  # (n_bins*n_bins,)
            x_high = x_bounds[:, 1].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)  # (n_bins*n_bins,)
            y_low = y_bounds[:, 0].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)  # (n_bins*n_bins,)
            y_high = y_bounds[:, 1].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)  # (n_bins*n_bins,)

            bounds = torch.stack([x_low, x_high, y_low, y_high], dim=1)  # (n_bins*n_bins, 4)
            self.cached_linear_bounds = bounds.detach().cpu()
            self.cached_linear_n_bins = int(n_bins)
            cor = sample_multiple_from_2d_intervals(bounds, n_per_cell, device='cpu')

            x_coords = cor[:, :, 0]  # x-coordinates
            y_coords = cor[:, :, 1]  # y-coordinates
            rough_idx = (y_coords * self.image_width + x_coords).flatten().to(self.device)
            space_emb = graph.space_emb[rough_idx].to(self.device)  # [N_rough, D]
            feats = graph.feat[rough_idx].to(self.device)  # [N_rough, F]
            # t0 = time()
            with torch.no_grad():
                preds = self.model(space_emb)
                dif = torch.sum((feats - preds).abs(), dim=1)
            n_samples = min(n_samples, len(dif))
            _, topk_local = torch.topk(dif, n_samples)
            sampled_idx = rough_idx[topk_local]

        elif self.sample_type == '2d_grid_linear_weighted':
            # Fixed-size uniform grid sampling without top-k filtering.
            # Every sampled point is kept and weighted by its cell's mean sampled loss.
            _start = self.n_clusters_2d_start
            _end = self.n_clusters_2d_end
            n_bins = np.round(_start + ((_end - _start) / self.epochs) * inner_step).astype(int)
            n_per_cell = max(1, math.ceil(n_samples / (n_bins * n_bins)))

            bounds_1d = generate_equal_bins(0, self.image_width - 1, n_bins, device='cpu')
            x_bounds = bounds_1d
            y_bounds = bounds_1d

            x_low = x_bounds[:, 0].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
            x_high = x_bounds[:, 1].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
            y_low = y_bounds[:, 0].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
            y_high = y_bounds[:, 1].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
            bounds = torch.stack([x_low, x_high, y_low, y_high], dim=1)
            self.cached_linear_bounds = bounds.detach().cpu()
            self.cached_linear_n_bins = int(n_bins)

            cor = sample_multiple_from_2d_intervals(bounds, n_per_cell, device='cpu')
            x_coords = cor[:, :, 0]
            y_coords = cor[:, :, 1]

            rough_idx = (y_coords * self.image_width + x_coords).flatten().to(self.device)
            sampled_idx = rough_idx

            if self.use_weight_function:
                # Compute sampled per-point loss and assign each point its cell mean loss as weight.
                with torch.no_grad():
                    space_emb = graph.space_emb[sampled_idx].to(self.device)
                    feats = graph.feat[sampled_idx].to(self.device)
                    preds = self.model(space_emb)
                    per_point_loss = (feats - preds).pow(2).sum(dim=1)

                    n_cells = n_bins * n_bins
                    per_cell_mean = per_point_loss.view(n_cells, n_per_cell).mean(dim=1)
                    sampling_weight = per_cell_mean.repeat_interleave(n_per_cell)

                    # Keep average weight near 1 to avoid changing global loss scale.
                    mean_w = sampling_weight.mean()
                    if torch.isfinite(mean_w) and mean_w.item() > 0:
                        sampling_weight = sampling_weight / mean_w
                    else:
                        sampling_weight = torch.ones_like(sampling_weight)
            dif = None

            # sampled_idx = rough_idx[topk_local]


            # pass
            # if inner_step % 1 == 0:
            #     graph = graphtreebuilder_2d_adaptive_single_image()
            # n_per_cell = max(1, math.ceil(n_samples / len(graph.cluster_set[0])))

            # cor = sample_multiple_from_2d_intervals(bounds, n_per_cell, device='cpu')
            # graph = graph.update_with_samples(cor)

            # and then add weight to each sample, add weight to graph.


        elif self.sample_type == '2d_grid_fixed':
            # Fixed-size uniform grid partition sampler.
            # Partition space into a uniform grid of cells with fixed pixel size.
            # Sample exactly k_per_cell points from each cell uniformly at random.
            # Weight each sample by its cell's mean sampled MSE loss (proportional
            # to average difficulty of that region), then normalize to unit mean.
            n_bins = max(1, self.image_width // self.cell_size)
            k = self.k_per_cell

            bounds_1d = generate_equal_bins(0, self.image_width - 1, n_bins, device='cpu')
            x_low = bounds_1d[:, 0].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
            x_high = bounds_1d[:, 1].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
            y_low = bounds_1d[:, 0].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
            y_high = bounds_1d[:, 1].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
            bounds = torch.stack([x_low, x_high, y_low, y_high], dim=1)  # (n_cells, 4)

            n_cells = n_bins * n_bins
            cor = sample_multiple_from_2d_intervals(bounds, k, device='cpu')  # (n_cells, k, 2)
            x_coords = cor[:, :, 0]  # (n_cells, k)
            y_coords = cor[:, :, 1]  # (n_cells, k)
            sampled_idx = (y_coords * self.image_width + x_coords).flatten().to(self.device)

            with torch.no_grad():
                space_emb = graph.space_emb[sampled_idx].to(self.device)
                feats = graph.feat[sampled_idx].to(self.device)
                preds = self.model(space_emb)
                per_point_loss = (feats - preds).pow(2).sum(dim=1)  # (n_cells * k,)

                # Weight each sample by its cell's mean loss, then normalize.
                per_cell_mean = per_point_loss.view(n_cells, k).mean(dim=1)  # (n_cells,)
                sampling_weight = per_cell_mean.repeat_interleave(k)          # (n_cells * k,)
                mean_w = sampling_weight.mean()
                if torch.isfinite(mean_w) and mean_w.item() > 0:
                    sampling_weight = sampling_weight / mean_w
                else:
                    sampling_weight = torch.ones_like(sampling_weight)

        elif self.sample_type == '2d_grid_stratified':
            # Fixed-grid stratified sampling with proportional or Neyman allocation.
            # The partition never changes; only the allocation across cells adapts (and
            # only under Neyman). Weighting each point by N_h / n~_h makes the resulting
            # loss an unbiased estimate of the full-grid mean loss -- see
            # stratified_continuous_allocation / stratified_poisson_counts.
            if self._strat_bounds is None:
                self._build_stratified_partition()
            bounds = self._strat_bounds
            cell_n = self._strat_cell_n

            if self.stratified_allocation == "neyman":
                self._refresh_stratified_sigma(graph, inner_step)
                scores = cell_n * self._strat_sigma
            else:
                scores = cell_n

            alloc = stratified_continuous_allocation(
                scores, n_samples, self.stratified_min_alloc_frac
            )
            counts = stratified_poisson_counts(alloc)
            if int(counts.sum().item()) == 0:  # degenerate draw; fall back to one point
                counts[torch.randint(counts.numel(), (1,))] = 1

            samples_xy, cell_ids, _ = sample_variable_from_2d_intervals_vcounts(
                bounds, counts, device=self.device
            )
            sampled_idx = samples_xy[:, 1] * self.image_width + samples_xy[:, 0]

            # Hansen-Hurwitz weights use the *expected* allocation n~_h, not the realised
            # count n_h: with the realised count, cells that happen to draw zero samples
            # drop out of the estimator and low-variance regions are systematically
            # under-counted. No clipping -- it would reintroduce bias.
            sampling_weight = (cell_n / alloc.clamp_min(1e-12))[cell_ids]
            dif = None

        elif self.sample_type == "2d_cluster_slic":
            # For 2D graphs, we can still use the 3D cluster sampling function
            # but it will sample from the 2D clusters.
            # This is a workaround to use the same sampling function.
            # In practice, you might want to implement a separate 2D sampling function.
            # Here we assume the graph has been clustered already.

            if inner_step % 100 == 0:
                _start = self.n_clusters_2d_start
                _end = self.n_clusters_2d_end
                n_clusters = _start + ((_end - _start) / self.epochs) * inner_step
                graph_2d_cluster_single_image(graph, n_clusters, 0.01, 'grid')


            _start = self.n_clusters_2d_start
            _end = self.n_clusters_2d_end
            n_clusters = _start + ((_end - _start) / self.epochs) * inner_step
            graph_2d_cluster_single_image(graph, n_clusters, 0.01, 'grid')

            num_per_cluster = max(1, math.ceil(n_samples / len(graph.cluster_set[0])))
            rough_idx = sample_random_node_indices_per_cluster(
                graph, cluster_dim='2d', num_per_cluster=num_per_cluster
                )
            space_emb = graph.space_emb[rough_idx].to(self.device)  # [N_rough, D]
            feats = graph.feat[rough_idx].to(self.device)  # [N_rough, F]
            with torch.no_grad():
                preds = self.model(space_emb)
                dif = torch.sum((feats - preds).abs(), dim=1)
            n_samples = min(n_samples, len(dif))
            _, topk_local = torch.topk(dif, n_samples)
            sampled_idx = rough_idx[topk_local.cpu()]
            # sampling_weight = torch.ones_like(sampled_idx, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Sampling type {self.sample_type} is not implemented.")

        # print("---sampled_idx---" + str(sampled_idx))
        graph.to(self.device)
        sampled_data_kwargs = dict(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
        )
        if sampling_weight is not None:
            sampled_data_kwargs['weight'] = sampling_weight
        sampled_graph = Data(**sampled_data_kwargs)

        # print("---sampled_graph---" + str(sampled_graph))

        if save_image:
            self.save_image_path = os.path.join(self.save_samples_path, f"2d_i{inner_step}")
            if self.sample_type != "NMT":
                dif = None
            self._save_sample_images(graph, sampled_graph, dif=dif)
        return sampled_graph.to(self.device)


class INRSingle2dAdaptiveSamplerWrapper(InrSamplerWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        iters: int,
        device: str = "cuda:0",
        sample_rate: float = 0.5,
        mode: str = "loss",
        weight_mode: str = "inverse_value",
        weight_value_eps: float = 1e-6,
        weight_clip_ratio: float = 10.0,
        equal_cell_topk: bool = False,
        equal_cell_topk_count_mode: str = "same",
        equal_cell_topk_weight_mode: str = "none",
        power_for_loss_as_weight: float = 0.2,
        grid_update_interval: int = 100,
        adaptive_iterations: int = 8,
        subdivision_percentage: float = 20.0,
        count_floor_mode: str = "min_one",
        count_floor_frac: float = 0.1,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        image_width: int = 512,
        ):
        super().__init__(
            model=model,
            iters=iters,
            device=device,
            sample_type="2d_grid_adaptive",
            sample_rate=sample_rate,
            save_samples_path=save_samples_path,
            save_interval=save_interval,
            image_width=image_width,
        )
        self.mode = mode
        valid_weight_modes = {"none", "inverse_value", "unbiased_weight", "sampled_dif", "loss_powered_weight"}
        if weight_mode not in valid_weight_modes:
            raise ValueError(
                f"Unknown weight_mode '{weight_mode}'. Expected one of {sorted(valid_weight_modes)}"
            )
        self.weight_mode = weight_mode
        self.weight_value_eps = float(weight_value_eps)
        self.weight_clip_ratio = float(weight_clip_ratio)
        self.equal_cell_topk = bool(equal_cell_topk)

        valid_count_modes = {"same", "poisson"}
        if equal_cell_topk_count_mode not in valid_count_modes:
            raise ValueError(
                "Unknown equal_cell_topk_count_mode "
                f"'{equal_cell_topk_count_mode}'. Expected one of {sorted(valid_count_modes)}"
            )
        self.equal_cell_topk_count_mode = equal_cell_topk_count_mode

        valid_equal_topk_weight_modes = {"none", "unbiased_weight", "loss_powered_weight"}
        if equal_cell_topk_weight_mode not in valid_equal_topk_weight_modes:
            raise ValueError(
                "Unknown equal_cell_topk_weight_mode "
                f"'{equal_cell_topk_weight_mode}'. Expected one of {sorted(valid_equal_topk_weight_modes)}"
            )
        self.equal_cell_topk_weight_mode = equal_cell_topk_weight_mode
        self.power_for_loss_as_weight = float(power_for_loss_as_weight)

        self.grid_update_interval = max(1, int(grid_update_interval))
        self.adaptive_iterations = max(1, int(adaptive_iterations))
        self.subdivision_percentage = float(subdivision_percentage)
        if count_floor_mode not in ("min_one", "soft"):
            raise ValueError(
                f"Unknown count_floor_mode '{count_floor_mode}'. Expected 'min_one' or 'soft'."
            )
        self.count_floor_mode = count_floor_mode
        self.count_floor_frac = float(count_floor_frac)

        # Store graph reference for evaluation function
        self.cached_graph = None
        self.last_grid_update_step: Optional[int] = None
        self.cached_bounds: Optional[torch.Tensor] = None
        self.cached_cell_sizes: Optional[torch.Tensor] = None
        self.cached_values: Optional[torch.Tensor] = None
        self.cached_grid: Optional[HierarchicalImageGrid] = None

    def _normalize_and_clip_cell_weights(self, per_cell_weight: torch.Tensor) -> torch.Tensor:
        """Clamp invalid/extreme cell weights, then normalize by mean for stability."""
        return normalize_and_clip_cell_weights(per_cell_weight, self.weight_clip_ratio)

    def _should_refresh_cache(self, inner_step: int) -> bool:
        """Refresh cached bounds/sizes/values on first use and every `grid_update_interval` steps."""
        if self.last_grid_update_step is None:
            return True
        if self.cached_bounds is None or self.cached_cell_sizes is None or self.cached_values is None:
            return True
        return (inner_step - self.last_grid_update_step) >= self.grid_update_interval

    def _refresh_cached_leaf_properties(
        self,
        eval_fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rebuild adaptive grid and cache leaf bounds/areas/values."""
        grid = HierarchicalImageGrid(self.image_width, self.image_width, initial_grid_size=8)
        grid.iterative_subdivision(
            eval_fn,
            iterations=self.adaptive_iterations,
            percentage=self.subdivision_percentage,
            batch_mode=True,
        )

        bounds, cell_sizes, values = grid.get_leaf_properties_tensor(
            evaluation_function=eval_fn,
            device=self.device,
        )
        self.cached_bounds = bounds
        self.cached_cell_sizes = cell_sizes
        self.cached_values = values
        self.cached_grid = grid
        return bounds, cell_sizes, values


    def _create_evaluation_function(self, graph: Data, mode: str = 'gradient'):
        """
        Create evaluation function with graph context.
        Uses the new quadtree API where cells are passed directly as ImageCell objects.
        
        Args:
            graph: The graph data to evaluate cells against
            
        Returns:
            Evaluation function that takes cells and returns variance estimates
        """
        def evaluate_cells(cells: list) -> list:
            """
            Evaluate gradient variance for a list of cells.
            
            Args:
                cells: List of ImageCell objects
                
            Returns:
                List of variance values (one per cell)
            """
            # Build tensor of cell coordinates: [N, 4] with format [y_start, y_end, x_start, x_end]
            cell_coords = []
            cell_areas = []

            for cell in cells:
                # Note: cell boundaries are now inclusive after quadtree fix
                cell_coords.append([cell.y_start, cell.y_end, cell.x_start, cell.x_end])
                cell_areas.append(cell.area)

            cell_coords_tensor = torch.tensor(cell_coords, device=self.device)
            cell_areas_tensor = torch.tensor(cell_areas, device=self.device, dtype=torch.float32)

            # Compute gradient variance for all cells at once
            if mode == 'gradient':
                grad_variances = cell_grad_variance_estimate_with_jacrev(
                    cell_coords_tensor, graph, self.model, self.device
                )

                # Weight by cell area (larger cells contribute more)
                weighted_std = grad_variances.sqrt() * cell_areas_tensor

                return weighted_std.tolist()

            if mode == 'loss_true':
                loss_variance = loss_variance_ground_truth(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_variance.sqrt() * cell_areas_tensor
                return cell_value.tolist()

            if mode == 'loss':
                loss_variance = cell_loss_variance_estimate_with_random_sampling(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_variance.sqrt() * cell_areas_tensor
                return cell_value.tolist()

            if mode == 'loss_no_sqrt':
                loss_variance = cell_loss_variance_estimate_with_random_sampling(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_variance * cell_areas_tensor
                return cell_value.tolist()

            if mode == "loss_sqrt_std":
                loss_sqrt_variance = cell_sqrt_loss_variance_estimate_with_random_sampling(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_sqrt_variance * cell_areas_tensor
                return cell_value.tolist()

            if mode == "loss_sqrt_no_area":
                loss_sqrt_variance = cell_sqrt_loss_variance_estimate_with_random_sampling(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_sqrt_variance.sqrt()
                return cell_value.tolist()

            if mode == "loss_sqrt_std2":
                loss_sqrt_variance = cell_sqrt_loss_variance_estimate_with_random_sampling(
                    cell_coords_tensor, graph, self.model, self.device
                )

                cell_value = loss_sqrt_variance.sqrt() * cell_areas_tensor
                return cell_value.tolist()

            raise ValueError(f"Unknown adaptive evaluation mode: {mode}")

        return evaluate_cells


    def sample(
        self,
        inner_step: int,
        graph: Data,
        save_image: bool = False,
    ) -> Data:
        n_t = graph.cor.shape[0]
        n_samples = max(int(n_t * self.sample_rate), 1)

        eval_fn = self._create_evaluation_function(graph, mode=self.mode)

        if self._should_refresh_cache(inner_step):
            bounds, cell_area, values = self._refresh_cached_leaf_properties(eval_fn)
            self.last_grid_update_step = inner_step
        else:
            if self.cached_bounds is None or self.cached_cell_sizes is None or self.cached_values is None:
                raise RuntimeError("Adaptive cache is empty when attempting to reuse it.")
            bounds, cell_area, values = self.cached_bounds, self.cached_cell_sizes, self.cached_values

        n_cells = bounds.shape[0]

        if self.equal_cell_topk and self.equal_cell_topk_count_mode == "same":
            n_per_cell = max(1, math.ceil(n_samples / n_cells))
            counts = torch.full((n_cells,), n_per_cell, dtype=torch.long, device=self.device)
            expected_counts = counts.to(torch.float32)
        else:
            counts, expected_counts = adaptive_cell_counts(
                values, n_samples, self.count_floor_mode, self.count_floor_frac
            )

        samples_xy, cell_ids, ptr = sample_variable_from_2d_intervals_vcounts(bounds, counts, device=self.device)
        sampled_idx = samples_xy[:, 1] * self.image_width + samples_xy[:, 0]

        graph_on_device = graph.to(self.device)
        # Weight modes divide by a per-cell count. Under "min_one" that is the realised
        # count (existing behaviour); under "soft" it is the expected allocation, which is
        # what keeps the weighting meaningful once cells are allowed to draw zero samples.
        counts_f = expected_counts.to(torch.float32)
        cell_area_f = cell_area.to(torch.float32)
        values_f = values.to(torch.float32)

        # Run model inference once if needed
        feats = preds = None
        if self.equal_cell_topk or self.weight_mode in ("sampled_dif", "loss_powered_weight"):
            with torch.no_grad():
                feats = graph_on_device.feat[sampled_idx]
                preds = self.model(graph_on_device.space_emb[sampled_idx])

        if self.equal_cell_topk:
            sampled_dif = (feats - preds).abs().sum(dim=1)
            keep_k = min(n_samples, sampled_dif.numel())
            _, topk_local = torch.topk(sampled_dif, keep_k)
            selected_idx = sampled_idx[topk_local]

            sampled_data_kwargs = dict(
                cor=graph_on_device.cor[selected_idx],
                time=graph_on_device.time[selected_idx],
                feat=graph_on_device.feat[selected_idx],
                space_emb=graph_on_device.space_emb[selected_idx],
            )

            weight_mode = self.equal_cell_topk_weight_mode
            if weight_mode in ("unbiased_weight", "loss_powered_weight"):
                counts_fc = counts_f.clamp_min(1.0)
                cell_area_fc = cell_area_f.clamp_min(1.0)
                if weight_mode == "loss_powered_weight":
                    per_point_loss = (feats - preds).pow(2).sum(dim=1)
                    per_cell_loss_sum = torch.zeros(n_cells, device=self.device, dtype=torch.float32)
                    per_cell_loss_sum.index_add_(0, cell_ids, per_point_loss.to(torch.float32))
                    per_cell_mean_loss = per_cell_loss_sum / counts_fc
                    per_cell_weight = (
                        torch.pow(per_cell_mean_loss.clamp_min(0.0), self.power_for_loss_as_weight)
                        * cell_area_fc / counts_fc
                    )
                else:  # unbiased_weight
                    per_cell_weight = cell_area_fc / counts_fc
                per_cell_weight = self._normalize_and_clip_cell_weights(per_cell_weight)
                sampled_data_kwargs["weight"] = per_cell_weight[cell_ids[topk_local]]

            return Data(**sampled_data_kwargs).to(self.device)

        # Standard (non-topk) path: compute per-sample weights
        if self.weight_mode == "sampled_dif":
            sampled_weights = self._normalize_and_clip_cell_weights((feats - preds).abs().sum(dim=1))
        elif self.weight_mode == "loss_powered_weight":
            counts_fc = counts_f.clamp_min(1.0)
            cell_area_fc = cell_area_f.clamp_min(1.0)
            per_point_loss = (feats - preds).pow(2).sum(dim=1)
            per_cell_loss_sum = torch.zeros(n_cells, device=self.device, dtype=torch.float32)
            per_cell_loss_sum.index_add_(0, cell_ids, per_point_loss.to(torch.float32))
            per_cell_mean_loss = per_cell_loss_sum / counts_fc
            per_cell_weight = (
                torch.pow(per_cell_mean_loss.clamp_min(0.0), self.power_for_loss_as_weight)
                * cell_area_fc / counts_fc
            )
            sampled_weights = self._normalize_and_clip_cell_weights(per_cell_weight)[cell_ids]
        elif self.weight_mode == "none":
            sampled_weights = self._normalize_and_clip_cell_weights(torch.ones_like(cell_area_f))[cell_ids]
        elif self.weight_mode == "inverse_value":
            per_cell_weight = cell_area_f / values_f.clamp_min(self.weight_value_eps)
            sampled_weights = self._normalize_and_clip_cell_weights(per_cell_weight)[cell_ids]
        elif self.weight_mode == "unbiased_weight":
            per_cell_weight = cell_area_f.clamp_min(1.0) / counts_f.clamp_min(1.0)
            sampled_weights = self._normalize_and_clip_cell_weights(per_cell_weight)[cell_ids]
        else:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

        return Data(
            cor=graph_on_device.cor[sampled_idx],
            time=graph_on_device.time[sampled_idx],
            feat=graph_on_device.feat[sampled_idx],
            space_emb=graph_on_device.space_emb[sampled_idx],
            weight=sampled_weights,
        ).to(self.device)


class INRSingle3dSamplerWrapper(InrSamplerWrapper):
    """
    Sampler for single-volume 3D datasets (e.g. SOMA ocean data). Mirrors
    INRSingle2dSamplerWrapper, but coordinates are 3-column (x, y, z) and the
    coordinate grid is not fully dense (an ocean/land mask leaves gaps inside the
    bounding box), so grid-based sample types resolve candidate voxels through an
    index-grid lookup (build_3d_index_grid) instead of a dense flatten-index formula.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        iters: int,
        device: str = "cuda:0",
        sample_type: str = "random",
        sample_rate: float = 0.5,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        grid_shape=(1, 1, 1),
        cell_size: int = 32,
        k_per_cell: int = 1,
        n_bins_3d: int = 8,
        stratified_allocation: str = "neyman",
        stratified_n_bins=16,
        stratified_min_alloc_frac: float = 0.1,
        stratified_update_interval: int = 100,
        stratified_pilot_per_cell: int = 16,
    ):
        super().__init__(
            model=model,
            iters=iters,
            device=device,
            sample_type=sample_type,
            sample_rate=sample_rate,
            save_samples_path=save_samples_path,
            save_interval=save_interval,
        )
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.cell_size = cell_size
        self.k_per_cell = k_per_cell
        self.n_bins_3d = n_bins_3d
        self._index_grid: Optional[torch.Tensor] = None

        if sample_type == "3d_grid_stratified":
            if stratified_allocation not in ("proportional", "neyman"):
                raise ValueError(
                    f"Unknown stratified_allocation '{stratified_allocation}'. "
                    "Expected 'proportional' or 'neyman'."
                )
            self.stratified_allocation = stratified_allocation
            self.stratified_n_bins = stratified_n_bins
            self.stratified_min_alloc_frac = float(stratified_min_alloc_frac)
            self.stratified_update_interval = max(1, int(stratified_update_interval))
            self.stratified_pilot_per_cell = int(stratified_pilot_per_cell)
            self._strat_bounds: Optional[torch.Tensor] = None
            self._strat_cell_n: Optional[torch.Tensor] = None      # N_h: valid voxels per cell
            self._strat_cell_vol: Optional[torch.Tensor] = None    # geometric box volume per cell
            self._strat_sigma: Optional[torch.Tensor] = None
            self._strat_last_update: Optional[int] = None

    def _get_index_grid(self, graph: Data) -> torch.Tensor:
        if self._index_grid is None:
            self._index_grid = build_3d_index_grid(graph.cpu(), self.grid_shape)
        return self._index_grid

    def _build_stratified_partition(self, graph: Data) -> None:
        """Build the fixed voxel grid once; cache bounds, geometric volumes and valid counts.

        Volumes may be non-dense (SOMA's land mask), so the number of *valid* voxels N_h
        differs from the geometric box volume. Both are needed: N_h drives the allocation
        and the weights, while the volume converts a valid-point budget into the number of
        geometric draws to make (draws landing on a hole are discarded).
        """
        bounds = build_uniform_grid_bounds_3d(self.grid_shape, self.stratified_n_bins, device='cpu')
        vol = ((bounds[:, 1] - bounds[:, 0] + 1)
               * (bounds[:, 3] - bounds[:, 2] + 1)
               * (bounds[:, 5] - bounds[:, 4] + 1)).to(torch.float32)

        # Assign every real coordinate to its cell (x-major, then y, then z -- the
        # ordering cross_bins_3d produces) and count valid voxels per cell.
        n_cells = bounds.size(0)
        x_hi = bounds[:, 1].unique(sorted=True)
        y_hi = bounds[:, 3].unique(sorted=True)
        z_hi = bounds[:, 5].unique(sorted=True)
        cor = graph.cor.cpu().long()
        ix = torch.bucketize(cor[:, 0], x_hi)
        iy = torch.bucketize(cor[:, 1], y_hi)
        iz = torch.bucketize(cor[:, 2], z_hi)
        flat = ix * (y_hi.numel() * z_hi.numel()) + iy * z_hi.numel() + iz
        cell_n = torch.bincount(flat, minlength=n_cells).to(torch.float32)

        self._strat_bounds = bounds
        self._strat_cell_vol = vol.to(self.device)
        self._strat_cell_n = cell_n.to(self.device)

    def _refresh_stratified_sigma(self, graph: Data, inner_step: int) -> None:
        """Re-estimate sigma_h (within-cell std of the per-point squared error) on interval."""
        due = (
            self._strat_sigma is None
            or self._strat_last_update is None
            or (inner_step - self._strat_last_update) >= self.stratified_update_interval
        )
        if not due:
            return

        loss_var = cell_loss_variance_batched_3d(
            self._strat_bounds, self.grid_shape, graph, self.model, self.device,
            max_samples_per_cell=self.stratified_pilot_per_cell, use_sqrt=False,
            index_grid=self._get_index_grid(graph),
        )
        self._strat_sigma = loss_var.clamp_min(0.0).sqrt().to(self.device)
        self._strat_last_update = inner_step

    def sample(
        self,
        inner_step: int,
        graph: Data,
        save_image: bool = False,
    ) -> Data:
        """Sample coordinates from a single 3D volumetric graph."""
        n_t = graph.cor.shape[0]
        n_samples = max(int(n_t * self.sample_rate), 1)
        sampling_weight = None

        if self.sample_type == "random":
            sampled_idx = torch.randperm(n_t, device=self.device)[:n_samples]

        elif self.sample_type == "NMT":
            with torch.no_grad():
                graph = graph.to(self.device)
                se = graph.space_emb
                n_pts = se.shape[0]
                _chunk = 2_000_000
                if n_pts > _chunk:
                    # Chunk the full-volume forward: for NS3D 512x512x64 (16.8M
                    # voxels) a single forward through the deep/wide SIREN OOMs an
                    # 80GB GPU. NMT is pointwise (per-voxel |gt-pred|), so per-chunk
                    # dif is exactly equivalent. Small inputs keep the original path.
                    dif = torch.cat([
                        torch.sum(torch.abs(graph.feat[i:i + _chunk] - self.model(se[i:i + _chunk])), 1)
                        for i in range(0, n_pts, _chunk)
                    ])
                else:
                    preds = self.model(se)
                    dif = torch.sum(torch.abs(graph.feat - preds), 1)
                _, sampled_idx = torch.topk(dif, n_samples)

        elif self.sample_type == "3d_grid_linear":
            index_grid = self._get_index_grid(graph)
            Dx, Dy, Dz = self.grid_shape
            n_bins = self.n_bins_3d
            x_bounds = generate_equal_bins(0, Dx - 1, n_bins, device='cpu')
            y_bounds = generate_equal_bins(0, Dy - 1, n_bins, device='cpu')
            z_bounds = generate_equal_bins(0, Dz - 1, n_bins, device='cpu')
            bounds = cross_bins_3d(x_bounds, y_bounds, z_bounds)  # (n_bins^3, 6)

            n_per_cell = max(1, math.ceil(n_samples / bounds.size(0)))
            cor = sample_multiple_from_3d_intervals(bounds, n_per_cell, device='cpu')  # (n_cells, k, 3)
            xx, yy, zz = cor[..., 0].flatten(), cor[..., 1].flatten(), cor[..., 2].flatten()
            node_idx = index_grid[xx, yy, zz]
            rough_idx = node_idx[node_idx >= 0].to(self.device)

            space_emb = graph.space_emb[rough_idx].to(self.device)
            feats = graph.feat[rough_idx].to(self.device)
            with torch.no_grad():
                preds = self.model(space_emb)
                dif = torch.sum((feats - preds).abs(), dim=1)
            n_samples = min(n_samples, len(dif))
            _, topk_local = torch.topk(dif, n_samples)
            sampled_idx = rough_idx[topk_local]

        elif self.sample_type == "3d_grid_fixed":
            # Fixed-size uniform voxel grid partition sampler, analogous to 2d_grid_fixed.
            # Cells that fall entirely on land (no valid ocean voxel) contribute nothing.
            index_grid = self._get_index_grid(graph)
            Dx, Dy, Dz = self.grid_shape
            nx = max(1, Dx // self.cell_size)
            ny = max(1, Dy // self.cell_size)
            nz = max(1, Dz // self.cell_size)
            k = self.k_per_cell

            x_bounds = generate_equal_bins(0, Dx - 1, nx, device='cpu')
            y_bounds = generate_equal_bins(0, Dy - 1, ny, device='cpu')
            z_bounds = generate_equal_bins(0, Dz - 1, nz, device='cpu')
            bounds = cross_bins_3d(x_bounds, y_bounds, z_bounds)  # (n_cells, 6)
            n_cells = bounds.size(0)

            cor = sample_multiple_from_3d_intervals(bounds, k, device='cpu')  # (n_cells, k, 3)
            node_idx = index_grid[cor[..., 0], cor[..., 1], cor[..., 2]]  # (n_cells, k)
            cell_ids_grid = torch.arange(n_cells).unsqueeze(1).expand(n_cells, k)
            valid = node_idx >= 0

            sampled_idx = node_idx[valid].to(self.device)
            cell_ids = cell_ids_grid[valid].to(self.device)

            with torch.no_grad():
                space_emb = graph.space_emb[sampled_idx].to(self.device)
                feats = graph.feat[sampled_idx].to(self.device)
                preds = self.model(space_emb)
                per_point_loss = (feats - preds).pow(2).sum(dim=1)

            per_cell_sum = torch.zeros(n_cells, device=self.device)
            per_cell_count = torch.zeros(n_cells, device=self.device)
            per_cell_sum.index_add_(0, cell_ids, per_point_loss)
            per_cell_count.index_add_(0, cell_ids, torch.ones_like(per_point_loss))
            per_cell_mean = per_cell_sum / per_cell_count.clamp_min(1.0)

            sampling_weight = per_cell_mean[cell_ids]
            mean_w = sampling_weight.mean()
            if torch.isfinite(mean_w) and mean_w.item() > 0:
                sampling_weight = sampling_weight / mean_w
            else:
                sampling_weight = torch.ones_like(sampling_weight)

        elif self.sample_type == "3d_grid_stratified":
            # 3D analog of 2d_grid_stratified. The extra wrinkle is holes: a draw that
            # lands on a masked-out voxel is discarded, so to end up with an expected
            # m_h *valid* points in cell h we must make m_h * vol_h / N_h geometric
            # draws. Weighting the survivors by N_h / m_h then keeps the estimator
            # unbiased; for a dense cube vol_h == N_h and this reduces to the 2D case.
            if self._strat_bounds is None:
                self._build_stratified_partition(graph)
            bounds = self._strat_bounds
            cell_n = self._strat_cell_n
            cell_vol = self._strat_cell_vol
            index_grid = self._get_index_grid(graph)

            if self.stratified_allocation == "neyman":
                self._refresh_stratified_sigma(graph, inner_step)
                scores = cell_n * self._strat_sigma
            else:
                scores = cell_n

            # Cells that are entirely holes must not receive any budget.
            occupied = cell_n > 0
            m = torch.zeros_like(cell_n)
            m[occupied] = stratified_continuous_allocation(
                scores[occupied], n_samples, self.stratified_min_alloc_frac
            )

            draws = torch.zeros_like(m)
            draws[occupied] = m[occupied] * cell_vol[occupied] / cell_n[occupied]
            counts = stratified_poisson_counts(draws)

            samples_xyz, cell_ids, _ = sample_variable_from_3d_intervals_vcounts(
                bounds, counts, device=self.device
            )
            node_idx = index_grid.to(self.device)[
                samples_xyz[:, 0], samples_xyz[:, 1], samples_xyz[:, 2]
            ]
            keep = node_idx >= 0
            sampled_idx = node_idx[keep]
            cell_ids = cell_ids[keep]
            if sampled_idx.numel() == 0:  # degenerate draw; fall back to one random point
                sampled_idx = torch.randint(n_t, (1,), device=self.device)
                sampling_weight = torch.ones(1, device=self.device)
            else:
                sampling_weight = (cell_n / m.clamp_min(1e-12))[cell_ids]

        else:
            raise NotImplementedError(f"Sampling type {self.sample_type} is not implemented for 3D volumes.")

        graph.to(self.device)
        sampled_data_kwargs = dict(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
        )
        if sampling_weight is not None:
            sampled_data_kwargs['weight'] = sampling_weight
        sampled_graph = Data(**sampled_data_kwargs)
        return sampled_graph.to(self.device)


_VOLUMETRIC_MODE_TO_ESTIMATOR = {
    "loss": (cell_loss_variance_estimate_with_random_sampling_3d, False),
    "loss_no_sqrt": (cell_loss_variance_estimate_with_random_sampling_3d, False),
    "loss_sqrt_std": (cell_sqrt_loss_variance_estimate_with_random_sampling_3d, True),
    "loss_sqrt_no_area": (cell_sqrt_loss_variance_estimate_with_random_sampling_3d, True),
    "loss_sqrt_std2": (cell_sqrt_loss_variance_estimate_with_random_sampling_3d, True),
}


class INRSingle3dAdaptiveSamplerWrapper(InrSamplerWrapper):
    """
    Octree-based adaptive sampler for single-volume 3D datasets (e.g. SOMA ocean
    data). 3D analog of INRSingle2dAdaptiveSamplerWrapper: mirrors its cell
    caching/topk/weight-mode logic almost exactly, differing only in how the
    partition is built (octree instead of quadtree) and how sampled voxel
    coordinates are resolved to dataset indices (index-grid lookup, since the
    coordinate volume is not fully dense).

    Only the random-sampling-based cell-utility modes are supported (loss,
    loss_no_sqrt, loss_sqrt_std, loss_sqrt_no_area, loss_sqrt_std2) — the
    jacrev-based "gradient" mode and "loss_true" require a dense grid reshape
    and are not implemented for volumetric data.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        iters: int,
        device: str = "cuda:0",
        sample_rate: float = 0.5,
        mode: str = "loss",
        weight_mode: str = "inverse_value",
        weight_value_eps: float = 1e-6,
        weight_clip_ratio: float = 10.0,
        equal_cell_topk: bool = False,
        equal_cell_topk_count_mode: str = "same",
        equal_cell_topk_weight_mode: str = "none",
        power_for_loss_as_weight: float = 0.2,
        grid_update_interval: int = 100,
        adaptive_iterations: int = 8,
        subdivision_percentage: float = 20.0,
        count_floor_mode: str = "min_one",
        count_floor_frac: float = 0.1,
        initial_grid_size: int = 8,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        grid_shape=(1, 1, 1),
        ):
        super().__init__(
            model=model,
            iters=iters,
            device=device,
            sample_type="3d_grid_adaptive",
            sample_rate=sample_rate,
            save_samples_path=save_samples_path,
            save_interval=save_interval,
        )
        if mode not in _VOLUMETRIC_MODE_TO_ESTIMATOR:
            raise NotImplementedError(
                f"adaptive_mode '{mode}' is not implemented for 3D volumes. "
                f"Supported modes: {sorted(_VOLUMETRIC_MODE_TO_ESTIMATOR)}"
            )
        self.mode = mode
        valid_weight_modes = {"none", "inverse_value", "unbiased_weight", "sampled_dif", "loss_powered_weight"}
        if weight_mode not in valid_weight_modes:
            raise ValueError(
                f"Unknown weight_mode '{weight_mode}'. Expected one of {sorted(valid_weight_modes)}"
            )
        self.weight_mode = weight_mode
        self.weight_value_eps = float(weight_value_eps)
        self.weight_clip_ratio = float(weight_clip_ratio)
        self.equal_cell_topk = bool(equal_cell_topk)

        valid_count_modes = {"same", "poisson"}
        if equal_cell_topk_count_mode not in valid_count_modes:
            raise ValueError(
                "Unknown equal_cell_topk_count_mode "
                f"'{equal_cell_topk_count_mode}'. Expected one of {sorted(valid_count_modes)}"
            )
        self.equal_cell_topk_count_mode = equal_cell_topk_count_mode

        valid_equal_topk_weight_modes = {"none", "unbiased_weight", "loss_powered_weight"}
        if equal_cell_topk_weight_mode not in valid_equal_topk_weight_modes:
            raise ValueError(
                "Unknown equal_cell_topk_weight_mode "
                f"'{equal_cell_topk_weight_mode}'. Expected one of {sorted(valid_equal_topk_weight_modes)}"
            )
        self.equal_cell_topk_weight_mode = equal_cell_topk_weight_mode
        self.power_for_loss_as_weight = float(power_for_loss_as_weight)

        self.grid_update_interval = max(1, int(grid_update_interval))
        self.adaptive_iterations = max(1, int(adaptive_iterations))
        self.subdivision_percentage = float(subdivision_percentage)
        if count_floor_mode not in ("min_one", "soft"):
            raise ValueError(
                f"Unknown count_floor_mode '{count_floor_mode}'. Expected 'min_one' or 'soft'."
            )
        self.count_floor_mode = count_floor_mode
        self.count_floor_frac = float(count_floor_frac)
        self.initial_grid_size = max(1, int(initial_grid_size))
        self.grid_shape = tuple(int(v) for v in grid_shape)

        self.last_grid_update_step: Optional[int] = None
        self.cached_bounds: Optional[torch.Tensor] = None
        self.cached_cell_sizes: Optional[torch.Tensor] = None
        self.cached_values: Optional[torch.Tensor] = None
        self.cached_grid: Optional[HierarchicalVoxelGrid] = None
        self._index_grid: Optional[torch.Tensor] = None

    def _get_index_grid(self, graph: Data) -> torch.Tensor:
        if self._index_grid is None:
            self._index_grid = build_3d_index_grid(graph.cpu(), self.grid_shape)
        return self._index_grid

    def _should_refresh_cache(self, inner_step: int) -> bool:
        if self.last_grid_update_step is None:
            return True
        if self.cached_bounds is None or self.cached_cell_sizes is None or self.cached_values is None:
            return True
        return (inner_step - self.last_grid_update_step) >= self.grid_update_interval

    def _refresh_cached_leaf_properties(self, eval_fn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rebuild adaptive octree and cache leaf bounds/volumes/values."""
        grid = HierarchicalVoxelGrid(self.grid_shape, initial_grid_size=self.initial_grid_size)
        grid.iterative_subdivision(
            eval_fn,
            iterations=self.adaptive_iterations,
            percentage=self.subdivision_percentage,
            batch_mode=True,
        )

        bounds, cell_sizes, values = grid.get_leaf_properties_tensor(
            evaluation_function=eval_fn,
            device=self.device,
        )
        self.cached_bounds = bounds
        self.cached_cell_sizes = cell_sizes
        self.cached_values = values
        self.cached_grid = grid
        return bounds, cell_sizes, values

    def _create_evaluation_function(self, graph: Data, mode: str = 'loss'):
        """Create evaluation function with graph context (see INRSingle2dAdaptiveSamplerWrapper)."""
        estimator_fn, use_sqrt = _VOLUMETRIC_MODE_TO_ESTIMATOR[mode]
        weight_by_volume = mode != "loss_sqrt_no_area"

        def evaluate_cells(cells: list) -> list:
            cell_coords = []
            cell_volumes = []

            for cell in cells:
                cell_coords.append([cell.x_start, cell.x_end, cell.y_start, cell.y_end, cell.z_start, cell.z_end])
                cell_volumes.append(cell.volume)

            cell_coords_tensor = torch.tensor(cell_coords, device=self.device)
            cell_volumes_tensor = torch.tensor(cell_volumes, device=self.device, dtype=torch.float32)

            variance = estimator_fn(cell_coords_tensor, self.grid_shape, graph, self.model, self.device)

            if mode == "loss_no_sqrt":
                cell_value = variance * cell_volumes_tensor
            elif use_sqrt:
                cell_value = variance.sqrt()
                if weight_by_volume:
                    cell_value = cell_value * cell_volumes_tensor
            else:
                cell_value = variance.sqrt() * cell_volumes_tensor

            return cell_value.tolist()

        return evaluate_cells

    def sample(
        self,
        inner_step: int,
        graph: Data,
        save_image: bool = False,
    ) -> Data:
        n_t = graph.cor.shape[0]
        n_samples = max(int(n_t * self.sample_rate), 1)

        eval_fn = self._create_evaluation_function(graph, mode=self.mode)

        if self._should_refresh_cache(inner_step):
            bounds, cell_volume, values = self._refresh_cached_leaf_properties(eval_fn)
            self.last_grid_update_step = inner_step
        else:
            if self.cached_bounds is None or self.cached_cell_sizes is None or self.cached_values is None:
                raise RuntimeError("Adaptive cache is empty when attempting to reuse it.")
            bounds, cell_volume, values = self.cached_bounds, self.cached_cell_sizes, self.cached_values

        n_cells = bounds.shape[0]

        if self.equal_cell_topk and self.equal_cell_topk_count_mode == "same":
            n_per_cell = max(1, math.ceil(n_samples / n_cells))
            counts = torch.full((n_cells,), n_per_cell, dtype=torch.long, device=self.device)
            expected_counts = counts.to(torch.float32)
        else:
            counts, expected_counts = adaptive_cell_counts(
                values, n_samples, self.count_floor_mode, self.count_floor_frac
            )

        samples_xyz, cell_ids, ptr = sample_variable_from_3d_intervals_vcounts(bounds, counts, device=self.device)

        # Resolve sampled voxel coordinates to dataset node indices; cells that
        # land entirely on masked-out (e.g. land) voxels contribute nothing.
        index_grid = self._get_index_grid(graph)
        node_idx = index_grid[samples_xyz[:, 0].cpu(), samples_xyz[:, 1].cpu(), samples_xyz[:, 2].cpu()]
        valid = node_idx >= 0
        sampled_idx = node_idx[valid].to(self.device)
        cell_ids = cell_ids[valid.to(self.device)]

        graph_on_device = graph.to(self.device)
        # See the 2D sampler: under "soft" the weight modes must divide by the expected
        # allocation, since cells are allowed to draw zero samples.
        counts_f = expected_counts.to(torch.float32)
        cell_volume_f = cell_volume.to(torch.float32)
        values_f = values.to(torch.float32)

        feats = preds = None
        if self.equal_cell_topk or self.weight_mode in ("sampled_dif", "loss_powered_weight"):
            with torch.no_grad():
                feats = graph_on_device.feat[sampled_idx]
                preds = self.model(graph_on_device.space_emb[sampled_idx])

        if self.equal_cell_topk:
            sampled_dif = (feats - preds).abs().sum(dim=1)
            keep_k = min(n_samples, sampled_dif.numel())
            _, topk_local = torch.topk(sampled_dif, keep_k)
            selected_idx = sampled_idx[topk_local]

            sampled_data_kwargs = dict(
                cor=graph_on_device.cor[selected_idx],
                time=graph_on_device.time[selected_idx],
                feat=graph_on_device.feat[selected_idx],
                space_emb=graph_on_device.space_emb[selected_idx],
            )

            weight_mode = self.equal_cell_topk_weight_mode
            if weight_mode in ("unbiased_weight", "loss_powered_weight"):
                counts_fc = counts_f.clamp_min(1.0)
                cell_volume_fc = cell_volume_f.clamp_min(1.0)
                if weight_mode == "loss_powered_weight":
                    per_point_loss = (feats - preds).pow(2).sum(dim=1)
                    per_cell_loss_sum = torch.zeros(n_cells, device=self.device, dtype=torch.float32)
                    per_cell_loss_sum.index_add_(0, cell_ids, per_point_loss.to(torch.float32))
                    per_cell_mean_loss = per_cell_loss_sum / counts_fc
                    per_cell_weight = (
                        torch.pow(per_cell_mean_loss.clamp_min(0.0), self.power_for_loss_as_weight)
                        * cell_volume_fc / counts_fc
                    )
                else:  # unbiased_weight
                    per_cell_weight = cell_volume_fc / counts_fc
                per_cell_weight = normalize_and_clip_cell_weights(per_cell_weight, self.weight_clip_ratio)
                sampled_data_kwargs["weight"] = per_cell_weight[cell_ids[topk_local]]

            return Data(**sampled_data_kwargs).to(self.device)

        if self.weight_mode == "sampled_dif":
            sampled_weights = normalize_and_clip_cell_weights(
                (feats - preds).abs().sum(dim=1), self.weight_clip_ratio
            )
        elif self.weight_mode == "loss_powered_weight":
            counts_fc = counts_f.clamp_min(1.0)
            cell_volume_fc = cell_volume_f.clamp_min(1.0)
            per_point_loss = (feats - preds).pow(2).sum(dim=1)
            per_cell_loss_sum = torch.zeros(n_cells, device=self.device, dtype=torch.float32)
            per_cell_loss_sum.index_add_(0, cell_ids, per_point_loss.to(torch.float32))
            per_cell_mean_loss = per_cell_loss_sum / counts_fc
            per_cell_weight = (
                torch.pow(per_cell_mean_loss.clamp_min(0.0), self.power_for_loss_as_weight)
                * cell_volume_fc / counts_fc
            )
            sampled_weights = normalize_and_clip_cell_weights(per_cell_weight, self.weight_clip_ratio)[cell_ids]
        elif self.weight_mode == "none":
            sampled_weights = normalize_and_clip_cell_weights(
                torch.ones_like(cell_volume_f), self.weight_clip_ratio
            )[cell_ids]
        elif self.weight_mode == "inverse_value":
            per_cell_weight = cell_volume_f / values_f.clamp_min(self.weight_value_eps)
            sampled_weights = normalize_and_clip_cell_weights(per_cell_weight, self.weight_clip_ratio)[cell_ids]
        elif self.weight_mode == "unbiased_weight":
            per_cell_weight = cell_volume_f.clamp_min(1.0) / counts_f.clamp_min(1.0)
            sampled_weights = normalize_and_clip_cell_weights(per_cell_weight, self.weight_clip_ratio)[cell_ids]
        else:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

        return Data(
            cor=graph_on_device.cor[sampled_idx],
            time=graph_on_device.time[sampled_idx],
            feat=graph_on_device.feat[sampled_idx],
            space_emb=graph_on_device.space_emb[sampled_idx],
            weight=sampled_weights,
        ).to(self.device)


# 2d cluster sampler
def graph_2d_cluster_single_image(graph, n_segments, compactness=1, cluster_type='slic'):
    T = graph.T.sum()

    W, H = graph.cor.max(axis=0)[0] +1
    W = W.item()
    H = H.item()

    graph.T = torch.tensor(1)
    T = graph.T.sum()

    graph.cluster_set = [defaultdict(dict)]
    graph.segments = [defaultdict(dict)]
    for t in range(T):
        image = graph.feat.reshape(W, H)

        if cluster_type == 'slic':
            segments = slic(image,
                            n_segments=n_segments,
                            compactness=compactness,
                            start_label=0,
                            channel_axis=None)
        elif cluster_type == 'grid':
            # Choose grid dimensions to approximate n_segments given aspect ratio
            grid_rows = int(np.sqrt(n_segments * W / H)) or 1
            grid_cols = int(np.ceil(n_segments / grid_rows))

            # Compute row sizes so that differences ≤ 1
            base_row = W // grid_rows
            extra_rows = W % grid_rows
            row_sizes = [base_row + (1 if i < extra_rows else 0) for i in range(grid_rows)]

            # Compute column sizes so that differences ≤ 1
            base_col = H // grid_cols
            extra_cols = H % grid_cols
            col_sizes = [base_col + (1 if j < extra_cols else 0) for j in range(grid_cols)]

            # Assign labels
            segments = np.zeros((W, H), dtype=np.int64)
            label = 0
            r_start = 0
            for i, r_size in enumerate(row_sizes):
                c_start = 0
                for j, c_size in enumerate(col_sizes):
                    r_end = r_start + r_size
                    c_end = c_start + c_size
                    segments[r_start:r_end, c_start:c_end] = label
                    label += 1
                    c_start += c_size
                r_start += r_size
        else:
            raise NotImplementedError(f"Unknown cluster_type: {cluster_type}")

        segments_flat = torch.tensor(segments).reshape(-1)
        for i in range(segments.max() + 1):
            mask = segments_flat == i
            graph.cluster_set[t][i] = mask.nonzero(as_tuple=True)[0]
        graph.segments[t] = segments


class EVOSSampler:
    def __init__(self, cfg, img, graph):
        self.cfg = cfg
        self._st = cfg.sampling.type
        self.use_ratio_scheduler = mt_scheduler_factory(cfg.sampling.sample_num_schedular)
        self.book = {}
        self.num_epochs = cfg.optim.epochs
        self.input_img = img
        self.graph = graph
        self.sample_num = graph.space_emb.shape[0]
        self.C, self.H, self.W = img.shape
        self.transform = Transform(cfg)
        self.device = torch.device(cfg.sampling.device)

    def _reset_rng(self):
        generator = torch.Generator()
        seed = generator.seed()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _recover_rng(self):
        fix_seed(self.cfg.data.seed)

    def _init_sampler(self):
        if self.cfg.sampling.type == "EVOS":
            self._evos_init()

    def _evos_init(self):
        coords, gt = self._get_data()
        coords = coords.to(self.device)
        gt = gt.to(self.device)
        self.input_img = self.input_img.to(self.device)
        if self.cfg.sampling.lap_coff > 0 or self.cfg.sampling.crossover_method != "no":
            self.cached_gt_lap = compute_laplacian(self.input_img).squeeze()
        self.full_coords = coords
        self.full_gt = gt
        self.sample_num = coords.shape[0]

    def _get_cur_use_ratio(self, epoch):
        return self.use_ratio_scheduler(
            epoch, self.num_epochs, self.cfg.sampling.rate
        )
        # TODO: Pass in number of epochs

    def _sampler_get_coords_gt(self, epoch, graph):
        # coords, gt = self.graph.space_emb, self.graph.feat
        coords, gt = self.full_coords, self.full_gt
        # TODO: Pass in coords and gt
        self.cur_use_ratio = self._get_cur_use_ratio(epoch)

        self._reset_rng()

        if self._evos_is_fitness_eval_iter(epoch):
            return coords, gt, None
        else:
            selection_mask = self._evos_get_selection_mask(epoch)
            _coords = self.full_coords[selection_mask]
            _gt = self.full_gt[selection_mask]
            return _coords, _gt, selection_mask

        self._recover_rng()
        return _coords, _gt, selection_mask

    def _sampler_compute_loss(self, pred, gt, epoch):
        _st = self.cfg.sampling.type
        mse = self.compute_mse(pred, gt)
        if self._evos_is_fitness_eval_iter(epoch):

            self._evos_frequency_aware_crossover(pred, gt, epoch) # crossover
            return self._evos_cross_frequency_loss(mse, pred)
        else:
            if self.cfg.sampling.lap_coff <= 0 or epoch > self.cfg.sampling.use_laplace_epoch:
                return mse
            else:
                profile_pred = self.book["freeze_profile_pred"]
                _mask = self.book["freeze_mask"]
                pseudo_full_pred = profile_pred.clone()

                indices = torch.arange(_mask.shape[0], device=pred.device)[~_mask]
                pseudo_full_pred[indices] = pred

                r_img = self.reconstruct_img(pseudo_full_pred)
                lap_loss = (
                    F.mse_loss(
                        compute_laplacian(r_img).squeeze(),
                        self.cached_gt_lap,
                        reduction="none",
                    )
                    .flatten()[~_mask]
                    .mean()
                )

        return mse + self.cfg.sampling.lap_coff * lap_loss


    def _evos_get_mutation_ratio(self, epoch):
        if self.cfg.sampling.mutation_method == "constant":
            return self.cfg.sampling.init_mutation_ratio * self.cfg.sampling.rate
        elif self.cfg.sampling.mutation_method == "linear":
            _start = self.cfg.sampling.init_mutation_ratio
            _end = self.cfg.sampling.end_mutation_ratio  # max = 1
            ratio = _start + ((_end - _start) / self.cfg.optim.epochs) * epoch
            return ratio * self.cfg.sampling.rate
        elif self.cfg.sampling.mutation_method == "exp":
            _start = self.cfg.sampling.init_mutation_ratio
            _end = self.cfg.sampling.end_mutation_ratio
            _lamda = -np.log(_end / _start) / self.cfg.optim.epochs
            ratio = _start * np.exp(-_lamda * epoch)
            return ratio * self.cfg.sampling.rate
        else:
            raise NotImplementedError

    def _evos_get_selection_mask(self, epoch):
        mutation_ratio = self._evos_get_mutation_ratio(epoch)
        first_select_ratio = self.cur_use_ratio - mutation_ratio
        first_select_num = int(first_select_ratio * self.sample_num)

        sorted_map_index = self.book["sorted_map_index"]
        first_select_indices = sorted_map_index[-first_select_num:]

        # Augmented Unbiased Mutation
        # These are additional points to supplement children found from parents,
        # which were non-surviving points
        mutation_num = int(mutation_ratio * self.sample_num)
        remain_indices = sorted_map_index[:-first_select_num]
        sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[
            :mutation_num
        ]
        mutation_indicies = remain_indices[sample_index]

        selected_indices = torch.cat([first_select_indices, mutation_indicies])

        _mask = torch.ones(self.sample_num, dtype=torch.bool, device=self.device)
        _mask[selected_indices] = False
        self.book["freeze_mask"] = _mask

        selection_mask = torch.zeros(
            self.sample_num, dtype=torch.bool, device=self.device
        )
        selection_mask[selected_indices] = True
        return selection_mask

    def _evos_is_fitness_eval_iter(self, epoch):
        _cur_interval = self._evos_get_cur_interval(epoch)
        return epoch % _cur_interval == 1

    def _evos_get_cur_interval(self, epoch):
        if self.cfg.sampling.profile_interval_method == "fixed":
            return self.cfg.sampling.init_interval
        elif self.cfg.sampling.profile_interval_method == "lin_dec":
            _start = self.cfg.sampling.init_interval
            _end = self.cfg.sampling.end_interval
            _cur_interval = _start + ((_end - _start) / self.cfg.optim.epochs) * epoch
            return int(_cur_interval)

    def _evos_frequency_aware_crossover(self, pred, gt, epoch):
        error_map = F.mse_loss(pred, gt, reduction="none").mean(1)
        if self.cfg.sampling.crossover_method == "add":
            r_img = self.reconstruct_img(pred)
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.cfg.sampling.lap_coff if self.cfg.sampling.lap_coff > 0 else 1e-5
            error_map = error_map + cross_lap_coff * laplace_map.flatten()
        elif self.cfg.sampling.crossover_method == "no":
            pass

        if self.cfg.sampling.profile_guide == "value":
            sorted_map_index = torch.argsort(error_map.flatten())
        elif self.cfg.sampling.profile_guide == "diff_1":
            # to deprecated ...
            last_error_map = self.book.get("error_map", None)
            if last_error_map is None:
                last_error_map = torch.zeros_like(error_map)
            guidance_map = torch.abs(error_map - last_error_map)
            sorted_map_index = torch.argsort(guidance_map.flatten())
        else:
            raise NotImplementedError

        self.book["freeze_profile_pred"] = pred.detach()
        self.book["error_map"] = error_map.detach()
        self.book["sorted_map_index"] = sorted_map_index

        if self.cfg.sampling.crossover_method == "select":
            r_img = self.reconstruct_img(pred)
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.cfg.sampling.lap_coff if self.cfg.sampling.lap_coff > 0 else 1e-5
            laplace_error_map = cross_lap_coff * laplace_map.flatten()
            sorted_lap_map_index = torch.argsort(laplace_error_map.flatten())
            self.book["sorted_lap_map_index"] = sorted_lap_map_index

            mutation_ratio = self._evos_get_mutation_ratio(epoch)
            freeze_ratio = 1 - self.cur_use_ratio + mutation_ratio

            freezed_num = int(freeze_ratio * self.sample_num)
            selected_num = self.sample_num - freezed_num

            l2_error_selected_index = sorted_map_index[-selected_num:]
            lap_error_selected_index = sorted_lap_map_index[-selected_num:]
            isin = torch.isin(l2_error_selected_index, lap_error_selected_index)

            selected_index = l2_error_selected_index[isin]

            remain_num = selected_num - selected_index.shape[0]
            l2_remain_index = l2_error_selected_index[~isin]
            isin2 = torch.isin(lap_error_selected_index, l2_error_selected_index)
            lap_remain_index = lap_error_selected_index[~isin2]

            l2_remain_num = int(
                remain_num
                * (error_map.mean() / (laplace_error_map.mean() + error_map.mean()))
            )
            l2_remain_num = min(l2_remain_num, l2_remain_index.shape[0])
            lap_remain_num = remain_num - l2_remain_num
            all_selected_index = torch.cat(
                [
                    lap_remain_index[-lap_remain_num:],
                    l2_remain_index[-l2_remain_num:],
                    selected_index,
                ]
            )
            # Non surviving points, didn't generate children
            all_remain_index = sorted_map_index[
                ~torch.isin(sorted_map_index, all_selected_index)
            ]
            select_sorted_index = torch.cat([all_remain_index, all_selected_index])
            self.book["sorted_map_index"] = select_sorted_index

    def _evos_cross_frequency_loss(self, cur_loss, pred):
        if self.cfg.sampling.lap_coff > 0:
            r_img = self.reconstruct_img(pred)
            lap_loss = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap
            )
            cur_loss += self.cfg.sampling.lap_coff * lap_loss
        return cur_loss

    def compute_mse(self, pred, gt):  # From EVOS base_trainer.py
        #  return torch.mean((pred - gt) ** 2)
        return F.mse_loss(pred, gt)

    def reconstruct_img(self, data) -> torch.tensor:    # From EVOS img_trainer.py
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # c,h,w
        return img

    def _decode_img(self, data):    # From EVOS img_trainer.py
        data = self.transform.inverse(data)
        data = data * 255.0
        data = torch.clamp(data, min=0, max=255)
        return data

    def _parse_input_data(self):
        img = self.input_img.permute(2, 0, 1)  # c,h,w
        self.input_img = img
        self.gt = img
        self.C, self.H, self.W = img.shape

    def _encode_img(self, img):
        img = torch.clamp(img, min=0, max=255)
        img = img / 255.0
        img = self.transform.tranform(img)
        return img

    def _get_data(self):
        img = self.input_img
        # img = self._encode_img(img)
        gt = img.permute(1, 2, 0).reshape(-1, self.C)  # h*w, C
        coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        return coords, gt

    def sample(
        self,
        graph,
        epoch,
        inner_step=1,
        modulations: torch.Tensor=None,
        save_image=False):
        # Get coords function
        # Return the output in Data() structure to fit inr_sampling pipeline
        coords, gt, sel_mask = self._sampler_get_coords_gt(epoch, graph)



        if sel_mask != None:
            graph = Data(
                cor = coords,
                feat = gt,
                time = graph.time[sel_mask],
                space_emb = graph.space_emb[sel_mask],
                T=graph.T,  # global property (total time frames) remains unchanged
                # latent_vector=graph.latent_vector  # global latent vector remains unchanged
            )
        else:
            graph = Data(
                cor = coords,
                feat = gt,
                time = graph.time,
                space_emb = graph.space_emb,
                T=graph.T,  # global property (total time frames) remains unchanged
                # latent_vector=graph.latent_vector  # global latent vector remains unchanged
            )

        return graph


@torch.no_grad()
def sample_counts_poisson(values: torch.Tensor, expected_total: float, eps: float = 1e-12) -> torch.Tensor:
    """
    values: [N] >= 0
    Guarantees at least 1 sample per cell. Reserves one sample per cell as a
    baseline, then distributes the remaining budget via Poisson proportionally
    to cell values.
    """
    if values.dim() != 1:
        raise ValueError("values must be 1D [N].")
    if expected_total < 0:
        raise ValueError("expected_total must be >= 0.")

    n_cells = values.numel()
    v = values.clamp_min(0)
    s = v.sum()

    # If budget doesn't exceed the per-cell minimum, just give 1 to each cell.
    remaining = expected_total - n_cells
    if remaining <= 0 or s <= eps:
        return torch.ones(n_cells, dtype=torch.long, device=values.device)

    lam = (remaining * v) / s   # [N], extra samples distributed proportionally
    counts = torch.poisson(lam).to(torch.long) + 1  # +1 guarantees minimum of 1
    return counts


# ============================================================================
# Fixed-grid stratified sampling (proportional / Neyman allocation, unbiased)
#
# Textbook stratified sampling over a *fixed* uniform grid, used as the baseline
# that isolates how much of ACES's benefit comes from adaptive partitioning as
# opposed to stratification plus optimal allocation. Shared by the 2D and 3D
# samplers; see `2d_grid_stratified` / `3d_grid_stratified`.
# ============================================================================

def stratified_continuous_allocation(
    scores: torch.Tensor,
    n_total: float,
    min_alloc_frac: float = 0.1,
    max_iters: int = 50,
) -> torch.Tensor:
    """Continuous per-cell allocation proportional to `scores`, floored and renormalised.

    Neyman allocation is ``ñ_h = n · N_h σ_h / Σ_j N_j σ_j`` (pass ``scores = N_h σ_h``);
    proportional allocation is the same with ``scores = N_h``. Cells whose share falls
    below ``min_alloc_frac · (n / H)`` are raised to that floor and the remaining budget
    is rescaled over the unconstrained cells (water-filling), so ``Σ ñ_h = n`` still holds.

    The floor bounds the Horvitz-Thompson weight blow-up: since each point is weighted
    ``N_h / ñ_h``, a cell with a vanishing ñ_h would otherwise contribute a single point
    with an enormous weight and dominate the batch. The floor is *deterministic*, so it
    costs a little Neyman optimality but does not introduce any bias.

    Args:
        scores: (H,) non-negative per-cell allocation scores.
        n_total: total expected sample budget n.
        min_alloc_frac: floor as a fraction of the uniform share n/H; <=0 disables it.
        max_iters: water-filling iterations (converges monotonically, typically in 2-3).

    Returns:
        (H,) float tensor of continuous allocations summing to `n_total`.
    """
    n_cells = scores.numel()
    uniform = float(n_total) / max(n_cells, 1)
    s = scores.to(torch.float64).clamp_min(0.0)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    if not torch.isfinite(s.sum()) or s.sum() <= 0:
        return torch.full_like(scores, uniform, dtype=torch.float32)

    alloc = float(n_total) * s / s.sum()

    floor_val = max(float(min_alloc_frac), 0.0) * uniform
    if floor_val > 0:
        for _ in range(max_iters):
            below = alloc < floor_val
            if not bool(below.any()):
                break
            free = ~below
            remaining = float(n_total) - float(below.sum()) * floor_val
            free_mass = alloc[free].sum()
            if remaining <= 0 or free_mass <= 0:
                alloc = torch.full_like(alloc, uniform)
                break
            alloc = torch.where(below, torch.full_like(alloc, floor_val),
                                alloc * (remaining / free_mass))
        alloc = alloc.clamp_min(floor_val)

    return alloc.to(torch.float32)


def stratified_poisson_counts(alloc: torch.Tensor) -> torch.Tensor:
    """Draw ``n_h ~ Poisson(ñ_h)`` independently per cell.

    Deliberately *not* `sample_counts_poisson`, which reserves one sample per cell and
    therefore realises ``E[n_h] = 1 + λ_h``. Unbiasedness of the ``N_h / ñ_h`` weighting
    requires ``E[n_h] = ñ_h`` exactly, so a plain Poisson draw is what is needed here;
    cells that draw zero samples are fine and need no special handling.
    """
    return torch.poisson(alloc.clamp_min(0.0)).to(torch.long)


def adaptive_cell_counts(values, expected_total, floor_mode="min_one", floor_frac=0.1):
    """Per-cell sample counts for the adaptive samplers, plus the expected allocation.

    Two schemes:

    ``min_one`` (default, the published ACES behaviour)
        ``sample_counts_poisson``: reserve one sample per cell, then distribute the
        remainder by Poisson. The reserve acts as an exploration floor, guaranteeing every
        leaf is represented every step. Its drawback is that it cannot honour a budget
        smaller than the number of cells -- with ``n <= n_cells`` it degenerates to one
        sample per cell and the batch becomes ``n_cells`` regardless of the budget. Even
        well inside the budget it is expensive: with 226 leaves and a budget of 294, 77%
        of the budget goes to the reserve and only 23% is allocated by score.

    ``soft``
        Floor the *expected* allocation instead of the realised one:
        ``n~_h = max(n * v_h / sum(v), floor_frac * n / H)`` renormalised to sum to ``n``
        (water-filling), then ``n_h ~ Poisson(n~_h)``. The budget is respected exactly for
        any number of cells, coverage is maintained in expectation rather than per draw,
        and the score-driven share of the budget is not eaten by the reserve. Cells may
        draw zero samples on a given step, so weight modes should divide by the expected
        allocation (returned here) rather than the realised count.

    Returns:
        (counts, expected_counts): realised integer counts, and the expected allocation
        each cell was drawn against.
    """
    if floor_mode == "min_one":
        counts = sample_counts_poisson(values, expected_total=expected_total)
        # Existing behaviour divides weights by the realised count; preserve that exactly.
        return counts, counts.to(torch.float32)
    if floor_mode == "soft":
        alloc = stratified_continuous_allocation(values, expected_total, floor_frac)
        counts = stratified_poisson_counts(alloc)
        if int(counts.sum().item()) == 0:  # degenerate draw; keep at least one sample
            counts[torch.argmax(alloc)] = 1
        return counts, alloc
    raise ValueError(
        f"Unknown count floor mode '{floor_mode}'. Expected 'min_one' or 'soft'."
    )


def build_uniform_grid_bounds_2d(image_width: int, n_bins: int, device='cpu') -> torch.Tensor:
    """Uniform n_bins x n_bins partition as (H, 4) bounds [x_low, x_high, y_low, y_high]."""
    bounds_1d = generate_equal_bins(0, image_width - 1, n_bins, device=device)
    x_low = bounds_1d[:, 0].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
    x_high = bounds_1d[:, 1].unsqueeze(1).expand(n_bins, n_bins).reshape(-1)
    y_low = bounds_1d[:, 0].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
    y_high = bounds_1d[:, 1].unsqueeze(0).expand(n_bins, n_bins).reshape(-1)
    return torch.stack([x_low, x_high, y_low, y_high], dim=1)


def build_uniform_grid_bounds_3d(grid_shape, n_bins, device='cpu') -> torch.Tensor:
    """Uniform partition of a (Dx, Dy, Dz) volume as (H, 6) bounds.

    `n_bins` may be a scalar (same count on every axis) or a length-3 sequence; volumes
    here are not cubic (e.g. 384 x 384 x 64), so per-axis counts are usually wanted.
    """
    Dx, Dy, Dz = (int(v) for v in grid_shape)
    if isinstance(n_bins, (int, float)):
        nx = ny = nz = int(n_bins)
    else:
        n_list = [int(v) for v in n_bins]
        if len(n_list) != 3:
            raise ValueError(f"3D n_bins must be a scalar or length-3 sequence, got {n_bins}")
        nx, ny, nz = n_list
    nx, ny, nz = max(1, min(nx, Dx)), max(1, min(ny, Dy)), max(1, min(nz, Dz))

    return cross_bins_3d(
        generate_equal_bins(0, Dx - 1, nx, device=device),
        generate_equal_bins(0, Dy - 1, ny, device=device),
        generate_equal_bins(0, Dz - 1, nz, device=device),
    )


def sample_multiple_from_2d_intervals(bounds, n_samples, device='cuda'):
    """
    Sample multiple 2D coordinate pairs from each 2D interval.
    Optimized for large n_samples.
    
    Args:
        bounds: tensor of shape (n_cells, 4) where each row is [x_low, x_high, y_low, y_high]
        n_samples: number of samples per cell
        cell_sizes: (deprecated) kept for backward compatibility
        device: device to run on
    
    Returns:
        tensor of shape (n_cells, n_samples, 2) containing (x, y) coordinates
    """
    bounds = bounds.to(device)
    n_cells = bounds.size(0)

    # Calculate ranges for each cell (add 1 for inclusive sampling)
    x_range = bounds[:, 1] - bounds[:, 0] + 1  # (n_cells,)
    y_range = bounds[:, 3] - bounds[:, 2] + 1  # (n_cells,)

    # Generate random values directly in the target shape
    rand_vals = torch.rand(n_cells, n_samples, 2, device=device, dtype=torch.float32)

    # Vectorized sampling using broadcasting
    # rand_vals[:, :, 0] for x, rand_vals[:, :, 1] for y
    x_samples = bounds[:, 0:1] + torch.floor(rand_vals[:, :, 0] * x_range.unsqueeze(1))
    y_samples = bounds[:, 2:3] + torch.floor(rand_vals[:, :, 1] * y_range.unsqueeze(1))

    # Stack efficiently without intermediate tensors
    samples = torch.stack([x_samples, y_samples], dim=2).long()

    return samples


@torch.no_grad()
def sample_variable_from_2d_intervals_vcounts(bounds: torch.Tensor,
                                      counts: torch.Tensor,
                                      device: str = "cuda"):
    """
    Variable number of integer (x, y) samples per cell, sampled uniformly from inclusive 2D box bounds.

    Args:
        bounds: (n_cells, 4) each row [x_low, x_high, y_low, y_high] (integer-like)
        counts: (n_cells,) number of samples for each cell (int/long), can be zero
        device: 'cuda' or 'cpu'

    Returns:
        samples: (total_samples, 2) long tensor, packed samples
        cell_ids: (total_samples,) long tensor, indicates which cell each sample belongs to
        ptr: (n_cells+1,) long tensor, ptr[i]: start index of cell i in samples, ptr[i+1] end
             So samples[ptr[i]:ptr[i+1]] are samples from cell i.
    """
    bounds = bounds.to(device)
    counts = counts.to(device=device, dtype=torch.long)

    if bounds.dim() != 2 or bounds.size(1) != 4:
        raise ValueError("bounds must have shape (n_cells, 4).")
    if counts.dim() != 1 or counts.numel() != bounds.size(0):
        raise ValueError("counts must have shape (n_cells,).")
    if (counts < 0).any():
        raise ValueError("counts must be >= 0.")

    n_cells = bounds.size(0)
    total = int(counts.sum().item())

    # ptr for slicing back per cell
    ptr = torch.zeros(n_cells + 1, device=device, dtype=torch.long)
    if n_cells > 0:
        ptr[1:] = torch.cumsum(counts, dim=0)

    if total == 0:
        samples = torch.empty((0, 2), device=device, dtype=torch.long)
        cell_ids = torch.empty((0,), device=device, dtype=torch.long)
        return samples, cell_ids, ptr

    # Build cell_ids without Python loops
    cell_ids = torch.repeat_interleave(torch.arange(n_cells, device=device, dtype=torch.long), counts)

    # Precompute ranges (inclusive)
    x_low = bounds[:, 0].to(torch.long)
    x_high = bounds[:, 1].to(torch.long)
    y_low = bounds[:, 2].to(torch.long)
    y_high = bounds[:, 3].to(torch.long)

    x_range = (x_high - x_low + 1).clamp_min(1)  # avoid non-positive
    y_range = (y_high - y_low + 1).clamp_min(1)

    # Gather per-sample lows and ranges
    x_low_s = x_low[cell_ids]
    y_low_s = y_low[cell_ids]
    x_rng_s = x_range[cell_ids]
    y_rng_s = y_range[cell_ids]

    # Randoms in [0,1)
    r = torch.rand((total, 2), device=device, dtype=torch.float32)

    # Integer uniform in inclusive box
    x = x_low_s + torch.floor(r[:, 0] * x_rng_s.to(torch.float32)).to(torch.long)
    y = y_low_s + torch.floor(r[:, 1] * y_rng_s.to(torch.float32)).to(torch.long)

    samples = torch.stack((x, y), dim=1)
    return samples, cell_ids, ptr


def sample_multiple_from_3d_intervals(bounds, n_samples, device='cuda'):
    """
    Sample multiple 3D coordinate triples from each 3D interval (box).
    3D analog of sample_multiple_from_2d_intervals.

    Args:
        bounds: tensor of shape (n_cells, 6), each row [x_low, x_high, y_low, y_high, z_low, z_high]
        n_samples: number of samples per cell
        device: device to run on

    Returns:
        tensor of shape (n_cells, n_samples, 3) containing (x, y, z) coordinates
    """
    bounds = bounds.to(device)
    n_cells = bounds.size(0)

    x_range = bounds[:, 1] - bounds[:, 0] + 1
    y_range = bounds[:, 3] - bounds[:, 2] + 1
    z_range = bounds[:, 5] - bounds[:, 4] + 1

    rand_vals = torch.rand(n_cells, n_samples, 3, device=device, dtype=torch.float32)

    x_samples = bounds[:, 0:1] + torch.floor(rand_vals[:, :, 0] * x_range.unsqueeze(1))
    y_samples = bounds[:, 2:3] + torch.floor(rand_vals[:, :, 1] * y_range.unsqueeze(1))
    z_samples = bounds[:, 4:5] + torch.floor(rand_vals[:, :, 2] * z_range.unsqueeze(1))

    samples = torch.stack([x_samples, y_samples, z_samples], dim=2).long()
    return samples


@torch.no_grad()
def sample_variable_from_3d_intervals_vcounts(bounds: torch.Tensor,
                                      counts: torch.Tensor,
                                      device: str = "cuda"):
    """
    Variable number of integer (x, y, z) samples per cell, sampled uniformly from
    inclusive 3D box bounds. 3D analog of sample_variable_from_2d_intervals_vcounts.

    Args:
        bounds: (n_cells, 6) each row [x_low, x_high, y_low, y_high, z_low, z_high] (integer-like)
        counts: (n_cells,) number of samples for each cell (int/long), can be zero
        device: 'cuda' or 'cpu'

    Returns:
        samples: (total_samples, 3) long tensor, packed samples
        cell_ids: (total_samples,) long tensor, indicates which cell each sample belongs to
        ptr: (n_cells+1,) long tensor, ptr[i]: start index of cell i in samples, ptr[i+1] end
    """
    bounds = bounds.to(device)
    counts = counts.to(device=device, dtype=torch.long)

    if bounds.dim() != 2 or bounds.size(1) != 6:
        raise ValueError("bounds must have shape (n_cells, 6).")
    if counts.dim() != 1 or counts.numel() != bounds.size(0):
        raise ValueError("counts must have shape (n_cells,).")
    if (counts < 0).any():
        raise ValueError("counts must be >= 0.")

    n_cells = bounds.size(0)
    total = int(counts.sum().item())

    ptr = torch.zeros(n_cells + 1, device=device, dtype=torch.long)
    if n_cells > 0:
        ptr[1:] = torch.cumsum(counts, dim=0)

    if total == 0:
        samples = torch.empty((0, 3), device=device, dtype=torch.long)
        cell_ids = torch.empty((0,), device=device, dtype=torch.long)
        return samples, cell_ids, ptr

    cell_ids = torch.repeat_interleave(torch.arange(n_cells, device=device, dtype=torch.long), counts)

    x_low = bounds[:, 0].to(torch.long)
    x_high = bounds[:, 1].to(torch.long)
    y_low = bounds[:, 2].to(torch.long)
    y_high = bounds[:, 3].to(torch.long)
    z_low = bounds[:, 4].to(torch.long)
    z_high = bounds[:, 5].to(torch.long)

    x_range = (x_high - x_low + 1).clamp_min(1)
    y_range = (y_high - y_low + 1).clamp_min(1)
    z_range = (z_high - z_low + 1).clamp_min(1)

    x_low_s = x_low[cell_ids]
    y_low_s = y_low[cell_ids]
    z_low_s = z_low[cell_ids]
    x_rng_s = x_range[cell_ids]
    y_rng_s = y_range[cell_ids]
    z_rng_s = z_range[cell_ids]

    r = torch.rand((total, 3), device=device, dtype=torch.float32)

    x = x_low_s + torch.floor(r[:, 0] * x_rng_s.to(torch.float32)).to(torch.long)
    y = y_low_s + torch.floor(r[:, 1] * y_rng_s.to(torch.float32)).to(torch.long)
    z = z_low_s + torch.floor(r[:, 2] * z_rng_s.to(torch.float32)).to(torch.long)

    samples = torch.stack((x, y, z), dim=1)
    return samples, cell_ids, ptr


def cross_bins_3d(x_bounds, y_bounds, z_bounds):
    """
    Cartesian-cross 1D bin bounds along x, y, z into full 3D box bounds.

    Args:
        x_bounds, y_bounds, z_bounds: (nx, 2), (ny, 2), (nz, 2) tensors of [low, high] per axis

    Returns:
        tensor of shape (nx*ny*nz, 6): [x_low, x_high, y_low, y_high, z_low, z_high]
    """
    nx, ny, nz = x_bounds.size(0), y_bounds.size(0), z_bounds.size(0)
    x_low = x_bounds[:, 0].view(nx, 1, 1).expand(nx, ny, nz).reshape(-1)
    x_high = x_bounds[:, 1].view(nx, 1, 1).expand(nx, ny, nz).reshape(-1)
    y_low = y_bounds[:, 0].view(1, ny, 1).expand(nx, ny, nz).reshape(-1)
    y_high = y_bounds[:, 1].view(1, ny, 1).expand(nx, ny, nz).reshape(-1)
    z_low = z_bounds[:, 0].view(1, 1, nz).expand(nx, ny, nz).reshape(-1)
    z_high = z_bounds[:, 1].view(1, 1, nz).expand(nx, ny, nz).reshape(-1)
    return torch.stack([x_low, x_high, y_low, y_high, z_low, z_high], dim=1)


def generate_equal_bins(low, high, n_bins, device='cuda'):
    """
    Generate bin bounds with similar widths (max difference = 1).
    
    Args:
        low: lower bound of the range
        high: upper bound of the range (inclusive)
        n_bins: number of bins to create
        device: device to run on
    
    Returns:
        tensor of shape (n_bins, 2) containing [start, end] for each bin
    """
    device = torch.device(device)

    # Total range (inclusive)
    total_range = high - low + 1

    # Base width for each bin
    base_width = total_range // n_bins

    # Number of bins that need one extra element
    remainder = total_range % n_bins

    # Create bin bounds
    bounds = torch.zeros(n_bins, 2, dtype=torch.long, device=device)

    current_pos = low
    for i in range(n_bins):
        # First 'remainder' bins get base_width + 1, others get base_width
        width = base_width + (1 if i < remainder else 0)

        bounds[i, 0] = current_pos  # start of bin
        bounds[i, 1] = current_pos + width - 1  # end of bin (inclusive)

        current_pos += width

    return bounds


def _resolve_stratified_n_bins(cfg):
    """Read `sampling.stratified_n_bins`, allowing a scalar or a length-3 list (3D per-axis)."""
    n_bins = cfg.sampling.get("stratified_n_bins", 16)
    if isinstance(n_bins, (int, float)):
        return int(n_bins)
    return [int(v) for v in n_bins]


def create_inr_sampler(cfg, inr, graph, current_date_str, run_name, device='cuda'):
    """
    Build and return an INRSingle2dSamplerWrapper, INRSingle3dSamplerWrapper, or
    EVOSSampler based on cfg.sampling settings, or None if no sampling type is
    specified.
    """
    sampling_type = cfg.sampling.type

    if sampling_type is None:
        return None

    save_path = f'./sampled_frames/{current_date_str + run_name}'

    # Volumetric (3D) datasets (e.g. SOMA) are not fully dense, so grid_shape is
    # derived per-axis rather than assuming a single square image_width.
    if graph.cor.shape[1] == 3:
        grid_shape = tuple((graph.cor.max(dim=0)[0] + 1).tolist())
        if sampling_type == "3d_grid_adaptive":
            return INRSingle3dAdaptiveSamplerWrapper(
                model=inr,
                iters=0,
                device=device,
                sample_rate=cfg.sampling.rate,
                mode=cfg.sampling.get("adaptive_mode", "loss"),
                weight_mode=cfg.sampling.adaptive_weight_mode,
                weight_value_eps=cfg.sampling.get("adaptive_weight_value_eps", 1e-6),
                weight_clip_ratio=cfg.sampling.get("adaptive_weight_clip_ratio", 10.0),
                equal_cell_topk=cfg.sampling.get("adaptive_equal_cell_topk", False),
                equal_cell_topk_count_mode=cfg.sampling.get("adaptive_equal_cell_topk_count_mode", "same"),
                equal_cell_topk_weight_mode=cfg.sampling.get("adaptive_equal_cell_topk_weight_mode", "none"),
                power_for_loss_as_weight=cfg.sampling.get("power_for_loss_as_weight", 0.2),
                grid_update_interval=cfg.sampling.get("adaptive_grid_update_interval", 100),
                adaptive_iterations=cfg.sampling.get("adaptive_iterations", 8),
                subdivision_percentage=cfg.sampling.get("subdivision_percentage", 20.0),
                count_floor_mode=cfg.sampling.get("adaptive_count_floor_mode", "min_one"),
                count_floor_frac=cfg.sampling.get("adaptive_count_floor_frac", 0.1),
                initial_grid_size=cfg.sampling.get("adaptive_initial_grid_size", 8),
                save_samples_path=save_path,
                grid_shape=grid_shape,
            )

        if sampling_type not in ("random", "NMT", "3d_grid_linear", "3d_grid_fixed",
                                 "3d_grid_stratified"):
            raise NotImplementedError(
                f"Sampling type '{sampling_type}' is not implemented for 3D volumetric datasets."
            )

        return INRSingle3dSamplerWrapper(
            model=inr,
            iters=0,
            device=device,
            sample_rate=cfg.sampling.rate,
            sample_type=sampling_type,
            save_samples_path=save_path,
            grid_shape=grid_shape,
            cell_size=cfg.sampling.get("fixed_cell_size", 32),
            k_per_cell=cfg.sampling.get("k_per_cell", 1),
            n_bins_3d=cfg.sampling.get("n_bins_3d", 8),
            stratified_allocation=cfg.sampling.get("stratified_allocation", "neyman"),
            stratified_n_bins=_resolve_stratified_n_bins(cfg),
            stratified_min_alloc_frac=cfg.sampling.get("stratified_min_alloc_frac", 0.1),
            stratified_update_interval=cfg.sampling.get("stratified_update_interval", 100),
            stratified_pilot_per_cell=cfg.sampling.get("stratified_pilot_per_cell", 16),
        )

    image_width = graph.cor.max().item() + 1  # Set image width from space_emb shape

    # Map special 2d_cluster types to a unified sampler_name + cluster_type
    # cluster_map = {
    #     '2d_cluster_slic': 'slic',
    #     '2d_grid_linear': 'grid',
    # }
    if sampling_type == "2d_grid_adaptive":
        return INRSingle2dAdaptiveSamplerWrapper(
            model=inr,
            iters=0,
            device=device,
            sample_rate=cfg.sampling.rate,
            mode=cfg.sampling.get("adaptive_mode", "loss"),
            weight_mode=cfg.sampling.adaptive_weight_mode,
            weight_value_eps=cfg.sampling.get("adaptive_weight_value_eps", 1e-6),
            weight_clip_ratio=cfg.sampling.get("adaptive_weight_clip_ratio", 10.0),
            equal_cell_topk=cfg.sampling.get("adaptive_equal_cell_topk", False),
            equal_cell_topk_count_mode=cfg.sampling.get("adaptive_equal_cell_topk_count_mode", "same"),
            equal_cell_topk_weight_mode=cfg.sampling.get("adaptive_equal_cell_topk_weight_mode", "none"),
            power_for_loss_as_weight=cfg.sampling.get("power_for_loss_as_weight", 0.2),
            grid_update_interval=cfg.sampling.get("adaptive_grid_update_interval", 100),
            adaptive_iterations=cfg.sampling.get("adaptive_iterations", 8),
            subdivision_percentage=cfg.sampling.get("subdivision_percentage", 20.0),
            count_floor_mode=cfg.sampling.get("adaptive_count_floor_mode", "min_one"),
            count_floor_frac=cfg.sampling.get("adaptive_count_floor_frac", 0.1),
            save_samples_path=save_path,
            image_width=image_width,
        )

    if sampling_type == "EVOS":
        H = int(np.sqrt(len(graph.feat)))
        img = graph.feat.reshape(H, H)
        img = img.unsqueeze(0)
        return EVOSSampler(cfg, img, graph)

    # if sampling_type in cluster_map:
        # cluster_type = cluster_map[sampling_type]
        # Run your graph clustering side-effect for a single image
        # _start = cfg.sampling.n_clusters_2d_start
        # graph_2d_cluster_single_image(graph, _start, 0.01, cluster_type)


    return INRSingle2dSamplerWrapper(
        model=inr,
        iters=0,
        device=device,
        sample_rate=cfg.sampling.rate,
        sample_type=sampling_type,
        use_weight_function=cfg.sampling.get("use_weight_function", True),
        save_samples_path=save_path,
        n_clusters_2d_start=cfg.sampling.n_clusters_2d_start,
        n_clusters_2d_end=cfg.sampling.n_clusters_2d_end,
        epochs=cfg.optim.epochs,
        image_width=image_width,
        cell_size=cfg.sampling.get("fixed_cell_size", 32),
        k_per_cell=cfg.sampling.get("k_per_cell", 1),
        stratified_allocation=cfg.sampling.get("stratified_allocation", "neyman"),
        stratified_n_bins=_resolve_stratified_n_bins(cfg),
        stratified_min_alloc_frac=cfg.sampling.get("stratified_min_alloc_frac", 0.1),
        stratified_update_interval=cfg.sampling.get("stratified_update_interval", 500),
        stratified_pilot_per_cell=cfg.sampling.get("stratified_pilot_per_cell", 16),
    )
