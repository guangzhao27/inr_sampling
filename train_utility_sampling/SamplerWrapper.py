import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from collections import defaultdict
import math
import torch.nn.functional as F

from utils.data.unstructure_dataset import get_graph_t_idx
from util.misc import fix_seed
from components.laplacian import compute_laplacian_loss as compute_laplacian
from components.nmt import mt_scheduler_factory
from components.transform import Transform
from utils.quadtree import HierarchicalImageGrid, ImageCell
from train_utility_sampling.taylor_estimation import (
    grad_variance_ground_truth,
    cell_grad_variance_estimate_with_norm_corrected,
    cell_grad_variance_estimate_with_jacrev,
    loss_variance_ground_truth,
)


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
        sample_type (str): Type of sampling strategy ("random", "NMT", "3d_cluster"). Defaults to "random".
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
        sample_rate: float = 0.5,
        save_samples_path: Path = Path("logs/sampling"),
        save_interval: int = 100,
        image_width: int = 512,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.sample_type = sample_type
        self.sample_rate = sample_rate
        self.iters = iters

        if sample_type == "2d_grid_linear":
            self.n_clusters_2d_start = n_clusters_2d_start
            self.n_clusters_2d_end = n_clusters_2d_end
            self.epochs = epochs

        self.save_interval = save_interval
        self.save_samples_path = save_samples_path
        self.image_width = image_width


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
            # print("Chosen: " + str(chosen.device))
            # print("idx_tensor: " + str(idx_tensor.device))
            # print("perm: " + str(perm.device))
            # print("offset: " + str(offset.device))
            # chosen.to(device)
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
            cor = sample_multiple_from_2d_intervals(bounds, n_per_cell, device='cpu')  
            
            x_coords = cor[:, :, 0]  # x-coordinates
            y_coords = cor[:, :, 1]  # y-coordinates
            rough_idx = (y_coords * self.image_width + x_coords).flatten().to(self.device)
            space_emb = graph.space_emb[rough_idx].to(self.device)  # [N_rough, D]
            feats = graph.feat[rough_idx].to(self.device)  # [N_rough, F]
            # print(" time for 2d_cluster sampling rough idx: ", str(time()-t0))
            # t0 = time()
            with torch.no_grad():
                preds = self.model(space_emb)
                dif = torch.sum((feats - preds).abs(), dim=1)
            n_samples = min(n_samples, len(dif))
            _, topk_local = torch.topk(dif, n_samples)
            sampled_idx = rough_idx[topk_local]
            # print("time for 2d_cluster sampling: ", str(cal_time))
            
        elif self.sample_type == "2d_grid_adaptive":
            
            grid = HierarchicalImageGrid(1024, 1024, initial_grid_size=32)
            evaluation_function = lambda x: cell_grad_variance_estimate_with_jacrev(x, graph, self.model, self.device)
            grid.iterative_subdivision(evaluation_function, iterations=5, percentage=11)
            bounds, cell_size, _ = grid.get_leaf_properties_tensor()
            cell_num = bounds.shape[0]
            n_per_cell = max(1, math.ceil(n_samples / cell_num))
            cor = sample_multiple_from_2d_intervals(bounds, n_per_cell)
            x_coords = cor[:, :, 0]  # x-coordinates
            y_coords = cor[:, :, 1]  # y-coordinates
            rough_idx = (y_coords * self.image_width + x_coords).flatten().to(self.device)
            # space_emb = graph.space_emb[rough_idx].to(self.device)  # [N_rough, D]
            # feats = graph.feat[rough_idx].to(self.device)  # [N_rough, F]
            # print(" time for 2d_cluster sampling rough idx: ", str(time()-t0))
            # t0 = time()
            # with torch.no_grad():
            #     preds = self.model(space_emb)
            #     dif = torch.sum((feats - preds).abs(), dim=1)
            # n_samples = min(n_samples, len(dif))
            # _, topk_local = torch.topk(dif, n_samples)
            # random select n_samples from rough_idx
            sampled_idx = rough_idx[torch.randperm(len(rough_idx), device=self.device)[:n_samples]]
            
            # sampled_idx = rough_idx[topk_local]
            
            
            # pass
            # if inner_step % 1 == 0:
            #     graph = graphtreebuilder_2d_adaptive_single_image()
            # n_per_cell = max(1, math.ceil(n_samples / len(graph.cluster_set[0])))

            # cor = sample_multiple_from_2d_intervals(bounds, n_per_cell, device='cpu')
            # graph = graph.update_with_samples(cor)
            
            # and then add weight to each sample, add weight to graph. 
            

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
        sampled_graph = Data(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
        )

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
        
        # Store graph reference for evaluation function
        self.cached_graph = None
    
    
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
            
            if mode == 'losstrue':
                loss_variance = loss_variance_ground_truth(
                    cell_coords_tensor, graph, self.model, self.device
                )
                
                cell_value = loss_variance.sqrt() * cell_areas_tensor
                return cell_value.tolist()
        
        return evaluate_cells
    
    
    def sample(
        self, 
        inner_step: int,
        graph: Data,
        save_image: bool = False,
        mode: str = 'loss',
        ) -> Data:
        """
        Adaptive sampling: decide cell sizes based on gradient variance estimation,
        then sample coordinates from each cell.
        
        Args:
            inner_step: Current training step
            graph: Input graph with spatial data
            save_image: Whether to save visualization
            
        Returns:
            Sampled graph data
        """
        n_t = graph.cor.shape[0]
        n_samples = max(int(n_t * self.sample_rate), 1)
        
        # Create adaptive grid with gradient-based subdivision
        
        grid = HierarchicalImageGrid(self.image_width, self.image_width, initial_grid_size=16)
        
        # Create evaluation function with graph context
        eval_fn = self._create_evaluation_function(graph, mode=mode)
        
        # Perform adaptive subdivision based on gradient variance
        grid.iterative_subdivision(
            eval_fn, 
            iterations=5, 
            percentage=10, 
            batch_mode=True
        )
        
        # Get cell boundaries and sample from each cell
        bounds, cell_sizes, values = grid.get_leaf_properties_tensor(evaluation_function=eval_fn, device=self.device)
        
        counts = sample_counts_poisson(values, expected_total=n_samples)
        n_cells = bounds.shape[0]
        n_per_cell = max(1, int(np.ceil(n_samples / n_cells)))
        
        # Sample coordinates from cells (bounds are now inclusive)
        samples_xy, cell_ids, ptr = sample_variable_from_2d_intervals_vcounts(bounds, counts, device=self.device)
        
        # Convert (x,y) -> flat index
        x_coords = samples_xy[:, 0]  # (total_samples,)
        y_coords = samples_xy[:, 1]  # (total_samples,)
        sampled_idx = y_coords * self.image_width + x_coords  # (total_samples,)

        # Create per-sample weights:
        # Weight per sample in cell i: cell_area / n_samples_from_cell_i
        # Here cell_area = cell_sizes[i], n_samples_from_cell_i = counts[i]
        # Use gather via cell_ids
        counts_f = counts.to(torch.float32)
        cell_sizes_f = cell_sizes.to(torch.float32)

        # Avoid division by zero (even though samples only exist where counts>0, this is safer)
        per_cell_weight = cell_sizes_f / values  
        # proportional weights, values is the cell probability, cell_sizes_f is the cell area
        per_cell_weight = per_cell_weight / torch.sum(per_cell_weight) / n_samples
        sampled_weights = per_cell_weight[cell_ids]         # (total_samples,)
        graph.to(self.device)
        sampled_graph = Data(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
            weight=sampled_weights,
        )
        
        if save_image:
            self.save_image_path = os.path.join(
                self.save_samples_path, f"adaptive_i{inner_step}"
            )
            self._save_sample_images(graph, sampled_graph)
        
        return sampled_graph.to(self.device)


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
        # print("---coords---\n" + str(coords) + "\n---full_gt---\n" + str(gt))
        # TODO: Pass in coords and gt
        self.cur_use_ratio = self._get_cur_use_ratio(epoch)

        self._reset_rng()
 
        if self._evos_is_fitness_eval_iter(epoch):
            # print("===Fitness Epoch===\nEpoch: " + str(epoch))
            return coords, gt, None
        else:
            selection_mask = self._evos_get_selection_mask(epoch)
            _coords = self.full_coords[selection_mask]
            _gt = self.full_gt[selection_mask]
            # print("***Not Fitness Epoch, Selected Points***\nEpoch" + str(epoch))
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
            # print("===pred===\n" + str(pred))
            r_img = self.reconstruct_img(pred)
            # print("devices---->r_img: " + str(r_img.device) + " cached_gt_lap: " + str(self.cached_gt_lap.device))
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
        # print("===in reconstruct===\n" + str(data))
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # c,h,w
        # print("===pre decode img===\n" + str(img))
        # img = self._decode_img(img)
        # print("===post decode img===\n" + str(img))
        return img

    def _decode_img(self, data):    # From EVOS img_trainer.py
        data = self.transform.inverse(data)
        # print("===transform_inverse_data===\n" + str(data))
        data = data * 255.0
        data = torch.clamp(data, min=0, max=255)
        return data

    def _parse_input_data(self):
        img = self.input_img.permute(2, 0, 1)  # c,h,w
        # print("---img---\n" + str(img))
        self.input_img = img
        self.gt = img
        self.C, self.H, self.W = img.shape
        # print("---c---\n" + str(self.C))
        # print("\n---h---\n" + str(self.H))
        # print("\n---w---\n" + str(self.W))

    def _encode_img(self, img):
        # print("---img before encode---\n" + str(img))
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
    """
    if values.dim() != 1:
        raise ValueError("values must be 1D [N].")
    if expected_total < 0:
        raise ValueError("expected_total must be >= 0.")

    v = values.clamp_min(0)
    s = v.sum()
    if s <= eps:
        return torch.zeros_like(v, dtype=torch.long)

    lam = (expected_total * v) / s  # [N]
    counts = torch.poisson(lam)     # float tensor, integer-valued
    return counts.to(torch.long)


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


def create_inr_sampler(cfg, inr, graph, current_date_str, run_name, device='cuda'):
    """
    Build and return an INRSingle2dSamplerWrapper or EVOSSampler based on cfg.sampling 
    settings, or None if no sampling type is specified.
    """
    sampling_type = cfg.sampling.type
    image_width = graph.cor.max().item() + 1  # Set image width from space_emb shape
    
    if sampling_type is None:
        return None

    # Map special 2d_cluster types to a unified sampler_name + cluster_type
    # cluster_map = {
    #     '2d_cluster_slic': 'slic',
    #     '2d_grid_linear': 'grid',
    # }
    save_path = f'./sampled_frames/{current_date_str + run_name}'
    if sampling_type == "2d_grid_adaptive":
        return INRSingle2dAdaptiveSamplerWrapper(
            model=inr,
            iters=0,
            device=device,
            sample_rate=cfg.sampling.rate,
            save_samples_path=save_path,
            image_width = image_width
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
        save_samples_path=save_path,
        n_clusters_2d_start=cfg.sampling.n_clusters_2d_start,
        n_clusters_2d_end=cfg.sampling.n_clusters_2d_end,
        epochs = cfg.optim.epochs,
        image_width = image_width
    )
