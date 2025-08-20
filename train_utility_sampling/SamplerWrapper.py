import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple
from pathlib import Path
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from collections import defaultdict
import random
from utils.data.unstructure_dataset import get_graph_t_idx
from skimage.segmentation import slic
from .sampler import mt_sampler, save_samples, save_losses
import math
# from .strategy import strategy_factory
# https://github.com/chen2hang/INT_NonparametricTeaching/blob/bfc995d43c81584e9d2d4c8aa93571c662c129e0/src/nmt.py

# From EVOS
from util.misc import fix_seed
from components.laplacian import compute_laplacian_loss as compute_laplacian
from components.nmt import mt_scheduler_factory
from components.transform import Transform
import torch.nn.functional as F
from tqdm import trange

class InrSamplerWrapper:
    """
    Wrapper class for the coordinate sampling algorithms.

    Args:
        model (torch.nn.Module): The model to be trained.
        iters (int): The number of iterations to train the model.
        scheduler (str, optional): The type of scheduler to use. Defaults to "step".
        strategy (str, optional): The type of strategy to use. Defaults to "incremental".
        starting_ratio (float, optional): The starting ratio for the NMT algorithm. Defaults to 0.2.
        top_k (bool, optional): Whether to use top-k sampling. Defaults to True.
        save_samples_path (Path, optional): The path to save the samples. Defaults to Path("logs/sampling").
        save_losses_path (Path, optional): The path to save the losses. Defaults to Path("logs/losses").
        save_name (str, optional): The name to save the samples and losses. Defaults to None.
        save_interval (int, optional): The interval to save the samples and losses. Defaults to 1000.

    Elaborations on some important Args:

        <schedulers> determine the ratio of samples to be taken at each iteration.
        Types of schedulers available (defined in scheduler.py):
            - "step": mt_step
            - "linear": mt_linear
            - "cosine": mt_cosineAnnealing
            - "reverse-cosine": mt_revCosineAnnealing
            - "constant": mt_constant
        
        <strategies> determine the intervals at which samples are taken.
        Types of strategies available (defined in strategy.py):
            - "incremental": incremental
            - "reverse-incremental": revIncremental
            - "expoenential": exponential
            - "dense": dense
            - "void": void

        <top_k> determines whether to we select the samples based on the highest loss values.
        If otherwise, we will select samples randomly.
    """
    def __init__(
                self,
                n_clusters_2d_start: 100,
                n_clusters_2d_end: 100,
                epochs: 5000,
                model: torch.nn.Module,
                iters: int,
                device: str="cuda:0",
                sample_type: str="random",
                sample_rate: float=0.5,
                save_samples_path: Path=Path("logs/sampling"),
                # scheduler: str="step",
                #  strategy: str="incremental",
                #  starting_ratio: float=0.2,
                #  top_k: bool=True,
                #  save_samples_path: Path=Path("logs/sampling"),
                #  save_losses_path: Path=Path("logs/losses"),
                #  save_name: str=None,
                save_interval: int=100,
                ):
        self.n_clusters_2d_start = n_clusters_2d_start
        self.n_clusters_2d_end = n_clusters_2d_end
        self.epochs = epochs
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        # self.scheduler = mt_scheduler_factory(scheduler)
        # self.strategy = strategy_factory(strategy)
        self.sample_type = sample_type
        self.sample_rate = sample_rate
        self.iters = iters
        self.save_interval = save_interval
        self.save_samples_path = save_samples_path

    def sample(self, outer_step: int, inner_step: int, graph: Data, modulations: torch.Tensor=None, save_image=False, ) -> Data:
        """
        Perform the NMT sampling.

        Args:
            outer_step (int): The current meta outer step, control if or not to sample.
            inner_step (int): indicate the sampling iteration, for save image only.
            
            modulation is the meta learning trained modulation
        """
        if hasattr(graph.T, "sum"):
            # covers both np.ndarray and np.int64
            T = graph.T.sum()
        else:
            # integer fallback
            T = graph.T
        if self.sample_type == "random":            
            sampled_indices = []
            # For each time frame, randomly sample a subset of nodes.
            for t in range(T):
                # Get indices for nodes belonging to time frame t.
                indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
                n_t = indices_t.numel()
                # Ensure at least one node is sampled per time frame.
                n_samples = max(int(n_t * self.sample_rate), 1)
                # Randomly permute indices and select n_samples of them.
                perm = torch.randperm(n_t, device=self.device)[:n_samples]
                sampled_indices.append(indices_t[perm])
            
            # Concatenate indices from all time frames.
            # sampled_indices = torch.cat(sampled_indices, dim=0)
            # Sort the combined indices by time to preserve temporal order.
            sampled_idx = torch.cat(sampled_indices, dim=0)
            
        elif self.sample_type == "NMT":
            assert modulations is not None, "Modulations must be provided for NMT sampling."
                
            with torch.no_grad():
                # forward pass
                graph = graph.to(self.device)
                modulations = modulations.to(self.device)
                preds = self.model.modulated_forward(graph.space_emb, modulations[graph.time.cpu()])
                
                features = graph.feat
                dif = torch.sum(torch.abs(features - preds), 1)
                sampled_indices = []
                for t in range(T):
                    indices_t = (graph.time == t).nonzero(as_tuple=True)[0]
                    n_t = indices_t.numel()
                    _, top_idx = torch.topk(dif[indices_t], int(self.sample_rate * n_t))
                    sampled_indices.append(indices_t[top_idx])
                
                sampled_idx = torch.cat(sampled_indices, dim=0)
        elif self.sample_type == "3d_cluster":
            assert modulations is not None, "Modulations must be provided for clustering sampling."
            W, H = graph.cor.max(axis=0)[0] +1
            n_samples = max(1, int(W * H * T * self.sample_rate))
            
            # this is just an estimate of the num_per_cluster, can not be accurate
            # because the number of nodes in each cluster can vary.
            num_per_cluster = max(1, math.ceil(n_samples / len(graph.cluster_set[0])))
            
            rough_idx = sample_random_node_indices_per_cluster(
                graph, cluster_dim='3d', num_per_cluster=num_per_cluster
                )
            times = graph.time[rough_idx]                      # shape = [N_rough]
            space_emb = graph.space_emb[rough_idx].to(self.device)  # [N_rough, D]
            feats     = graph.feat[rough_idx].to(self.device)       # [N_rough, F]
            mod       = modulations.to(self.device)
            with torch.no_grad():
                # forward pass
                # graph = graph.to(self.device)
                # modulations = modulations.to(self.device)
                preds = self.model.modulated_forward(space_emb, mod[times.cpu()])  # [N_rough, F]
                diffs = torch.sum((feats - preds).abs(), dim=1)      
            sampled_per_t = []
            
            # TODO: make it sample over whole graph
            for t in range(T):
                # local positions in the rough_idx array
                local_mask = (times == t).nonzero(as_tuple=True)[0]
                if local_mask.numel() == 0:
                    continue

                count = min(int(W*H*self.sample_rate), local_mask.numel())
                _, topk_local = torch.topk(diffs[local_mask], count)

                # map those back to the original-graph indices
                selected_global = rough_idx[local_mask[topk_local.cpu()]]
                sampled_per_t.append(selected_global)

            # 4) Concatenate all selected indices
            sampled_idx = torch.cat(sampled_per_t, dim=0)                
            
        else:
            raise NotImplementedError(f"Sampling type {self.sample_type} is not implemented.")

        sampled_graph = Data(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
            T=graph.T,  # global property (total time frames) remains unchanged
            latent_vector=graph.latent_vector  # global latent vector remains unchanged
        )
        # sampled_graph = sampled_graph.to(device)
        
        if save_image:
            self.save_image_path = os.path.join(self.save_samples_path,  f"{self.sample_type}_o{outer_step}_i{inner_step}")
            if 'dif' not in locals():
                dif = None
            self._save_sample_images(graph, sampled_graph, dif=dif)
        return sampled_graph


    def get_ratio(self):
        return self.ratio
    
    def get_interval(self):
        return self.mt_intervals

    def get_saved_samples_path(self):
        return self.save_sample_path

    def get_saved_losses_path(self):
        return self.save_loss_path

    def get_saved_tint_path(self):
        return self.save_tint_path

    def _tint_data_with_samples(self, data, sample_idx, tint_color: List[float]=[0.5, 0.0, 0.0]):
        """Relabel the data with given vis_label at the sample_idx indices."""
        if sample_idx is None: 
            return None
        
        new_data = data.detach().clone()
        vis_label = torch.tensor(tint_color).to(data.device)
        if data.shape[-1] == 1:
            vis_label = vis_label[0]

        new_data[sample_idx] = torch.clamp(new_data[sample_idx] + vis_label, max=1.0)

        return new_data
    
    def _preprocess_img(self, image, h, w, c):
        """Preprocess the image for saving."""
        if torch.min(image) < 0:
            image = image.clamp(-1, 1).view(h, w, c)       # clip to [-1, 1]
            image = (image + 1) / 2                        # [-1, 1] -> [0, 1]
        else:
            image = image.clamp(0, 1).view(h, w, c)       # clip to [0, 1]

        image = image.cpu().detach().numpy()

        return image
    
    def _save_image(self, img, path, h, w, color_mode="RGB"):
        """Save the image to the given path."""
        img = Image.fromarray((img *255).astype(np.uint8), mode=color_mode)
        if img.size[0] > 512:
            img = img.resize((512, int(512*h/w)), Image.LANCZOS)
        img.save(path)
        print(f"Image saved to {path}")
        
    
    def _save_sample_images(self, graph, sampled_graph, dif=None):
        """
        Save an image for each time frame with red dots indicating sampled positions.
        
        Parameters:
        - graph: torch_geometric Data object with attributes cor (coords), time, feat (field values), T (total frames)
        - sorted_idx: 1D tensor of indices of sampled nodes
        """
        # Create directory if it doesn't exist
        os.makedirs(self.save_image_path, exist_ok=True)
        
        if isinstance(graph.T, torch.Tensor) and graph.T.dim() >=1:
            T_show = graph.T[0]
        else:
            T_show = graph.T
        W = graph.cor[:, 0].max()+1
        H = graph.cor[:, 1].max()+1
        
        # Loop over each time frame
        for t in range(T_show):
            # Mask for all nodes at time t
            frame_mask = (graph.time == t)
            coords = graph.cor[frame_mask].cpu().numpy()
            values = graph.feat[frame_mask].cpu().numpy()
            
            # Mask for sampled indices at time t
            sampled_frame_mask = (sampled_graph.time == t)
            sampled_coords = sampled_graph.cor[sampled_frame_mask].cpu().numpy()
            
            # Plot the full field
            plt.figure()
            field = values.reshape(H, W)  # Reshape for color mapping
            plt.imshow(field, cmap='viridis', origin='lower')
            plt.axis('off')            
            # Overlay sampled points
            plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0],
                        c='red', s=0.015625,
            )
            plt.title(f'Time Frame {t}')
            
            
            
            # Save figure
            filename = Path(self.save_image_path) / f'frame_{t:03d}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            if dif is not None:
                # Overlay differences if provided
                plt.figure()
                dif_frame = dif[frame_mask].cpu().numpy().reshape(H, W)
                plt.imshow(dif_frame, cmap='hot', alpha=0.5, origin='lower')
                plt.axis('off')            
                plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0],
                            c='red', s=10,
                )
                plt.title(f'Time Frame {t}')
                filename = Path(self.save_image_path) / f'frame_{t:03d}_dif.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()


def sample_random_node_indices_per_cluster(
    graph,
    cluster_dim = '2d',
    num_per_cluster: int = 1,
    ) -> torch.Tensor:
    """
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
        elif self.sample_type == "2d_cluster":
            # For 2D graphs, we can still use the 3D cluster sampling function
            # but it will sample from the 2D clusters.
            # This is a workaround to use the same sampling function.
            # In practice, you might want to implement a separate 2D sampling function.
            # Here we assume the graph has been clustered already. 

            use_2d_cluster_grid_scheduling = False
            if inner_step % 100 == 0 and use_2d_cluster_grid_scheduling:
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

        sampled_graph = Data(
            cor=graph.cor[sampled_idx],
            time=graph.time[sampled_idx],
            feat=graph.feat[sampled_idx],
            space_emb=graph.space_emb[sampled_idx],
        )
        
        if save_image:
            self.save_image_path = os.path.join(self.save_samples_path, f"2d_i{inner_step}")
            if self.sample_type != "NMT":
                dif = None
            self._save_sample_images(graph, sampled_graph, dif=dif)
        return sampled_graph

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

    def _sampler_get_coords_gt(self, epoch):
        coords, gt = self.full_coords, self.full_gt
        
        self.cur_use_ratio = self._get_cur_use_ratio(epoch)

        self._reset_rng()
 
        if self._evos_is_fitness_eval_iter(epoch):
            return coords, gt, None
        
        selection_mask = self._evos_get_selection_mask(epoch)
        _coords = self.full_coords[selection_mask]
        _gt = self.full_gt[selection_mask]
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
        return F.mse_loss(pred, gt)

    def reconstruct_img(self, data) -> torch.tensor:    # From EVOS img_trainer.py
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # c,h,w
        img = self._decode_img(img)
        return img

    def _decode_img(self, data):    # From EVOS img_trainer.py
        data = self.transform.inverse(data)
        return data

    def _parse_input_data(self):
        img = self.input_img.permute(2, 0, 1)  # c,h,w
        self.input_img = img
        self.gt = img
        self.C, self.H, self.W = img.shape

    def _encode_img(self, img):
        img = self.transform.tranform(img)
        return img

    def _get_data(self):
        img = self.input_img
        img = self._encode_img(img)
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
        inner_step=1, 
        modulations: torch.Tensor=None, 
        save_image=False):

        coords, gt, sel_mask = self._sampler_get_coords_gt(inner_step)

        if sel_mask != None:
            graph = Data(
                cor = coords,
                feat = gt,
                time = graph.time[sel_mask],
                space_emb = graph.space_emb[sel_mask],
                T=graph.T,  # global property (total time frames) remains unchanged
            )

        else:
            graph = Data(
                cor = graph.cor,
                feat = gt,
                time = graph.time,
                space_emb = graph.space_emb,
                T=graph.T,  # global property (total time frames) remains unchanged
            )

        return graph