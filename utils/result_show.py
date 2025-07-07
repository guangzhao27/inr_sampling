import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import torch
import sys
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append('/pscratch/sd/g/gzhao27/INR/SOMA')
from coral.utils.models.load_inr import create_inr_instance, load_inr_model
from train_utility_sampling.train_utility import (
    validation_step,
    split_graph_by_time,
)
from train_utility_sampling.metalearning_sampling import graph_outer_step as outer_step
from typing import List


# from train_utility.train_utility import split_graph_by_time, outer_step

# from your_model_factory import create_inr_instance  # wherever you defined it

def display_videos_side_by_side(videos, titles=None, interval=100, max_frames=None):
    """
    Display multiple videos side-by-side as an animation in a Jupyter notebook.
    
    Parameters:
        videos (list of arrays): List of 3D or 4D numpy arrays (T, H, W) or (T, H, W, C)
        titles (list of str): Optional list of titles for each subplot.
        interval (int): Delay between frames in milliseconds.
        max_frames (int): Optional limit on the number of frames to display.
    
    Returns:
        HTML object containing the animation.
    """
    num_videos = len(videos)
    titles = titles if titles is not None else [f'Video {i}' for i in range(num_videos)]
    
    np_list = [np.array(v) for v in videos]
    
    # Optionally truncate the number of frames
    if max_frames is not None:
        np_list = [v[:max_frames] for v in np_list]
    
    fig, axes = plt.subplots(1, num_videos, figsize=(5 * num_videos, 5))
    if num_videos == 1:
        axes = [axes]
    
    images = []
    for ax, video, title in zip(axes, np_list, titles):
        im = ax.imshow(video[0])
        ax.set_title(title)
        ax.axis('off')
        images.append(im)
    
    def update(frame):
        for im, video in zip(images, np_list):
            im.set_array(video[frame])
        return images
    
    ani = FuncAnimation(fig, update, frames=range(np_list[0].shape[0]), blit=True, interval=interval)
    plt.close(fig)  # Prevent duplicate display in notebook
    ani.save('animation.gif', writer='pillow')  # Save the animation as a GIF if needed
    return HTML(ani.to_jshtml())



def graph_to_videos(graph, image_pred=None)-> np.ndarray:
    """
    Convert a graph with spatial-temporal data into ground truth and predicted video arrays.
    
    Parameters:
        graph: A graph object with the following attributes:
            - graph.T: a scalar tensor indicating the number of time steps.
            - graph.cor: a tensor of coordinates (N, 2).
            - graph.time: a tensor of time indices (N,).
            - graph.feat: a tensor of feature values (N,).
        image_pred (optional): A tensor of predicted values aligned with graph.feat.
    
    Returns:
        video_gt: A (T, X, Y) NumPy array of ground truth values.
        video_pred: A (T, X, Y) NumPy array of predicted values.
    """
    # Determine spatial dimensions from max coordinates
    X, Y = graph.cor.max(dim=0).values.int().tolist()
    X += 1
    Y += 1

    T = int(graph.T.sum())

    # Initialize output videos with NaNs
    video_np = np.full((T, X, Y), np.nan, dtype=np.float32)
    
    # values_np = graph.feat.cpu().numpy()
    if image_pred is not None:
        values_np = image_pred
    else:
        values_np = graph.feat
    # video_pred = np.full((T, X, Y), np.nan, dtype=np.float32)

    for t in range(T):
        tidx = (graph.time % T == t)
        coordinates = graph.cor[tidx]
        values_gt = values_np[tidx]

        coordinates_np = coordinates.cpu().numpy()
        values_gt_np = values_gt.cpu().numpy().reshape(-1)

        for i, (x, y) in enumerate(coordinates_np):
            video_np[t, x, y] = values_gt_np[i]

    return video_np



def show_inr_results_from_save_path(
    input_dim: int,
    output_dim: int,
    inr_save_path_list: List[str],
    sampled_graph,
    titles: List[str] = None,
    device: torch.device | str = None,
    interval: int = 100,
    sub_array_num: int = 1,
):
    """
    Instantiate an INR from cfg, load its state_dict from inr_results['inr'],
    and display GT vs prediction side-by-side for one sample.

    Args:
        cfg:           configuration object for create_inr_instance
        inr_results:   dict containing at least the key 'inr' → state_dict
        input_dim:     input_dim passed to create_inr_instance
        output_dim:    output_dim passed to create_inr_instance
        test_loader:   DataLoader yielding graph objects
        alpha:         latent-scale tensor
        inner_steps:   number of inner‐loop steps for outer_step
        sub_array_num: how many temporal splits to make on the graph
        device:        torch device (str or torch.device); if None, picks cuda if available
        interval:      ms between animation frames

    Returns:
        HTML animation object for display in Jupyter
    """
    
    video_gt = graph_to_videos(sampled_graph)
    video_list = [video_gt]
    for inr_save_path in inr_save_path_list:
        # ——— 1. Device & INR instantiation —————————————————————————————
        inr_results = torch.load(inr_save_path, weights_only=False)
        cfg = inr_results['cfg']
        alpha = inr_results['alpha']
        inner_steps = cfg.optim.inner_steps
        device = (
            torch.device(device)
            if isinstance(device, (str, torch.device))
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        # create & load weights
        torch.set_default_dtype(torch.float32)
        inr = create_inr_instance(cfg, input_dim=input_dim, output_dim=output_dim, device=device)
        inr.load_state_dict(inr_results["inr"])
        inr.to(device).eval()
        alpha = alpha.to(device)


        # ——— 3. Split graph in time chunks ————————————————————————————
        sub_graph_list = split_graph_by_time(sampled_graph, sub_array_num)

        # ——— 4. Run inference on each sub-graph ————————————————————————
        preds = []
        for g in sub_graph_list:
            graph_time = g.T.sum()
            g.to(device)
            outputs = outer_step(
                inr,
                g,
                inner_steps,
                alpha,
                is_train=False,
                return_reconstructions=True,
            )
            preds.append(outputs["reconstructions"])

        pred_feats = torch.cat(preds, dim=0)

        # ——— 5. Convert to (T, X, Y) videos —————————————————————————————
        
        video_pred = graph_to_videos(sampled_graph, pred_feats)
        video_list.append(video_pred)
    
    if titles is None:
        titles = [f"INR {i+1}" for i in range(len(video_list))]

    # ——— 6. Display side-by-side animation ————————————————————————
    return display_videos_side_by_side(
        video_list,
        titles=titles,
        interval=interval,
    )


# from IPython.display import display

# html_anim = display_videos_side_by_side([video_gt, video_inr], titles=['Ground Truth', 'INR Output'])
# display(html_anim)