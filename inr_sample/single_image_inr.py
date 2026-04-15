import os
import sys
from pathlib import Path
# sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import hydra
import numpy as np
import torch
import torch.nn as nn
from util.logger import log
from itertools import islice
from time import time


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["WANDB_DIR"] = str(REPO_ROOT / "coral" / "wandb")
os.environ["RESULTS_DIR"] = ''

import wandb
from omegaconf import DictConfig, OmegaConf
from utils.data.unstructure_dataset import (
    GraphNavierStokes,
    collate_graph_inr,
    create_piecewise_dataset,
    get_graph_t_idx,
    create_ns_dataset,
    create_burgers2d_dataset,
    create_poolboiling2d_dataset,
)
from train_utility_sampling.train_utility import (
    train_step, 
    validation_step,
    save_sampling_result,
    train_step_single_image,
    validation_step_single_image,
)
from train_utility_sampling.SamplerWrapper import (
    create_inr_sampler,
)
from utils.data.load_data import set_seed
from utils.load_inr import create_inr_instance, load_inr_model
from datetime import datetime
from time import time
from torch_geometric.data import Data
import pandas as pd 
# import seaborn as sns
import matplotlib.pyplot as plt

def initialize_wandb(cfg):
    if cfg.wandb.use_wandb:
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        dataset_name = cfg.data.dataset_name
        run_name = cfg.wandb.name
        run_dir = None

        print("run dir given", run_dir)
        run = wandb.init(
            #entity=entity,
            project=project,
            name=run_name,
            id=run_id,
            dir=run_dir,
            # resume='allow',
        )
        # if run_dir is not None:
        #     os.symlink(run.dir.split("/files")[0], run_dir)

        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        print("id", run.id)
        print("dir", run.dir)
        return run
    
# def create_inr_sampler(cfg, inr, graph, current_date_str, run_name, device='cuda'):
#     """
#     Build and return an INRSingle2dSamplerWrapper or EVOSSampler based on cfg.sampling 
#     settings, or None if no sampling type is specified.
#     """
#     sampling_type = cfg.sampling.type
#     image_width = graph.cor.max().item() + 1  # Set image width from space_emb shape
    
#     if sampling_type is None:
#         return None

#     # Precompute SLIC clusters once for the slic sampler.
#     if sampling_type == "2d_cluster_slic":
#         _start = cfg.sampling.n_clusters_2d_start
#         graph_2d_cluster_single_image(graph, _start, 0.01, "slic")
#         sampler_name = sampling_type
#     elif sampling_type == "2d_grid_adaptive":
#         save_path = f'./sampled_frames/{current_date_str + run_name}'
#         return INRSingle2dAdaptiveSamplerWrapper(
#             model=inr,
#             iters=0,
#             device=device,
#             sample_rate=cfg.sampling.rate,
#             mode=cfg.sampling.get("adaptive_mode", "loss"),
#             grid_update_interval=cfg.sampling.get("adaptive_grid_update_interval", 100),
#             save_samples_path=save_path,
#             image_width=image_width,
#         )
#     elif sampling_type == "EVOS":
#         H = int(np.sqrt(len(graph.feat)))
#         img = graph.feat.reshape(H, H)
#         img = img.unsqueeze(0)
#         # print("===img for evos===\n" + str(img))
#         # print("===shape for evos===\n" + str(img.shape))
#         return EVOSSampler(cfg, img, graph)

#     else:
#         sampler_name = sampling_type

#     save_path = f'./sampled_frames/{current_date_str + run_name}'
#     return INRSingle2dSamplerWrapper(
#         model=inr,
#         iters=0,
#         device=device,
#         sample_rate=cfg.sampling.rate,
#         sample_type=sampler_name,
#         use_weight_function=cfg.sampling.get("use_weight_function", True),
#         save_samples_path=save_path,
#         n_clusters_2d_start=cfg.sampling.n_clusters_2d_start,
#         n_clusters_2d_end=cfg.sampling.n_clusters_2d_end,
#         epochs = cfg.optim.epochs,
#         image_width = image_width
#     )


def _build_trainset(cfg, latent_dim, data_path, dataset_name, data_type, seed, space_factor):
    """Create the dataset and return (trainset, input_dim, output_dim, transforms)."""
    if dataset_name == "NS":
        input_dim = 2
        output_dim = 1
        trainset = create_ns_dataset(
            datapath=data_path,
            data_type=data_type,
            seed=seed,
            single_image=True,
        )
        feat_transform, feat_inv_transform = None, None
    elif dataset_name == "Burgers2D":
        input_dim = 2
        output_dim = 1
        trainset = create_burgers2d_dataset(
            data_dir=data_path,
            nu=cfg.data.get("burgers2d_nu", 0.01),
            latent_dim=latent_dim,
            space_factor=space_factor,
            seed=seed,
            single_image=True,
            sample_idx=cfg.data.get("burgers2d_sample_idx", 0),
        )
        feat_transform, feat_inv_transform = None, None
    elif dataset_name == "Piecewise2D":
        input_dim = 2
        output_dim = 1
        trainset = create_piecewise_dataset(
            resolution=cfg.data.piecewise_resolution,
            alpha=cfg.data.piecewise_alpha,
            beta=cfg.data.piecewise_beta,
            eps=cfg.data.piecewise_eps if cfg.data.piecewise_eps else None,
        )
        feat_transform, feat_inv_transform = None, None
    elif dataset_name == "PoolBoiling2D":
        input_dim = 2
        output_dim = 1
        trainset = create_poolboiling2d_dataset(
            data_dir=data_path,
            latent_dim=latent_dim,
            space_factor=space_factor,
            seed=seed,
            single_image=True,
            sample_idx=cfg.data.get("poolboiling_sample_idx", 0),
            field_key=cfg.data.get("poolboiling_key", "temperature"),
            condition=cfg.data.get("poolboiling_condition", 100),
            file_name=cfg.data.get("poolboiling_file", None),
        )
        feat_transform, feat_inv_transform = None, None
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    return trainset, input_dim, output_dim, feat_transform, feat_inv_transform


def _select_single_graph(trainset, cfg, dataset_name):
    """Pick one graph from trainset (by configurable index) and one time frame."""
    graph_sample_idx = int(cfg.data.get("graph_sample_idx", 0))
    if graph_sample_idx < 0 or graph_sample_idx >= len(trainset):
        raise IndexError(
            f"data.graph_sample_idx={graph_sample_idx} is out of range for trainset size {len(trainset)}"
        )

    graph = trainset[graph_sample_idx]

    t = cfg.data.single_time_frame
    if dataset_name == "Piecewise2D":
        t = 0

    indices_t = get_graph_t_idx(graph, t)
    if len(indices_t) == 0:
        t_min = int(graph.time.min().item()) if graph.time.numel() > 0 else -1
        t_max = int(graph.time.max().item()) if graph.time.numel() > 0 else -1
        raise ValueError(
            f"No nodes found for data.single_time_frame={t}. Available time range is [{t_min}, {t_max}]"
        )

    return Data(
        cor=graph.cor[indices_t],
        feat=graph.feat[indices_t],
        time=torch.zeros(len(indices_t)),
        space_emb=graph.space_emb[indices_t],
        T=torch.tensor(1),
    )

@hydra.main(config_path="../config/", config_name="inr_sample.yaml")
def main(cfg: DictConfig) -> None:
    #Initialization
    trl_vals_arr = []
    time_vals_arr = []
    step_vals_arr = []

    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)
    current_date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # data
    saved_checkpoint = cfg.saved_checkpoint
    if saved_checkpoint:
        checkpoint = torch.load(cfg.checkpoint_path)
        cfg = checkpoint['cfg']
        print('---------Load Checkpoint------------------')
    
    print(OmegaConf.to_yaml(cfg))
    
    # Gradient norms
    grad_norms = []

    #wandb
    run_name = cfg.wandb.name
    run_name_str = run_name if run_name else "unnamed"

    #data
    data_path = cfg.data.data_path
    dataset_name = cfg.data.dataset_name
    split_ratios = tuple(cfg.data.split_ratios)
    space_factor = cfg.data.space_factor
    time_factor = cfg.data.time_factor
    seed = cfg.data.seed
    
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_type = cfg.data.data_type
    mmap_dir = cfg.data.mmap_dir
    data_to_encode = cfg.data.data_to_encode
    val_missing_rate = cfg.data.val_missing_rate
    sub_array_num = cfg.data.sub_array_num

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr  # lr_inr is learning parameter initial value, also the meta learning rate
    lr_code = cfg.optim.lr_code 
    meta_lr_code = cfg.optim.meta_lr_code # meta learning rate to learn
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    epochs = cfg.optim.epochs
    evo_every_epochs = cfg.optim.evo_every_epochs

    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim

    # wandb
    run = initialize_wandb(cfg)

    device = torch.device("cuda")
    set_seed(seed)
    
   
   
    """ define dataset """
    trainset, input_dim, output_dim, feat_transform, feat_inv_transform = _build_trainset(
        cfg=cfg,
        latent_dim=latent_dim,
        data_path=data_path,
        dataset_name=dataset_name,
        data_type=data_type,
        seed=seed,
        space_factor=space_factor,
    )

    ntrain = len(trainset)

    graph = _select_single_graph(trainset, cfg, dataset_name)
    graph = graph.to(device)
    graph_ori = graph.clone()
    # print("indices ->\n" + str(len(indices_t)))

    # print("---time shape---\n", str(graph.time.shape))
    # print("---T---\n", str(graph.T))
    # print("---initial graph---\n" + str(graph) + "\n---coordinates---\n" + str(graph.cor))
    # print("---features---\n" + str(graph.feat) + "\n---time---\n" + str(graph.time))
    # print("---space_emb---\n" + str(graph.space_emb) + "\n---T---\n" + str(graph.T) + "\n---end---")

    log.start_timer("final")
    total_train_time = 0
    t0 = time()
    
    """ Initialize model, optimizer """
    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device=device
    )  # add sampler to model
    trainable_params = sum(p.numel() for p in inr.parameters() if p.requires_grad)
    print("inr model parameters:", {trainable_params})
    
    alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))
    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code

    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": lr_inr},
            {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    if cfg.sampling.type == "EVOS":
        epoch_start = 1
    else:   # EVOS uses 1 based indexing for epochs
        epoch_start = 0
    best_loss = np.inf
    

    
    """ Update model and optimizer with checkpoint"""
    if saved_checkpoint:
        inr.load_state_dict(checkpoint['inr'])
        optimizer.load_state_dict(checkpoint['optimizer_inr']) 
        epoch_start = checkpoint['epoch']
        alpha = checkpoint['alpha']
        best_loss = checkpoint['loss']
        cfg = checkpoint['cfg']
        print("epoch_start, alpha, best_loss", epoch_start, alpha.item(), best_loss)

    # if cfg.wandb.use_wandb:
    #     wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)
    
    # if cfg.sampling.type in ['3d_cluster']:
    #     cluster_dim = '2d'
    #     add_cluster_label(train_loader, 1000, 0.01, cluster_dim=cluster_dim)
    #     add_cluster_label(val_loader, 1000, 0.01, cluster_dim=cluster_dim)
    
    # graph = Data(
    #     cor=graph.cor[indices_t],
    #     feat=graph.feat[indices_t],
    #     time=torch.full((len(indices_t),), 995, dtype=torch.long),  # actual time
    #     space_emb=graph.space_emb[indices_t],
    #     T=torch.tensor(1000),
    # )


    ''' Begin the sampling setting '''
    print("before:", cfg.sampling.adaptive_weight_mode)
    inr_sampler = create_inr_sampler(cfg, inr, graph, current_date_str, run_name)
    if cfg.sampling.type == "EVOS":
        inr_sampler._evos_init()
        
    # for _ in range(3):  # a few steps to warm up
    #     train_loss, rel_train_loss, grad_norm = train_step_single_image(
    #         -1, graph, inr, 
    #         device=device,
    #         use_rel_loss=False,
    #         optimizer=optimizer,
    #         sampler=inr_sampler,
    #         cfg = cfg
    #         )

    # torch.cuda.synchronize()  # make sure GPU ops are done
    
    # Main Training Loop
    ''' Begin the training process '''
    # log.start_timer("step")
    for step in range(epoch_start, epochs):
        
        use_rel_loss = True
        step_show = step % evo_every_epochs == 0
        step_show_last = step == epochs - 1

        # Start Timer
        t1 = time()
        train_loss, rel_train_loss, grad_norm = train_step_single_image(
            step, graph, inr, 
            device=device,
            use_rel_loss=use_rel_loss,
            optimizer=optimizer,
            sampler=inr_sampler,
            cfg = cfg
            )
        t_step = time() - t1
        # print("total train time for step", step, ":", t_step)
        total_train_time += t_step

        torch.cuda.synchronize()
        if True in (step_show, step_show_last):
            if cfg.sampling.type != None:
                if cfg.sampling.type != "EVOS" and step % 100 == 0:
                    # if cfg.sampling.type == "2d_grid_linear":  # Comment this block out to remove scheduler
                    #     print("New cluster graph made")
                    #     _start = cfg.sampling.n_clusters_2d_start
                    #     _end = cfg.sampling.n_clusters_2d_end
                    #     n_clusters = _start + ((_end - _start) / epochs) * step
                    #     graph_2d_cluster_single_image(graph_ori, n_clusters, 0.01, 'grid')
                    inr_sampler.sample(
                        inner_step=step, 
                        graph=graph, 
                        save_image=True,
                    )
                elif cfg.sampling.type == "EVOS" and step % 100 == 0:
                    inr_sampler.sample(
                        inner_step=step, 
                        graph=graph, 
                        save_image=True,
                        epoch=step
                    )
            test_loss, rel_test_loss, psnr, ssim = validation_step_single_image(
                step, graph, inr, 
                device=device, 
                use_rel_loss=use_rel_loss,
                optimizer=optimizer,
                sampler=inr_sampler,
                cfg = cfg
                )

            if cfg.wandb.use_wandb:
                wandb.log(
                    {
                        "test_rel_loss": rel_test_loss,
                        "train_rel_loss": rel_train_loss,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                        "Time": total_train_time, 
                        "psnr": psnr,
                        "ssim": ssim,
                        "max_gpu_memory_GB": torch.cuda.max_memory_allocated() / 1e9,
                    },
                    step=step
                )
                trl_vals_arr.append(rel_test_loss)
                time_vals_arr.append(total_train_time)
                step_vals_arr.append(step)
            else:
                print(
                    f"Step {step}, Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Train Rel Loss: {rel_train_loss:.4f}, "
                    f"Test Rel Loss: {rel_test_loss:.4f}"
                )
            loss_to_check = rel_test_loss if use_rel_loss else test_loss
            if loss_to_check < best_loss:
                best_loss = loss_to_check

                dir_path = REPO_ROOT / "Results" / "checkpoints" / f"{current_date_str + run_name_str}"
                dir_path.mkdir(parents=True, exist_ok=True)
                savepath = dir_path / f"{step}.pt"

                # Build graph snapshot (CPU) for visualization
                _graph_data = {
                    "cor": graph_ori.cor.detach().cpu(),
                    "space_emb": graph_ori.space_emb.detach().cpu(),
                    "feat": graph_ori.feat.detach().cpu(),
                    "time": graph_ori.time.detach().cpu(),
                }

                _ckpt = {
                    "cfg": cfg,
                    "epoch": step,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "loss": best_loss,
                    "alpha": alpha,
                    "feat_transform": feat_transform,
                    "feat_inv_transform": feat_inv_transform,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "graph_data": _graph_data,
                }
                torch.save(_ckpt, str(savepath))

                # Also save best checkpoint to a local, predictable path
                _local_ckpt_dir = './checkpoints'
                os.makedirs(_local_ckpt_dir, exist_ok=True)
                _local_ckpt_path = os.path.join(
                    _local_ckpt_dir,
                    f'{current_date_str}_{run_name_str}_best.pt'
                )
                torch.save(_ckpt, _local_ckpt_path)
                print(f'[Checkpoint] Best model (step {step}, loss {best_loss:.6f}) saved to: {_local_ckpt_path}')
    
    log.end_timer("final")
    print("TIME:", total_train_time)

    # Save final checkpoint after training completes
    _graph_data_final = {
        "cor": graph_ori.cor.detach().cpu(),
        "space_emb": graph_ori.space_emb.detach().cpu(),
        "feat": graph_ori.feat.detach().cpu(),
        "time": graph_ori.time.detach().cpu(),
    }
    _final_ckpt_dir = './checkpoints'
    os.makedirs(_final_ckpt_dir, exist_ok=True)
    _final_ckpt_path = os.path.join(
        _final_ckpt_dir,
        f'{current_date_str}_{run_name_str}_final.pt'
    )
    torch.save(
        {
            "cfg": cfg,
            "epoch": epochs - 1,
            "inr": inr.state_dict(),
            "optimizer_inr": optimizer.state_dict(),
            "loss": best_loss,
            "alpha": alpha,
            "feat_transform": feat_transform,
            "feat_inv_transform": feat_inv_transform,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "graph_data": _graph_data_final,
        },
        _final_ckpt_path,
    )
    print(f'[Checkpoint] Final model saved to: {_final_ckpt_path}')

    # Output stats to file
    # with open('/sdcc/u/smccue/projects/inr_sampling/visuals/out.txt', 'a') as f:
    #     trl_vals = (','.join(str(v) for v in trl_vals_arr))
    #     time_vals = (','.join(str(v) for v in time_vals_arr))
    #     step_vals = (','.join(str(v) for v in step_vals_arr))
    #     print(str(cfg.sampling.type) + 
    #             "\ntrl\n" + str(trl_vals) +
    #             "\ntime\n" + str(time_vals) +
    #             "\nstep\n" + str(step_vals),
    #             file=f)  # Python 3.x

    return rel_test_loss

if __name__ == "__main__":
    main()
    time = log.end_all_timer()
    print('finish')