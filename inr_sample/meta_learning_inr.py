import os
import sys
from pathlib import Path
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf
from utils.data.unstructure_dataset import (
    GraphNavierStokes, 
    collate_graph_inr, 
    create_burgers2d_dataset, 
    create_ns_dataset,
    )
from train_utility_sampling.train_utility import (
    train_step, 
    validation_step,
    save_sampling_result,
)
from train_utility_sampling.SamplerWrapper import InrSamplerWrapper, add_cluster_label
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.plot import show
from datetime import datetime
from time import time
from torch_geometric.data import Data



os.environ["WANDB_DIR"] = '/pscratch/sd/g/gzhao27/INR/coral/wandb'
os.environ["RESULTS_DIR"] = ''

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

        # if data_to_encode is not None:
        #     RESULTS_DIR = (
        #         Path(os.getenv("WANDB_DIR")) / dataset_name / data_to_encode / "inr"
        #     )
        # else:
        #     RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"

        # os.makedirs(str(RESULTS_DIR), exist_ok=True)

@hydra.main(config_path="../config/", config_name="inr_sample.yaml")
def main(cfg: DictConfig) -> None:

    
    #Initialization

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
    
    
    #wandb
    run_name = cfg.wandb.name

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

    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim

    # wandb
    initialize_wandb(cfg)

    device = torch.device("cuda")
    set_seed(seed)
    
   
   
    """ define dataset """    
    if dataset_name == "NS":
        input_dim=2
        output_dim=1
        trainset, valset, testset = create_ns_dataset(
            datapath = data_path, 
            space_factor=space_factor,
            latent_dim=latent_dim,
            split_ratios = split_ratios,
            data_type=data_type, 
            seed=seed,
        )
        
        feat_transform, feat_inv_transform = None, None
    elif dataset_name == "SOMA":
        raw_np_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/impliciBottomDrag_np'
        input_dim = 3
        output_dim = 1
        feature_set = [10]
        missing_rate = 1 - cfg.sampling.rate
        trainset, valset, testset, feat_transform,  feat_inv_transform = create_soma_dataset(
            ntrain, mmap_dir, space_factor, time_factor, 
            latent_dim, missing_rate, val_missing_rate, 
            feature_set, data_path, )
    elif dataset_name == "Burgers":
        input_dim = 1
        output_dim = 1
        trainset, valset, testset, p_mean, p_std = create_burgers_dataset()
        feat_transform, feat_inv_transform = None, None
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    ntrain = len(trainset)
    nval = len(valset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_inr)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_inr)

    print("train", len(trainset))
    print("val", len(valset))


    """ Initialize model, optimizer """
    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device=device
    )  # add sampler to model
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



    ''' Begin the sampling setting '''
    inr_sampler = InrSamplerWrapper(
        model=inr,
        iters=0,
        device='cuda',
        sample_rate=cfg.sampling.rate,
        sample_type=cfg.sampling.type,
        save_samples_path=f'./sampled_frames/{current_date_str+run_name}',
    )
    
    if cfg.sampling.type in ['3d_cluster']:
        cluster_dim = '3d'
        add_cluster_label(train_loader, 10000, 0.01, cluster_dim=cluster_dim)
        add_cluster_label(val_loader, 10000, 0.01, cluster_dim=cluster_dim)
        
        
    
    sample_params = {'sample_type': cfg.sampling.type, 'sample_rate': cfg.sampling.rate}
    sample_params_val = {'sample_type': None, 'sample_rate': 1.0}
    
    
    ''' Begin the training process '''
    for step in range(epoch_start, epochs):
        
        use_rel_loss = True
        step_show = step % 10 == 0
        step_show_last = step == epochs - 1
        
        start = time()
        train_loss, rel_train_loss = train_step(
            step, train_loader, inr, 
            sub_array_num, device, 
            inner_steps, alpha, 
            use_rel_loss, optimizer, sample_params, sampler=inr_sampler
            )
        print('time:', time()-start)

        if True in (step_show, step_show_last):
            
            save_sampling_result(step, val_loader, inr, sub_array_num, device, inner_steps, alpha, inr_sampler)
            
            test_loss, rel_test_loss = validation_step(
                step, val_loader, inr, 
                sub_array_num, device, 
                inner_steps, alpha, use_rel_loss, sample_params_val
                )

            if cfg.wandb.use_wandb:
                wandb.log(
                    {
                        "test_rel_loss": rel_test_loss,
                        "train_rel_loss": rel_train_loss,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                    },
                    step=step
                )
            loss_to_check = rel_test_loss if use_rel_loss else test_loss
            if loss_to_check < best_loss:
                best_loss = loss_to_check

                savepath = f'/pscratch/sd/g/gzhao27/INR/SOMA/results/inr_sampling/{current_date_str+run_name}.pt'
                # print('savepath:', savepath)
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr": inr.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": best_loss,
                        "alpha": alpha,
                        "feat_transform": feat_transform,
                        "feat_inv_transform": feat_inv_transform, 
                        # "grid_tr": grid_tr,
                        # "grid_te": grid_te,
                    },
                    savepath,
                )
        
    return rel_test_loss

if __name__ == "__main__":
    main()
    print('finish')