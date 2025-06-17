import torch
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))
from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
import argparse
from train_utility_sampling.train_utility import (
    split_graph_by_time,
    train_step,
    validation_step, 
    save_sampling_result,
    train_step_single_image,
    )
from utils.data.unstructure_dataset import (
    GraphNavierStokes, 
    collate_graph_inr, 
    GraphSomaDataset, 
    GraphBurgers, 
    GraphNavierStokesSampling,
    get_graph_t_idx,
    )
from coralsoma.load_modulations import graph_ode_inr_predict, load_graph_modulations, load_soma_graph_modulations_each_frame
# sys.path.append(str(Path(__file__).parents[1]))
# sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
from utils.load_inr import create_inr_instance, load_inr_model

from coral.mlp import Derivative
from coral.utils.models.scheduling import ode_scheduling
from torchdiffeq import odeint
from torch_geometric.data import DataLoader as GDataLoader
import numpy as np
from train_utility_sampling.SamplerWrapper import (
    InrSamplerWrapper,
    graph_3d_cluster,
    graph_2d_cluster,
    add_cluster_label,
    sample_random_node_indices_per_cluster,
    INRSingle2dSamplerWrapper,
    graph_2d_cluster_single_image,
)
from mmap_ninja import RaggedMmap
from hydra import initialize, compose


NS_inr_save_name = 'NS_keep_for_test_file'
NS_inr_save_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/best_result/'
device = torch.device('cuda')
# seed = 42
# import random
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)



def set_dummy_config(model_type='siren'):
    config_path = "../config"  # or the appropriate relative/absolute path

    with initialize(config_path=config_path):
        cfg = compose(config_name="inr_sample.yaml",)
    
    cfg.inr.hidden_dim = 32
    cfg.inr.depth = 3
    cfg.inr.w0 = 30
    cfg.inr.model_type = model_type
    
    return cfg


def build_dummy_dataset():
    splits = slice(5)
    datapath = "/pscratch/sd/g/gzhao27/INR/data/NS2d/ns_mmap"
    images_mmap = RaggedMmap(datapath)
    raw = images_mmap[2]
    train_array = raw[..., splits]
    trainset = GraphNavierStokesSampling(
        raw=train_array, 
        missing_rate=0, 
    )
    return trainset

def test_burgers():
    print('Burgers test')
    datapath_dict  = {
        '/pscratch/sd/g/gzhao27/INR/data/1D_Burgers_Sols_Nu0.01.hdf5':([1, 2, 4], 0.01)
    }
    trainset = GraphBurgers(
        datapath_dict=datapath_dict,
        latent_dim=128,
        missing_rate=0.5,
    )
    
    graph = trainset[0]
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=2, shuffle=True, )
    graph = next(iter(train_loader))
    assert graph.cor.size(-1) == 1, 'feature size is not correct'
    assert graph.space_emb.size(-1) == 1
    

def test_soma():
    print('SOMA test')
    device = torch.device('cuda')
    
    trainset = GraphSomaDataset(
        data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        train_num=10, 
        feature_set=[10],
        space_factor=4,
        time_factor=2, 
        latent_dim=256,
    )
    feat_ori = trainset[0].feat
    
    feat_nor, inv_nor = trainset.create_normalize_from_dataset()
    trainset.update_feat_transform(feat_nor)

    graph = trainset[0]
    feat_new = trainset[0].feat
    
    assert (inv_nor(feat_new)-feat_ori).var() < 1e-5, 'the inverse transform is not correct'
    
    
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=3, shuffle=True, )
    graph = next(iter(train_loader))
    graph = graph.to(device)
    
    ori_feat = inv_nor(graph.feat)
    assert ori_feat.size(-1) == 1
    
    assert graph.cor.size(-1) == 3, 'length of graph is not correct'
    assert graph.feat.size(-1) == 1, 'feature size is not correct'
    
    savepath = f'/pscratch/sd/g/gzhao27/INR/SOMA/test.pt'
    torch.save(
        {
            "feat_transform": feat_nor,
            "feat_inv_transform": inv_nor, 
            # "grid_tr": grid_tr,
            # "grid_te": grid_te,
        },
        savepath,
    )
    
    inr_save_name = '2024-12-31SOMA-inr_w013_lr5.6e-05_depth6_hdim155-best-train-loss'
    inr_save_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results'
    data_to_encode = None
    input_dim=3
    output_dim=1
    inr, alpha = load_inr_model(
            inr_save_dir,
            inr_save_name,
            data_to_encode,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    
    z_mean, z_std, trainset = load_soma_graph_modulations_each_frame(
        inr_save_name=inr_save_name,
        inr=inr,
        trainset=trainset,
        type='train',
        device=device,
        inner_steps=3,
        alpha=alpha,
    )
    assert z_std > 0
    
    train_loader = GDataLoader(dataset=trainset, batch_size=3, shuffle=True, )
    graph = next(iter(train_loader))
    assert len(graph) == 3
    print('finish')
    

def test_ns():
    print("NS test")
    trainset = GraphNavierStokes(datanum=10, latent_dim=128)

    assert trainset.mask.shape == (10, 64, 64, 50), f"Expected shape (10, 64, 64, 50), but got {trainset.mask.shape}"
    assert len(trainset[0].feat) == len(trainset[0].time), "Expected length to be same, "


    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=10, shuffle=True, )
    graph = next(iter(train_loader))
    # assert graph.latent_vector[graph.time].size(-1) == 128, "index time for latent vector"

    input_dim = graph.cor.size(-1)
    output_dim = graph.feat.size(-1)
    
    NS_inr_path = os.join(NS_inr_save_dir, NS_inr_save_name+'.pt')
    tmp = torch.load(NS_inr_path, weights_only=False)
    latent_dim = tmp["cfg"].inr.latent_dim
    space_factor = tmp["cfg"].data.space_factor
    seed = tmp["cfg"].data.seed
    
    trainset = GraphNavierStokes(split='train', ssub=space_factor, datanum=100, missing_rate=0.5, latent_dim=latent_dim)
    
    inr, alpha = load_inr_model(
        NS_inr_save_dir,
        NS_inr_save_name,
        data_to_encode=None, 
        input_dim=input_dim,
        output_dim=output_dim,
    )
    load_graph_modulations(
        trainset,
        inr,
        inner_steps=3,
        alpha=alpha,
        batch_size=4,
    )
    
    timestamps_train = torch.arange(0, 50, 1).float().cuda()
    modulations = torch.rand(100, latent_dim).cuda()
    modulations[:50] = 1
    
    model = Derivative(1, latent_dim, hidden_c=512, depth=3).cuda()
    modulations = modulations.reshape(2, 50, latent_dim)
    modulations = modulations.permute(0, 2, 1)
    z_pred = ode_scheduling(odeint, model, modulations, timestamps_train, epsilon=0)
    assert z_pred.shape == (2, 128, 50)
    
    outputs = graph_ode_inr_predict(model, inr, graph)


def test_inr_sample(sample_type):
    print('inr sample test, with sample type:', sample_type)
    
    
    splits = slice(5)
    
    datapath = "/pscratch/sd/g/gzhao27/INR/data/NS2d/ns_mmap"
    images_mmap = RaggedMmap(datapath)
    raw = images_mmap[2]
    train_array = raw[..., splits]
    
    trainset = GraphNavierStokesSampling(
        raw=train_array, 
        missing_rate=0, 
    )
    
    latent_dim = 128
    
    trainset.initial_latent_vector(latent_dim)
    assert trainset[0].feat.shape == (50*64*64, 1)
    
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=len(trainset), shuffle=False, )
    graph = next(iter(train_loader))
    # assert graph.latent_vector[graph.time].size(-1) == 128, "index time for latent vector"

    input_dim = graph.cor.size(-1)
    output_dim = graph.feat.size(-1)
    
    inr, alpha = load_inr_model(
        NS_inr_save_dir,
        NS_inr_save_name,
        data_to_encode=None, 
        input_dim=input_dim,
        output_dim=output_dim,
    )
    
    sample_params = {
        'sample_type': 'random',
        'sample_rate': 0.5, 
    }
    
    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters()},
        ],
        lr=0.001,
        weight_decay=0,
    )
    
    # sampled_graph = inr.sample(graph, sample_params)
    # assert sampled_graph.feat.shape == (50*int(64*64*0.5)*5, 1)
    
    if sample_type is None: 
        N = 5
        sub_graph_list = split_graph_by_time(sampled_graph, N)
        assert sub_graph_list[0].feat.shape == (50*int(64*64*0.5)*5/N, 1)
        
        
        
        # graph = graph.to(device)
        train_loss, rel_train_loss = train_step(0, train_loader, inr, N, device, inner_steps=3, alpha=0.2, use_rel_loss=True,
                optimizer=optimizer, sample_params=sample_params)
        print('train loss:', train_loss)
        print('rel train loss:', rel_train_loss)
        assert train_loss < 0.1671 and train_loss > 0.1669
    # train loss: 0.1670
    
    elif sample_type == 'NMT':
        N = 1
        inr_sampler = InrSamplerWrapper(
            model=inr,
            iters=0,
            device='cuda',
            sample_type='NMT',
            sample_rate=0.01,
            save_samples_path='./sampled_frames',
            save_interval=100,
        )
        sampled_graph = inr_sampler.sample(outer_step = 10, inner_step=0, graph=graph, modulations=graph.latent_vector, save_image=True)
        
        save_sampling_result(
            step=12, val_loader=train_loader, inr=inr, sub_array_num=1, device=device, inner_steps=3, 
            alpha=alpha, sampler=inr_sampler,
        )
        # inr_sampler._save_sample_images(graph, sampled_graph)
        # assert sampled_graph.feat.shape == (50*int(64*64*0.125)*5, 1)
        train_loss, rel_train_loss = train_step(0, train_loader, inr, N, device, inner_steps=3, alpha=0.2, use_rel_loss=True,
                optimizer=optimizer, sample_params=sample_params, sampler=inr_sampler)
        print('train loss:', train_loss)
        print('rel train loss:', rel_train_loss)
        
    elif sample_type == '3d_cluster':
        sub_array_num = 1
        cluster_dim = '3d'
        add_cluster_label(train_loader, 10000, 0.01, cluster_dim=cluster_dim)
        graph = next(iter(train_loader))
        inr_sampler = InrSamplerWrapper(
            model=inr,
            iters=0,
            device='cuda',
            sample_type='3d_cluster',
            sample_rate=0.001,
            save_samples_path='./sampled_frames',
            save_interval=100,
        )
        sampled_graph = inr_sampler.sample(outer_step = 3, inner_step=0, graph=graph, modulations=graph.latent_vector, save_image=True)
        save_sampling_result(
            step=12, val_loader=train_loader, inr=inr, sub_array_num=1, device=device, inner_steps=3, 
            alpha=alpha, sampler=inr_sampler,
        )
        
        
        train_loss, rel_train_loss = train_step(0, train_loader, inr, sub_array_num, device, inner_steps=3, alpha=0.2, use_rel_loss=True,
                optimizer=optimizer, sample_params=sample_params, sampler=inr_sampler)
        print('train loss:', train_loss)
        print('rel train loss:', rel_train_loss)
    return locals()
        
def test_single_image_inr(sample_type=None):
    print('test single image inr')
    cfg = set_dummy_config(model_type='single_image_siren')
    cfg.sampling.cluster_type = 'grid'
    cfg.sampling.type = sample_type
    trainset = build_dummy_dataset()
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=1, shuffle=False, )
    graph = next(iter(train_loader))
    
    inr = create_inr_instance(cfg, input_dim=2, output_dim=1, device=device)
    
    t = 10
    indices_t = get_graph_t_idx(graph, t)
    
    graph = Data(
        cor=graph.cor[indices_t],
        feat=graph.feat[indices_t],
        time=torch.zeros(len(indices_t)),  # set time to 0 tensor
        space_emb=graph.space_emb[indices_t],
        T=torch.tensor(1),
    )
    if cfg.sampling.type in ['2d_cluster']:
        graph_2d_cluster_single_image(graph, 100, 0.01, cluster_type=cfg.sampling.cluster_type)
        
    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters()},
        ],
        lr=0.001,
        weight_decay=0,
    )
    
    if cfg.sampling.type is not None:
        inr_sampler = INRSingle2dSamplerWrapper(
            model=inr,
            iters=0,
            device='cuda',
            sample_rate=cfg.sampling.rate,
            sample_type=cfg.sampling.type,
            save_samples_path=f'./sampled_frames',
        )
    else:
        inr_sampler = None
        
    loss, rel_loss = train_step_single_image(
        0,
        graph, 
        inr, 
        device=device,
        use_rel_loss=True,
        optimizer=optimizer,
        sampler=inr_sampler,
    )
    if sample_type == '2d_cluster':
        inr_sampler = INRSingle2dSamplerWrapper(
            model=inr,
            iters=0,
            device='cuda',
            sample_rate=0.2,
            sample_type=cfg.sampling.type,
            save_samples_path=f'./sampled_frames',
        )
        
        graph_ori = graph
        sample_graph = inr_sampler.sample(
            inner_step=0, 
            graph=graph_ori, 
            save_image=True
        )
        n_samples = max(int(len(graph_ori.feat) * cfg.sampling.rate), 1)
        assert len(sample_graph.cor) == n_samples
    
    # loss, rel_loss = train_step_single_image(
    #         0,
    #         graph, 
    #         inr, 
    #         device=device,
    #         use_rel_loss=True,
    #         optimizer=optimizer,
    #         sampler=None,
    #     )
    
    # if sample_type is None:
    #     loss, rel_loss = train_step_single_image(
    #         0,
    #         graph, 
    #         inr, 
    #         device=device,
    #         use_rel_loss=True,
    #         optimizer=optimizer,
    #         sampler=None,
    #     )
    # elif sample_type == '2d_cluster':
    #     inr_sampler = InrSingle2dSamplerWrapper(
    #         model=inr,
    #         iters=0,
    #         device='cuda',
    #         sample_rate=cfg.sampling.rate,
    #         sample_type=cfg.sampling.type,
    #         save_samples_path=f'./sampled_frames',
    #     )
        
    #     loss, rel_loss = train_step_single_image(
    #         0,
    #         graph, 
    #         inr, 
    #         device=device,
    #         use_rel_loss=True,
    #         optimizer=optimizer,
    #         sampler=inr_sampler,
    #     )
        
        
    
    print('loss:', loss)
    print('rel loss:', rel_loss)
    
    
    return locals()
    
    

    
def test_3d_cluster():    
    print('test 3d cluster')
    trainset = build_dummy_dataset()
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=len(trainset), shuffle=False, )
    cluster_dim = '3d'
    add_cluster_label(train_loader, 1000, 0.01, cluster_dim=cluster_dim)
    graph = next(iter(train_loader))
    sampled_3d = sample_random_node_indices_per_cluster(graph, cluster_dim=cluster_dim)
    W, H = graph.cor.max(axis=0)[0] +1
    cl_num = 0
    for gidx in range(len(graph)):
        cl_num += len(graph.cluster_set[gidx])
    assert len(sampled_3d) == cl_num
    assert sampled_3d.min() < W*H
    assert sampled_3d.max() > W*H*249

def test_2d_cluster():
    print('test 2d cluster')
    trainset = build_dummy_dataset()
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=len(trainset), shuffle=False, )
    cluster_dim = '2d'
    add_cluster_label(train_loader, 100, 0.1, cluster_dim=cluster_dim)
    graph = next(iter(train_loader))
    sampled_2d = sample_random_node_indices_per_cluster(graph, cluster_dim=cluster_dim)
    W, H = graph.cor.max(axis=0)[0] +1
    cl_num = 0
    T_total = graph.T.sum()
    for t in range(T_total):
        cl_num += len(graph.cluster_set[t])
    assert len(sampled_2d) == cl_num
    assert sampled_2d.min() < W*H
    assert sampled_2d.max() > W*H*249
    
    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='test name')
    parser.add_argument('--sample_type', type=str, default=None, help='sample type')

    args = parser.parse_args()

    if args.name == "NS":
        test_ns()
    elif args.name == "SOMA":
        test_soma()
    elif args.name == "Burgers":
        test_burgers()
        
    if args.name in ['inr_sampling', 'INR_sampling', 'inr_sample', 'INR_sample']:
        test_inr_sample(args.sample_type)
    
    if args.name in ['3d_cluster']:
        test_3d_cluster()
        
    if args.name in ['2d_cluster']:
        test_2d_cluster()
        
    if args.name in ['single_image_inr', 'single_image_INR']:
        test_single_image_inr(args.sample_type)

    print('end')