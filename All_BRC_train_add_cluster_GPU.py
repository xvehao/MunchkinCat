import os
import os.path as osp
import sys
import time
import datetime
import random
import pandas as pd
import numpy as np
import argparse
import scipy.sparse
from scipy.sparse import issparse
from scipy.sparse import csr_matrix, vstack, hstack
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch.npu.amp import autocast
# TODO: npu CANN=8.1.T18 0603
# from torch_npu.contrib import transfer_to_npu
# torch.npu.set_compile_mode(jit_compile=False)
# torch.npu.config.allow_internal_format = False

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import timm

sys.path.append("/mnt/data/scripts/src")
from process_img import *
from utils_src import *
from model_finetune import *
from ContrastiveLossforDDP import *

from PIL import Image
from sklearn.model_selection import train_test_split
import copy
import pickle
from contextlib import suppress

import matplotlib.pyplot as plt

def Plot(loss_dict,save_path,grid=False,log=False,smooth=False,width=10,height=5):
    if smooth:
        from scipy.ndimage import uniform_filter1d
        train_loss = uniform_filter1d(loss_dict['train loss'], size=5)
    else:
        train_loss = loss_dict['train loss']

    plt.figure(figsize=(width, height))
    if log:
        plt.semilogy(train_loss, label='Train Loss', color='blue')
    else:
        plt.plot(train_loss, label='Train Loss', color='blue')
    plt.title('Training step Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    if grid:
        plt.grid(True)
    # plt.show()
    plt.savefig(osp.join(save_path,'train_loss_curves.png'))  # 保存为图片

    plt.figure(figsize=(width, height))
    if log:
        plt.semilogy(loss_dict['epoch loss'], label='Train Loss', color='blue')
        plt.semilogy(loss_dict['val loss'], label='Validation Loss', color='orange')
    else:
        plt.plot(loss_dict['epoch loss'], label='Train Loss', color='blue')
        plt.plot(loss_dict['val loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if grid:
        plt.grid(True)
    # plt.show()
    plt.savefig(osp.join(save_path,'epoch_loss_curves.png'))


def model_train(ddp_model,optimizer,scheduler,scaler,criterion,epochs,batch_size,gpu,\
    valid_every,gradient_accumulation_steps,clip_grad,early_stop_patience,early_stop_delta, logit_scale,\
    dataloader_train,dataloader_val,demo,save_path):
    
    is_master = gpu == 0
    train_loss_list = []
    epoch_loss_list = []
    val_loss_list = []
    criterion = criterion.to(gpu)
    logit_scale = logit_scale.to(gpu,dtype=torch.float)
    
    best_val_loss = float('inf')
    
    break_flag = torch.tensor([0], device=f'npu:{gpu}')
    
    for i in range(1, epochs):
        dataloader_train.sampler.set_epoch(i)
        ddp_model.train()
        dist.barrier()
        total_loss = 0
        start_time = time.time()
        step_time = time.time()
        
        for step, (xs,x_padding,position_gene_ids,img) in enumerate(dataloader_train):
            with autocast(enabled=True, dtype=torch.float):
                tx_emb, imge_emb = ddp_model(
                    x=xs.to(gpu, dtype=torch.float),
                    x_padding=x_padding.to(gpu),
                    position_gene_ids=position_gene_ids.to(gpu),
                    img=img.to(gpu, dtype=torch.float)
                )
                loss = criterion(tx_emb, imge_emb, logit_scale, smooth=0.1) / gradient_accumulation_steps
            # TODO:0603
            context = ddp_model.no_sync() if (step % gradient_accumulation_steps != 0) else suppress()
            
            with context:
                scaler.scale(loss).backward()
        
            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            training_loss = loss.item()* gradient_accumulation_steps
            total_loss += training_loss 
            train_loss_list.append(loss.item())
            
            # TODO: 0609 print
            if step % (gradient_accumulation_steps*20) == 0 and is_master:
                elapsed_step_time = time.time() - step_time
                print(f'    ==  Epoch: {i} | step {step}| Rank {gpu} | Training Loss: {training_loss:.6f} | Time: {elapsed_step_time:.2f}s  ==')
                step_time = time.time()
            
        
        # finished all samples in the dataloader_train
        end_time = time.time()
        elapsed_time = end_time - start_time  # 单位为秒
        
        if is_master:
            lr = optimizer.param_groups[0]['lr']
            print(f'    ==  Epoch: {i} | Rank {gpu} | Training Loss: {training_loss:.6f} | LR: {lr:.2e} | Time: {elapsed_time:.2f}s  ==')
            
            if demo:
                pass
            else:
                # TODO: 0611 save module
                torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }, osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_checkpoint_epoch{i}.pth"))
        
        dist.barrier()
        scheduler.step()
        epoch_loss_list.append(total_loss)
        
        # ========= 验证阶段 =========
        if i % valid_every == 0:
            ddp_model.eval()
            dist.barrier()
            val_loss = 0.0
            
            for _, (xs_val, x_padding_val, position_gene_ids_val, img_val) in enumerate(dataloader_val):
                with autocast(enabled=True, dtype=torch.float):
                    # TODO: 0606 module to avoid hang
                    tx_emb, imge_emb = ddp_model.module(
                        xs_val.to(gpu,dtype=torch.float),
                        x_padding_val.to(gpu),
                        position_gene_ids_val.to(gpu),
                        img_val.to(gpu,dtype=torch.float)
                    )
                    val_loss += criterion(tx_emb, imge_emb, logit_scale, smooth=0.1).item()
            
            val_loss /= len(dataloader_val)
            val_loss_list.append(val_loss)

            if is_master:
                print(f"Epoch: {i} | Rank {gpu}| Train Loss: {total_loss/len(dataloader_train):.4f} | Val Loss: {val_loss:.4f}")
                
                # 早停判断逻辑
                if val_loss < (best_val_loss - early_stop_delta):
                    print(f"↳ 验证损失改进 {best_val_loss:.4f} → {val_loss:.4f}")
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(ddp_model.module.state_dict())  # 保存深度复制参数
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"↳ 早停计数: {early_stop_counter}/{early_stop_patience}")
                    if early_stop_counter >= early_stop_patience:
                        print("!!! 早停触发，终止训练 !!!")
                        
                        # TODO: 在这里直接调用break，则只有GPU 0可以break
                        # break 
                        # 应设法将break信号同步至所有的进程中
                        break_flag.fill_(1)                
            
        if is_master: #i % (epochs // 100) == 0 and 
            if demo:
                torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }, osp.join(save_path,f"batch{batch_size}_epoch{epochs}_demo_finetune_checkpoint_epoch{i}.pth"))
            else:
                # TODO: 0611 save module
                torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }, osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_checkpoint_epoch{i}.pth"))
        
        # TODO: Wenxuan Chen, 0611
        # 此处将break_flag同步至所有GPU
        dist.all_reduce(break_flag, op=dist.ReduceOp.MAX)
        
        if break_flag.item() == 1:
            if is_master:
                if demo:
                    torch.save(best_model_state, osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_demo_model.pth"))
                else:
                    torch.save(best_model_state, osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_final_model.pth"))
            break
            
    dist.barrier()
    dist.destroy_process_group()
    if is_master:
        return {"train loss":train_loss_list, "epoch loss":epoch_loss_list,"val loss":val_loss_list}


def model_val(ddp_model,criterion,epochs,batch_size,gpu,\
logit_scale,dataloader_test,demo,save_path):
    is_master = gpu == 0

    criterion = criterion.to(gpu)
    logit_scale = logit_scale.to(gpu,dtype=torch.float)

    # 测试集评估逻辑-------------------------------------------------------------
    ddp_model.eval()
    dist.barrier()
    
    test_loss = 0.0
    with torch.no_grad():
        for test_batch in dataloader_test:  # 使用测试数据集
            xs_test, x_padding_test, position_gene_ids_test, img_test = test_batch
            with autocast(enabled=True, dtype=torch.float):
                tx_emb, imge_emb = ddp_model(
                    xs_test.to(gpu,dtype=torch.float), 
                    x_padding_test.to(gpu),
                    position_gene_ids_test.to(gpu),
                    img_test.to(gpu,dtype=torch.float)
                )
                test_loss += criterion(tx_emb, imge_emb, logit_scale, smooth=0.1).item()

    # 计算结果并显示
    test_loss /= len(dataloader_test)

    if is_master:
        print(f"测试评估完成 => 平均损失: {test_loss:.4f}")

        # 保存最终模型（增加测试结果记录）------------------------------------------
        # TODO: 0611 save module
        
        if demo:
            torch.save(ddp_model.module.state_dict(), osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_demo_model.pth"))
        else:
            torch.save(ddp_model.module.state_dict(), osp.join(save_path,f"batch{batch_size}_epoch{epochs}_finetune_final_model.pth"))

        print(f"模型已保存")

    dist.barrier()
    dist.destroy_process_group()
    # return test_loss

def train(gpu,args):
    rank = args.nr * args.gpus + gpu
    torch_npu.npu.set_device(gpu)
    dist_backend = 'hccl'
    dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
    dist.init_process_group(backend=dist_backend,world_size=args.world_size, rank=rank)

    torch.backends.mha.set_fastpath_enabled(False)
    print("Fastpath enabled:", torch.backends.mha.get_fastpath_enabled())

    model = args.model

    for param in model.parameters():
        param.data = param.data.float()
    
    model = model.float().to(gpu)
    
    if args.demo:
        ddp_model = DDP(model, device_ids=[gpu],broadcast_buffers=False) #find_unused_parameters=True
    else:
        ddp_model = DDP(model, device_ids=[gpu],broadcast_buffers=False)

    patch_name, all_patches = Get_Data_meta(args.traindata_dir,[],num_samples=float('inf'),demo=args.demo)

    all_idx = np.arange(all_patches)
    train_idx, temp_idx = train_test_split(
        all_idx, 
        test_size=0.3,  # 临时保留30%
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 从中取50%作为测试
        random_state=42
    )

    # 创建带独立变换的数据子集
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ConvertImageDtype(torch.float),  # 确保数据类型为 torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),  # 确保数据类型为 torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = CustomDataset(patch_name, train_idx, args.traindata_dir,transform = train_transform)
    val_set = CustomDataset(patch_name, val_idx, args.traindata_dir,transform=val_transform)
    test_set = CustomDataset(patch_name, test_idx, args.traindata_dir,transform=val_transform)

    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,collate_fn=collate_fn,pin_memory=True)
    val_sampler = DistributedSampler(val_set, num_replicas=args.world_size, rank=rank)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,collate_fn=collate_fn,pin_memory=True)
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank)
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,collate_fn=collate_fn,pin_memory=True)

    # logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))
    logit_scale = torch.tensor(np.log(1/0.07)) # TODO: Wenxuan Chen
    logit_scale.data = torch.clamp(logit_scale, min=np.log(1e-4), max=np.log(100)).detach()

    optimizer = torch.optim.AdamW(
        get_finetune_parameter_groups(ddp_model.module,logit_scale),
        betas=(0.9, 0.98),  # 调整beta参数
        eps=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=1000),  # Warmup
            CosineAnnealingLR(optimizer, T_max=args.epochs*len(dataloader_train))  # 余弦退火
        ],
        milestones=[1000]
    )

    scaler = torch.npu.amp.GradScaler()

    # criterion = CCL_contrastive_loss(n_clusters=args.n_clusters, omega=args.omega)
    criterion = CCL_contrastive_loss_ddp(n_clusters=args.n_clusters, omega=args.omega)

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # 添加格式字符串
    print(f"start training time on rank {gpu}:{formatted_time}")

    loss_dict = model_train(ddp_model,optimizer,scheduler,scaler,criterion,args.epochs,args.batch_size,gpu,\
        args.valid_every,args.grad_acc,args.clip_grad, args.patience,args.early_stop_delta, logit_scale,\
        dataloader_train,dataloader_val,args.demo,args.save_path)
    
    if gpu == 0:
        with open(osp.join(args.save_path,f"batch{args.batch_size}_epoch{args.epochs}_CCL_finetune_loss.pkl"), 'wb') as f:
            pickle.dump(loss_dict, f)
        # Plot(loss_dict)

def test(gpu,args):
    rank = args.nr * args.gpus + gpu
    torch_npu.npu.set_device(gpu)
    dist_backend = 'hccl'
    dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
    dist.init_process_group(backend=dist_backend,world_size=args.world_size, rank=rank)

    torch.backends.mha.set_fastpath_enabled(False)
    print("Fastpath enabled:", torch.backends.mha.get_fastpath_enabled())

    if args.demo:
        best_model_state_path =  osp.join(args.save_path,f"batch{args.batch_size}_epoch{args.epochs}_finetune_demo_model.pth")
    else:
        best_model_state_path =  osp.join(args.save_path,f"batch{args.batch_size}_epoch{args.epochs}_finetune_final_model.pth")

    best_model_state = torch.load(best_model_state_path,map_location="cpu")

    model = args.model
    model.load_state_dict(best_model_state)

    print(f"已加载最佳验证性能的模型参数 at Rank {gpu}")

    for param in model.parameters():
        param.data = param.data.float()
    
    model = model.float().to(gpu)
    
    ddp_model = DDP(model, device_ids=[gpu],broadcast_buffers=False) #find_unused_parameters=True

    patch_name, all_patches = Get_Data_meta(args.traindata_dir,[],num_samples=float('inf'),demo=args.demo)

    all_idx = np.arange(all_patches)
    train_idx, temp_idx = train_test_split(
        all_idx, 
        test_size=0.3,  # 临时保留30%
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 从中取50%作为测试
        random_state=42
    )

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),  # 确保数据类型为 torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = CustomDataset(patch_name, test_idx, args.traindata_dir,transform=val_transform)
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank)
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,collate_fn=collate_fn,pin_memory=True)
    
    logit_scale = torch.tensor(np.log(1/0.07)) # TODO: Wenxuan Chen
    logit_scale.data = torch.clamp(logit_scale, min=np.log(1e-4), max=np.log(100)).detach()

    criterion = CCL_contrastive_loss_ddp(n_clusters=args.n_clusters, omega=args.omega)
    
    print(f"\n=========== 开始最终测试评估 at Rank {gpu} ===========")
    model_val(ddp_model,criterion,args.epochs,args.batch_size,gpu,\
    logit_scale,dataloader_test,args.demo,args.save_path)

####################################Settings#################################
def main():
    parser = argparse.ArgumentParser(description='train on all samples')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 1)')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--ckpt_path', default=None, type=str,help='trained model')
    parser.add_argument('--config_path', default=None, type=str,help='model config')
    parser.add_argument('--save_path', default="/mnt/data/logs/", type=str,help='result save path')
    parser.add_argument('--model_path', default="/mnt/data/lyx/scFoundation/model/models/", type=str,help='orignal model saved')
    parser.add_argument('--crop_size',  type=int, default=112, help='crop size')
    parser.add_argument('--epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--patience', type=int, default=500, help='early stop patience')
    parser.add_argument('--early_stop_delta', type=float, default=0.00001, help='early stop delta')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--n_clusters", type=int, default=4, help='Number of K-means.')
    parser.add_argument("--omega", type=float, default=0.5, help='for loss function')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--clip_grad", type=float, default=2.0, help='clip grad')
    parser.add_argument("--unfrozen_layers", type=int, default=2, help='unfrozen layers of scFoundation encoders')
    parser.add_argument('--addr',  type=str, default="80.11.129.121", help='hostname')
    parser.add_argument('--port',  type=str, default='29688', help='hostname')
    parser.add_argument('--demo', action='store_true', help='if run a demo')


    args = parser.parse_args()

    SEED = 0
    if args.crop_size ==112:
        file_dir = '40X'
        suffix = "HE_2X_regist.tif"
    else:
        file_dir = '20X'
    
    crop_size = args.crop_size

    n_genes = 10
    num_samples=float('inf')
    tgthighres = 'a5'
    device = "cpu"
    Image.MAX_IMAGE_PIXELS = None

    # TODO:DDP
    args.world_size = args.gpus * args.nodes
    gpus = int(args.gpus)
    nr = int(args.nr)

    str_gpu = []
    for i in range(nr,nr+gpus):
        str_gpu +=[str(i)]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str_gpu)
    os.environ["FORCE_TORCHRU"] = "1"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    # 通过设置 FORCE_TORCHRUN=1，可以确保即使在某些情况下（例如脚本中没有显式调用 torchrun），也会强制使用 torchrun 来启动训练程序
    # addr为当前设备IP，可在shell脚本中使用 $(hostname -I |awk '{print $1}') 获取
    os.environ['MASTER_ADDR'] = args.addr # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = args.port # 随意一个可使用的port即可

    # TODO: debug for HcclAllgather
    torch.use_deterministic_algorithms(False)
    # import mindspore
    # mindspore.set_deterministic(False) 
    # os.environ['ASCEND_LAUNCH_BLOCKING'] = "1"
    # os.environ['HCCL_DETERMINISTIC'] = 'true'
    
    os.environ['ASCEND_LAUNCH_BLOCKING'] = ""
    os.environ['HCCL_DETERMINISTIC'] = 'false'


    st_path = "/mnt/data/Stomics/"
    data_dir = osp.join(st_path,"processed_h5ad",file_dir)
    traindata_dir = osp.join(st_path,"trainData",file_dir)

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    args.traindata_dir = traindata_dir

    # img_emb_dir = osp.join(st_path,'image_emb',file_dir)
    # tx_emb_dir = osp.join(st_path,'scFoundation_emb',file_dir)

    if args.ckpt_path is None:
        if args.model_path is None:
            print("error! nothing can be done.")
        else:
            tx_model_path = osp.join(args.model_path,"models.ckpt")
            img_model_path = osp.join(args.model_path, "Uni_v2.bin")
            timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
                }

            img_model_config_path = timm.create_model(**timm_kwargs)
            img_model_config_path.load_state_dict(torch.load(img_model_path, map_location="cpu"), strict=True)
            img_model_config_path.eval()
            tx_model_config_path,pretrainconfig = load_model_frommmf(tx_model_path,"cpu","rde")
    else:
        if args.config_path is not None:
            tx_model_config_path = osp.join(args.config_path,"scFoundation_config.pkl")
            img_model_config_path = osp.join(args.config_path,"Uni_config.pkl")
        else:
            print("error! nothing can be done.")

    random.seed(SEED)
    np.random.seed(SEED)  # numpy random generator
    torch.manual_seed(SEED)

    model = ContrastModel(st_input_dim=768, img_input_dim=1536, out_dim=1536,
            img_model=img_model_config_path, 
            pretrainmodel = tx_model_config_path,
            frozenmore=True,
            unfrozen_layers = args.unfrozen_layers)

    if args.ckpt_path is None:
        model = TransferWeight(tx_model_config_path,model,["encoder."]) #"decoder."
    else:
        model_data = torch.load(args.ckpt_path,map_location="cpu")
        model.load_state_dict(model_data)

    model.build()
    args.model = model

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # 添加格式字符串
    print(f"start timing:{formatted_time}")

    mp.spawn(train,nprocs=gpus,args=(args,))

    mp.spawn(test,nprocs=gpus,args=(args,))

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # 添加格式字符串
    print(f"end timing:{formatted_time}")

if __name__=='__main__':
    main()