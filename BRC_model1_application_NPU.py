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
# import torch_npu
# from torch.npu.amp import autocast
# TODO: npu CANN=8.1.T18 0603
# from torch_npu.contrib import transfer_to_npu
# torch.npu.set_compile_mode(jit_compile=False)
# torch.npu.config.allow_internal_format = False

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm
import tqdm

sys.path.append("/mnt/data/scripts/src")
from process_img import *
from utils_src import *
from model_finetune import *
from BRC_data_preprocess_img_h5 import *

from PIL import Image
import copy
import pickle
from contextlib import suppress

####################################Settings#################################
def main():
    parser = argparse.ArgumentParser(description='use for one sample')
    parser.add_argument('-s', '--sample', default="D04699A1", type=str,help='sample name')
    parser.add_argument('--ckpt_path', default=None, type=str,help='trained model')
    parser.add_argument('--ckpt', action='store_true', help='if load the ckpt or model')
    parser.add_argument('--config_path', default=None, type=str,help='model config')
    parser.add_argument('--save_path', default=None, type=str,help='feature save path')
    parser.add_argument('--pre_processed', action='store_true', help='if processed before input')
    parser.add_argument('-g', '--gpus', default="cpu",type=str,help='number of gpus')
    parser.add_argument('--crop_size',  type=int, default=112, help='crop size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--demo', action='store_true', help='if run a demo')

    args = parser.parse_args()

    n_genes = 10
    tgthighres = 'a5'
    torch.set_num_threads(128)  
    os.environ['OPENBLAS_NUM_THREADS'] = '128'
    print("Change the setting.")
    Image.MAX_IMAGE_PIXELS = None

    if args.crop_size ==112:
        file_dir = '40X'
        suffix = "HE_2X_regist.tif"
    else:
        file_dir = '20X'
    
    crop_size = args.crop_size

    st_path = "/mnt/data/Stomics/"
    raw_dir = osp.join(st_path,"rawdata",file_dir)
    data_dir = osp.join(st_path,"processed_h5ad",file_dir)
    traindata_dir = osp.join(st_path,"trainData",file_dir)
    tx_model_config_path = osp.join(args.config_path,"scFoundation_config.pkl")
    img_model_config_path = osp.join(args.config_path,"Uni_config.pkl")
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # 添加格式字符串
    print(f"start timing:{formatted_time}")

    if args.pre_processed:
        pass
    else:
        file_path =osp.join(raw_dir,f"{args.sample}_count_112.h5ad")
        process_h5ad_file(file_path, data_dir, traindata_dir, n_genes,suffix, tgthighres)

    save_patches_path = osp.join(traindata_dir,args.sample)
    patch_name = search_files(save_patches_path,"",".h5")
    all_patches = len(patch_name)
    all_idx = np.arange(all_patches)

    if args.demo:
        all_patches = 100
        all_idx = all_idx[:all_patches]
        patch_name = patch_name[:all_patches]
        print(f"Use {args.sample} {all_idx.shape[0]} patches.")

    torch.backends.mha.set_fastpath_enabled(False)
    print("Fastpath enabled:", torch.backends.mha.get_fastpath_enabled())

    model = ContrastModel(st_input_dim=768, 
            img_input_dim=1536, out_dim=1536, 
            img_model=img_model_config_path, 
            pretrainmodel = tx_model_config_path,
            frozenmore=True, unfrozen_layers =1)
    
    model_data = torch.load(args.ckpt_path,map_location="cpu")

    if args.ckpt:
        model.load_state_dict(model_data['model'])
    else:
        model.load_state_dict(model_data)

    if torch.npu.is_available():
        NPU_FLAG = True
        try:  
            gpus = int(args.gpus)
            device = torch.device(f'npu:{args.gpus}')
            print(f"Use NPU {args.gpus}.")
        except:
            NPU_FLAG = False

    if NPU_FLAG == False:
        print("Use CPU.")
        device = torch.device('cpu')
        model = model.to(device)
    else:
        for param in model.parameters():
            param.data = param.data.float()
        model = model.float().to(device)

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)  # numpy random generator
    torch.manual_seed(SEED)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),  # 确保数据类型为 torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_set = CustomDataset(patch_name, all_idx, traindata_dir,transform=val_transform)

    dataloader_val = DataLoader(val_set, batch_size=args.batch_size,collate_fn=collate_fn,shuffle=False, pin_memory=True)

    model.eval()
    tx_emb_list = []
    imge_emb_list = []

    with tqdm(total=all_patches,
                desc="Get features",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        with torch.no_grad():
            for _, (xs_val, x_padding_val, position_gene_ids_val, img_val) in enumerate(dataloader_val):
                tx_emb, imge_emb = model(
                    xs_val.to(device,dtype=torch.float),
                    x_padding_val.to(device),
                    position_gene_ids_val.to(device),
                    img_val.to(device,dtype=torch.float)
                )
                
                tx_emb_list.append(tx_emb.detach().cpu())
                imge_emb_list.append(imge_emb.detach().cpu())
                pbar.update(1)
    # tm_emb = np.squeeze(np.array(tx_emb_list))
    # img_emb = np.squeeze(np.array(imge_emb_list))
    tm_emb = torch.cat(tx_emb_list,dim=0).numpy()
    img_emb = torch.cat(imge_emb_list,dim=0).numpy()

    np.save(osp.join(args.save_path,f"{args.sample}_batchSize{args.batch_size}_tm_features.npy"),tm_emb)
    np.save(osp.join(args.save_path,f"{args.sample}_batchSize{args.batch_size}_img_features.npy"),img_emb)

    pd.DataFrame(patch_name).to_csv(osp.join(args.save_path,f"{args.sample}_batchSize{args.batch_size}_patch_names.txt"),sep="\t",header=None,index=None)

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # 添加格式字符串
    print(f"end timing:{formatted_time}")
    print("done!")

if __name__=='__main__':
    main()