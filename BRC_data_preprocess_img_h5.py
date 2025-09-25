import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import argparse
import scanpy as sc
import anndata as ad
import h5py
from PIL import Image
import logging
import torch
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加自定义模块路径
sys.path.append("/mnt/data/scripts/src")
from process_img import *
from utils_src import *
from model_finetune import *

def create_directory(path):
    """创建目录，如果目录已存在则不会报错"""
    os.makedirs(path, exist_ok=True)

def process_h5ad_file(file_path, raw_dir, trainData_dir, n_genes, suffix, tgthighres):
    """
    处理单个 h5ad 文件并生成 h5 文件。
    """
    # 获取文件名（不带扩展名）
    name = osp.splitext(osp.basename(file_path))[0]
    trainData_h5_dir = osp.join(trainData_dir, name)

    create_directory(trainData_h5_dir)

    try:
        adata = ad.read_h5ad(file_path)
        sc.pp.filter_cells(adata, min_genes=n_genes)

        try:
            image = Image.open(osp.join(raw_dir,f"{name}_{suffix}"))
        except:
            print(f"{name} image can not open.")
            return

        for i in range(adata.shape[0]):
            row = adata.X[i, :]
            img_x = adata.obs.iloc[i]["x"]
            img_y = adata.obs.iloc[i]["y"]
            if not isinstance(row, np.ndarray):
                row = row.toarray().flatten()
            else:
                row = row.flatten()

            # 计算归一化后的 log1p 值
            totalcount = row.sum()
            if totalcount == 0:
                totalcount = 1e-5  # 避免除零错误
            normalized = (row / totalcount) * 1e4
            tmpdata = np.log1p(normalized).tolist()

            # 构建特征张量
            log_total = np.log10(totalcount)
            pretrain_gene_x = torch.tensor(
                tmpdata + [log_total + float(tgthighres[1:]), log_total]
            ).unsqueeze(0)

            # 生成基因位置 ID
            data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            # 数据打包
            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
            
            x1 = img_x*2 
            x2 = img_x*2 + crop_size*2 
            y1 = img_y*2
            y2 = img_y*2 + crop_size*2 
            patch_name = "_".join([name,str(img_x),str(img_y),str(crop_size)])

            try:
                box = (int(x1), int(y1), int(x2), int(y2))# (left, upper, right, lower)
                tile = image.crop(box)
                img_arr = np.array(tile)[np.newaxis, ...]
                img_data = torch.from_numpy(img_arr).permute(0, 3, 1, 2).float().squeeze()
            except:
                logging.warning(f"patch {patch_name} Skipping...")

            # 转换为 NumPy 数组
            x = x.numpy()
            x_padding = x_padding.numpy()
            position_gene_ids = position_gene_ids.numpy()
            img_data = img_data.numpy()

            # 保存到 h5 文件
            h5_file_path = osp.join(trainData_h5_dir, f"{patch_name}.h5")
            with h5py.File(h5_file_path, 'w') as h5f:
                h5f.create_dataset('img_data', data=img_data, compression=None, dtype=np.float32)
                h5f.create_dataset('position_gene_ids', data=position_gene_ids, compression=None, dtype=np.int16)
                h5f.create_dataset('padding', data=x_padding, compression=None, dtype=bool)
                h5f.create_dataset('xs', data=x, compression=None, dtype=np.float32)

        logging.info(f"{name} processed successfully.")
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")

def main(data_dir, raw_dir, trainData_dir, n_genes, suffix, tgthighres):
    """
    主函数，处理所有 h5ad 文件。
    """
    processed_files = [osp.join(data_dir, i) for i in search_files(data_dir, "", ".h5ad")]
    for file_path in processed_files:
        process_h5ad_file(file_path, raw_dir, trainData_dir, n_genes,suffix, tgthighres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process all samples')
    parser.add_argument('--crop_size',  type=int, default=112, help='crop size')

    args = parser.parse_args()
    crop_size =args.crop_size

    if crop_size ==112:
        file_dir = '40X'
        suffix = "HE_2X_regist.tif"
    else:
        file_dir = '20X'
    st_path = "/mnt/data/Stomics/"
    local_dir = "/mnt/data/lyx/scFoundation/model/models"
    raw_dir = osp.join(st_path, "rawdata", file_dir)
    data_dir = osp.join(st_path, "processed_h5ad", file_dir)
    trainData_dir = osp.join(st_path, "trainData", file_dir)
    img_emb_dir = osp.join(st_path, 'image_emb', file_dir)
    tx_emb_dir = osp.join(st_path, 'scFoundation_emb', file_dir)
    tx_model_path = osp.join(local_dir, "models.ckpt")
    img_model_path = osp.join(local_dir, "Uni_v2.bin")

    n_genes = 10
    num_samples = float('inf')
    tgthighres = 'a5'
    device = "cpu"
    Image.MAX_IMAGE_PIXELS = None
    pretrainmodel, pretrainconfig = load_model_frommmf(tx_model_path, "cpu", "rde")

    main(data_dir, raw_dir, trainData_dir, n_genes, suffix, tgthighres)