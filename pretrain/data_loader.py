import os
import h5py
import anndata as ad
from torch.utils.data import Dataset
from torchvision import transforms
import torch

def Get_hest_meta(hest_dir="/mnt/DATA-4/hx/Ruipath/hest_data"):
    hest_h5_path = hest_dir + "/patches/"
    filenames = os.listdir(hest_h5_path)
    h5files = [filename for filename in filenames if filename.endswith(".h5")]
    slide_patch_id = []
    slide_ids = []
    for h5_file in h5files:
        slide_id = h5_file.replace(".h5", "")
        h5_path = os.path.join(hest_h5_path, h5_file)
        with h5py.File(h5_path, 'r') as f:
            num_patches = f['img'].shape[0]  # 假设数据存在 'imgs' 键下
        slide_patch_id.extend([(slide_id, i) for i in range(num_patches)])
        slide_ids.append(slide_id)

    return slide_ids, slide_patch_id

# class HESTDataset(Dataset):
#     def __init__(self, ids, root_dir, transform=None):
#         self.ids = ids
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         id = self.ids[idx]
#         h5path = f"{self.root_dir}/patches/{id}.h5"
#         with h5py.File(h5path, 'r') as f:
#             img = f['img'][:]
#             st_path =  f"{self.root_dir}/st/{id}.h5ad"
#             adata = ad.read_h5ad(st_path)

#         if self.transform:
#             img = self.transform(img)

#         return (img, adata)


class HESTDataset(Dataset):
    def __init__(self, data_root, 
                 img_trans=None, gene_trans=None,
                 patch_dir="patches", gene_dir="processed_st"):
        self.data_root = data_root
        self.patch_dir = os.path.join(data_root, patch_dir)
        self.gene_dir = os.path.join(data_root, gene_dir)
        self.img_trans = img_trans
        self.gene_trans = gene_trans
        _, self.index = Get_hest_meta(data_root)
        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        slide_id, patch_idx = self.index[idx]
        
        # 1. read image
        patch_path = os.path.join(self.patch_dir, f"{slide_id}.h5")
        with h5py.File(patch_path, 'r') as f:
            patch = f['img'][patch_idx]  # shape (H, W, C) or (C, H, W)

        if self.img_trans:
            # if need to convert to PIL Image
            from torchvision.transforms import ToPILImage
            image = ToPILImage()(patch)
            image = self.img_trans(image)
        else:
            img_trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            image = img_trans(patch)

        # 2. read gene vector
        st_path = os.path.join(self.gene_dir, f"{slide_id}.h5")
        with h5py.File(st_path, 'r') as f:
            gene_vector = f['st_processed'][patch_idx] 
        if self.gene_trans:
            gene_vector = self.gene_trans(gene_vector)
        else:
            gene_vector = torch.from_numpy(gene_vector).float()      

        return image, gene_vector
    
    # def _load_gene_vector(self, slide_id, patch_idx):
    #     gene_path = os.path.join(self.gene_dir, f"{slide_id}.h5ad")
    #     with h5py.File(gene_path, 'r') as f:
    #         # 检查是否是 sparse
    #         if isinstance(f['X'], h5py.Group):
    #             # sparse matrix (CSR)
    #             data = f['X/data'][:]
    #             indices = f['X/indices'][:]
    #             indptr = f['X/indptr'][:]
    #             from scipy.sparse import csr_matrix
    #             shape = f['X'].attrs['shape']
    #             mat = csr_matrix((data, indices, indptr), shape=shape)
    #             vec = mat[patch_idx].toarray().flatten()
    #         else:
    #             # dense matrix
    #             vec = f['X'][patch_idx]
    #     return vec