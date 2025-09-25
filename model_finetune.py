import os
import os.path as osp
import h5py
import numpy as np
import timm
import pickle
from scipy.sparse import issparse
from einops import rearrange, repeat
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from select_model import *
from performer import *
from utils_src import *
from graph_data import GraphData

def save_checkpoint(model,optimizer,step,max_checkpoints_to_keep: int = 5,model_dir: str = "./"):
    """Save a checkpoint to disk.

    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.

    Parameters
    ----------
    max_checkpoints_to_keep: int
        the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        If set to zero, the function will simply return as no checkpoint is saved.
    model_dir: str, default None
        Model directory to save checkpoint to. If None, revert to self.model_dir
    """
    if max_checkpoints_to_keep == 0:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': step
    }
    temp_file = os.path.join(model_dir, 'temp_checkpoint.pt')
    torch.save(data, temp_file)

    paths = [
        os.path.join(model_dir, 'checkpoint%d.pt' % (i + 1))
        for i in range(max_checkpoints_to_keep)
    ]
    if os.path.exists(paths[-1]):
        os.remove(paths[-1])
    for i in reversed(range(max_checkpoints_to_keep - 1)):
        if os.path.exists(paths[i]):
            os.rename(paths[i], paths[i + 1])
    os.rename(temp_file, paths[0])
    
def Build_global_graph(embedding,edge_index,label=None,deepchem=True):
    """构建graph Data对象
    embedding: Model 1 得到的融合图像和转录组信息的特征，shape: n_patch*dim
    edge_index: 整个WSI的graph edge index，shape: 2*num_edge
    label：每个patch的标签,
    deepchem: 选择创建deepchem的graphData,还是pyg的Data对象
    """
    if label is None:
        node_labels = torch.zeros([embedding.shape[0]])
        label_names = 'no label'
    else:
        if isinstance(label,list):
            label_names, label = np.unique(np.array(label), return_inverse=True)
        else:
            label_names, label = np.unique(label, return_inverse=True)
        node_labels = torch.tensor(label)  # 节点标签
    if deepchem:
        return label_names,GraphData(node_features = embedding,edge_index = edge_index.numpy(), node_pos_features=node_labels.numpy())
    else:
        return label_names,Data(x=embedding,edge_index=edge_index,y=node_labels)

def get_finetune_parameter_groups(model,logit_scale,pretrain_lr=1e-5, base_lr=1e-4, img_lr=5e-5):
    params = []
    
    # 预训练模型参数组（更低学习率）
    try:
        params.append({
            'params': [p for p in model.encoder.parameters() if p.requires_grad],
            'lr': pretrain_lr,
            'weight_decay': 0.0  # 通常预训练模型不适用权重衰减
        })
        
        # 文本特征融合层（中等学习率）
        text_params = list(model.st_fc1.parameters()) #+ list(model.st_fc2.parameters())
        params.append({
            'params': text_params,
            'lr': base_lr,
            'weight_decay': 0.05
        })
        
        # 图像处理层（独立学习率）
        img_params = list(model.img_fc1.parameters()) #+ list(model.img_fc2.parameters()) 
        params.append({
            'params': img_params,
            'lr': img_lr,
            'weight_decay': 0.05
        })
    except:
        params.append({
            'params': [p for p in model.module.encoder.parameters() if p.requires_grad],
            'lr': pretrain_lr,
            'weight_decay': 0.0  # 通常预训练模型不适用权重衰减
        })
        
        # 文本特征融合层（中等学习率）
        text_params = list(model.module.st_fc1.parameters()) #+ list(model.st_fc2.parameters())
        params.append({
            'params': text_params,
            'lr': base_lr,
            'weight_decay': 0.05
        })
        
        # 图像处理层（独立学习率）
        img_params = list(model.module.img_fc1.parameters()) #+ list(model.img_fc2.parameters()) 
        params.append({
            'params': img_params,
            'lr': img_lr,
            'weight_decay': 0.05
        })

    
    # 温度参数（独立配置）
    params.append({
        'params': [logit_scale],
        'lr': 0.001,
        'weight_decay': 0.0
    })
    
    return params

def Get_Data_meta(traindata_dir,test_samples=None,num_samples=float('inf'),demo=False):
    patch_name = []
    train_file = search_files(traindata_dir,"","")
    train_file = list(set(train_file)-set(test_samples))
    all_patches = 0
    used_samples = 0
    for name in train_file:
        save_patches_path = osp.join(traindata_dir,name)
        patches = search_files(save_patches_path,"",".h5")
        if demo:
            all_patches += 100
            patch_name += patches[:100]
        else:
            all_patches += len(patches)
            patch_name += patches
        if used_samples >= num_samples:
            break
    return patch_name, all_patches

class CustomDataset(Dataset):
    def __init__(self, patch_names, arry_idx, traindata_dir, transform=None):
        self.patch_names = patch_names
        self.patch_name_idxs = arry_idx
        self.traindata_dir = traindata_dir
        self.transform = transform

    def __len__(self):
        return self.patch_name_idxs.shape[0]

    def __getitem__(self, index):
        patch_name = self.patch_names[self.patch_name_idxs[index]]
        file_path = osp.join(self.traindata_dir, patch_name.split("_")[0], patch_name)

        with h5py.File(file_path, 'r') as f:
            xs = torch.tensor(f['xs'], dtype=torch.float32)
            x_padding = torch.tensor(f['padding'], dtype=torch.bool)
            position_gene_ids = torch.tensor(f['position_gene_ids'], dtype=torch.long)
            img = torch.tensor(f['img_data'], dtype=torch.float32)

        # if self.transform:
        #     # Convert img from torch.Tensor to numpy array and then to PIL Image
        #     img_np = img.numpy().transpose((1, 2, 0))  # Assuming img shape is (C, H, W)
        #     # Check if the image data is in [0, 1] range
        #     if img_np.max() <= 1:
        #         img_pil = Image.fromarray(np.uint8(img_np * 255))
        #     else:
        #         img_pil = Image.fromarray(np.uint8(img_np))
            
        #     img_pil = self.transform(img_pil)
        #     # The transform should already convert the image back to a tensor
        #     img = img_pil

        # Apply transformations if any
        if self.transform:
            img = self.transform(img)

        return (xs, x_padding, position_gene_ids, img)

class TransformSubset(Subset):
    """继承Subset并允许重写变换规则"""
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        # 当设置了自定义变换时进行覆盖
        if self.transform:
            return self.transform(data)
        return data


def TransferWeight(source_model,target_model,layer_name_list):
    target_state_dict = target_model.state_dict()
    
    for layer_name in layer_name_list:
        source_state_dict = source_model.state_dict()
        encoder_keys = [k for k in source_state_dict.keys() if k.startswith(layer_name)]
        encoder_keys_target = [k for k in target_state_dict.keys() if k.startswith(layer_name)]
        encoder_keys = [i for i in encoder_keys if i in encoder_keys_target]

        source_state_dict = {k: v for k, v in source_state_dict.items() if k in encoder_keys}

        for k,v in source_state_dict.items():
            # print(f"{k}' weights have added")
            target_state_dict[k]=v

    target_model.load_state_dict(target_state_dict, strict=False)
    return target_model

def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1) #有表达量的基因总数
    max_num = max(value_nums)


    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels

def getEncoderData(data, config):
    encoder_data_labels = data > 0
    encoder_data, encoder_data_padding = gatherData(data, encoder_data_labels,
                                                    config['pad_token_id'])
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    
    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels

def getEncoerDecoderData(data, data_raw, config):
    # data 和data_raw 都是基因表达
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels,
                                                    config['pad_token_id'])
    #encoder多了个pad id,用来筛选基因。
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels,\
          decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids
# pretrain_gene_x.float(),有效值的位置的gene flag，padding位置判断，有效值的位置的gene id，
# pretrain_gene_x.float()，所有位置的gene flag:False，pretrain_gene_x.float()，没有mask：None，所有基因位置的gene id

def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

def load_model_frommmf(best_ckpt_path, gpu="cpu",key='gene'):
    model_data = torch.load(best_ckpt_path,map_location=gpu)
    model_data = model_data[key]
    model_data = convertconfig(model_data)

    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict)
    return model,config

def get_parameter_groups(model,logit_scale,pretrain_lr=1e-5, base_lr=1e-4, img_lr=5e-5):
    params = []
    
    # 预训练模型参数组（更低学习率）
    params.append({
        'params': [p for p in model.encoder.parameters() if p.requires_grad],
        'lr': pretrain_lr,
        'weight_decay': 0.0  # 通常预训练模型不适用权重衰减
    })
    
    # 文本特征融合层（中等学习率）
    text_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    params.append({
        'params': text_params,
        'lr': base_lr,
        'weight_decay': 0.05
    })
    
    # 图像处理层（独立学习率）
    img_params = list(model.img_fc1.parameters()) + \
                list(model.img_fc2.parameters()) 
    params.append({
        'params': img_params,
        'lr': img_lr,
        'weight_decay': 0.05
    })
    
    # 温度参数（独立配置）
    params.append({
        'params': [logit_scale],
        'lr': 0.001,
        'weight_decay': 0.0
    })
    
    return params
    
def collate_fn(batch):
    xs, x_padding,position_gene_ids, img = zip(*batch)
    # Calculate max_len based on both x and pos lengths
    max_len_x = max(x.shape[1] for x in xs)
    max_len_pos = max(len(pos) for pos in position_gene_ids)
    max_len = max(max_len_x, max_len_pos)  # Use the larger of the two
    # print( max_len_x)

    batch_size = len(xs)
    x = torch.zeros(batch_size, max_len, dtype=torch.float)
    x_padding = torch.ones(batch_size, max_len, dtype=torch.bool)
    x_pos = torch.zeros(batch_size, max_len, dtype=torch.long)

    # Fill in the tensors
    for i, (x_seq, pos_seq) in enumerate(zip(xs, position_gene_ids)):
        x_seq_tensor = torch.tensor(x_seq, dtype=torch.float)
        pos_seq_tensor = torch.tensor(pos_seq, dtype=torch.long)

        # print(pos_seq.shape[1])
        # print(x.shape)
        x[i,:pos_seq.shape[1]] = x_seq_tensor.float()
        x_padding[i, :pos_seq.shape[1]] = False
        x_pos[i, :pos_seq.shape[1]] = pos_seq_tensor

    # Process images
    img = torch.stack(img, dim=0)

    return (x,x_padding,x_pos,img)

def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

class pytorchTransformerModule(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 norm_first=False,
                 
                 # TODO: freeze most layers but unfreeze the last layers 
                 unfrozen_layers = 2
                 
                 ):
        super(pytorchTransformerModule, self).__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                     dim_feedforward=dim * ff_mult,
                                                     batch_first=True,
                                                     norm_first=norm_first,
                                                     #activation="gelu",
                                                     ))

        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)
        
        # TODO
        self.num_frozen_layers = self.depth - unfrozen_layers
        
    def forward(self, x, padding_mask):
        b, n, _, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # x get encodings [B, N, D] , batch_first is True
        count = 0
        for mod in self.transformer_encoder:
            
            if count < self.num_frozen_layers:
                with torch.no_grad():
                    x = mod(x, src_key_padding_mask=padding_mask) # , src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
            else:
                x = mod(x, src_key_padding_mask=padding_mask)
            
            count += 1
            
            # print(f"layer {str(count)}:",torch.isnan(x).any())
        # x = self.transformer_encoder(x)
        x = self.norm(x)

        return x

class ContrastModel(nn.Module):
    def __init__(self, st_input_dim, img_input_dim,out_dim, 
    pretrainmodel, img_model =None, frozenmore=True,unfrozen_layers=1):
        super().__init__()
        self.frozenmore = frozenmore
        
        if img_model is None:
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
            self.img_model = timm.create_model(**timm_kwargs)
        else:
            if isinstance(img_model, str):
                with open(img_model, 'rb') as f:
                    timm_kwargs = pickle.load(f)
                    self.img_model = timm.create_model(**timm_kwargs)
            else:
                self.img_model = img_model
        
        if isinstance(pretrainmodel, str):
            with open(pretrainmodel, 'rb') as f:
                config = pickle.load(f)
                encoder_config =config['encoder']
                encoder = select_module(config, encoder_config, config['encoder']['module_type'])
                self.token_emb = AutoDiscretizationEmbedding2(config['encoder']['hidden_dim'], config['seq_len'], config['bin_num'], config['bin_alpha'], config['pad_token_id'], config['mask_token_id'])
                self.pos_emb = nn.Embedding(config['seq_len']+1, config['encoder']['hidden_dim'])  #RandomPositionalEmbedding(embed_dim, max_seq_len)
        else:           
            self.token_emb = pretrainmodel.token_emb
            self.pos_emb = pretrainmodel.pos_emb

        self.unfrozen_layers = unfrozen_layers
        self.encoder = pytorchTransformerModule(
            max_seq_len = 15000,
            dim = st_input_dim,
            depth=12,
            heads=12,
            ff_mult=4,
            norm_first=False,
            unfrozen_layers = unfrozen_layers # TODO
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.st_fc1 = nn.Sequential(
            nn.Linear(st_input_dim*4, out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim),
        )
        
        self.img_fc1 = nn.Sequential(
            nn.Linear(img_input_dim*2, out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim),
        )
    
    def build(self):
        if self.frozenmore:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False

        for na, param in self.encoder.transformer_encoder[-(self.unfrozen_layers):].named_parameters(): 
            print('self.encoder.transformer_encoder ',na,' have grad')
            param.requires_grad = True
        
        for param in self.img_model.parameters():
            param.requires_grad = False
    
    def forward(self, x,x_padding,position_gene_ids, img):
        
        with torch.no_grad():
            
            x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            
            position_emb = self.pos_emb(position_gene_ids)
            x += position_emb
        
        x = self.encoder(x, x_padding)
        
        geneemb3, _ = torch.max(x[:,:-2,:], dim=1)
        x = torch.concat([x[:,-1,:],x[:,-2,:],geneemb3,torch.mean(x[:,:-2,:], dim=1)],axis=1)
        x = self.st_fc1(x)
        
        with torch.no_grad():
            img_feat = self.img_model.forward_features(img)
        
        img_feat = torch.cat([img_feat[:,0,:],torch.squeeze(self.pooling(img_feat[:,1:,:].permute(0, 2, 1)))], dim=1)
        img_feat = self.img_fc1(img_feat)
        
        return F.normalize(x, p=2, dim=1), F.normalize(img_feat, p=2, dim=1)
    
class GINEncoder(torch.nn.Module):
    """
    Graph Information Network (GIN) encoder. This is a graph convolutional network that produces encoded representations for molecular graph inputs.
    num_features: int
        The number of node features
    embedding_dim: int
        The dimension of the output embedding
    num_gc_layers: int, optional (default 5)
        The number of graph convolutional layers to use
    """

    def __init__(self,
                 num_features: int,
                 embedding_dim: int,
                 num_gc_layers: int = 5):
        dim = int(
            embedding_dim / num_gc_layers
        )  # the output dimension of this encoder is modified by the number of GC layers, so this is necessary to ensure that the output dimension is consistent with the InfoGraphEncoder
        super().__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(),
                                Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data):
        """
        Encodes the input graph data.

        Parameters
        ----------
        data : BatchGraphData
            The batched input graph data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the encoded representation and intermediate embeddings.
        """
        xs = []
        x = data.node_features
        # TODO: debug 0616
        # print(f"data.node_features shape: {x.shape}")

        for i in range(self.num_gc_layers):
            # print(f"conv {i} shape: {x.shape}")
            x = F.relu(self.convs[i](x, data.edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, data.graph_index) for x in xs]
        x = torch.cat(xpool, 1)
        xs = torch.cat(xs, 1)
        return x, xs
    
class SubInfoGraph(nn.Module):
    def __init__(self,
                 num_features,
                 embedding_dim,
                 num_gc_layers=5,
                 prior=True,
                 gamma=.1,
                 measure='JSD',
                 average_loss=True,
                 n_tasks: Optional[int] = None,
                 n_classes: Optional[int] = None,
                 **kwargs):
        super(SubInfoGraph, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim * num_gc_layers
        self.num_gc_layers = num_gc_layers
        self.gamma = gamma
        self.prior = prior
        self.measure = measure
        self.average_loss = average_loss
        self.localloss = LocalMutualInformationLoss()._create_pytorch_loss(measure, average_loss)
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.encoder = GINEncoder(self.num_features,self.embedding_dim,self.num_gc_layers)

        self.local_d = MultilayerPerceptron(self.embedding_dim,self.embedding_dim,(self.embedding_dim,),skip_connection=True)
        self.global_d = MultilayerPerceptron(self.embedding_dim,self.embedding_dim,(self.embedding_dim,),skip_connection=True)
        self.prior_d = MultilayerPerceptron(self.embedding_dim,1, (self.embedding_dim,),activation_fn='sigmoid')

    def forward(self, data, get_feat=False):
        y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)
        if get_feat:
            return g_enc, l_enc
        else:
            local_global_loss = self.localloss(l_enc, g_enc, data.graph_index)
            if self.prior:
                prior = torch.rand_like(y)
                term_a = torch.log(self.prior_d(prior)).mean()
                term_b = torch.log(1.0 - self.prior_d(y)).mean()
                prior = -(term_a + term_b) * self.gamma
            else:
                prior = 0
            return local_global_loss + prior