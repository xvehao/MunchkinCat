import os
import os.path as osp
import sys
# sys.path.append(" /root/miniconda3/envs/Stomics_t25/lib/python3.11/site-packages")
import argparse
import deepchem as dc
import numpy as np
import pandas as pd
import anndata as ad
import time
import logging
import datetime
logger = logging.getLogger(__name__)

import torch
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph, radius_graph, GATConv,SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler, NeighborLoader

from torch.nn import GRU, Linear, ReLU, Sequential
from typing import Iterable, List, Tuple, Optional, Dict, Literal
from deepchem.metrics import to_one_hot
import deepchem as dc

from graph_data import BatchGraphData,GraphData
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.optimizers import Adam, Optimizer, LearningRateSchedule

from torch_geometric.nn import GINConv, NNConv, global_add_pool
from torch_geometric.nn.aggr import Set2Set

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

        for i in range(self.num_gc_layers):
            print(f"conv {i} shape: {x.shape}")
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
        return label_names,GraphData(node_features = embedding,edge_index = edge_index, node_pos_features=node_labels)
    else:
        return label_names,Data(x=embedding,edge_index=edge_index,y=node_labels)
        
def main():
    parser = argparse.ArgumentParser(description='Model2: use for one sample')
    parser.add_argument('-s', '--sample', default="D04699A1", type=str,help='sample name')
    parser.add_argument('--emb_path', default=None, type=str,help='feature save path')
    parser.add_argument('--spatial_file', default=None, type=str,help='spatial infomation save path')
    parser.add_argument('--model_path', default="./", type=str,help='model2 save path')
    parser.add_argument('-g', '--gpus', default="cpu",type=str,help='number of gpus')
    parser.add_argument('--model1_batch_size', type=int, default=32, help='model1 batch size')
    parser.add_argument('--model2_batch_size', type=int, default=2, help='model2 batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-k', '--k_neighbor', type=int, default=32, help='number of neighbors')
    parser.add_argument('--spatial', action='store_true',help='if use spatial information')
    parser.add_argument('--crop_size',  type=int, default=112, help='crop size')
    parser.add_argument('--train', action='store_true', help='if train')
    parser.add_argument('--demo', action='store_true', help='if run a demo')

    args = parser.parse_args()

    if args.crop_size ==112:
        file_dir = '40X'
        suffix = "HE_2X_regist.tif"
    else:
        file_dir = '20X'
    
    st_path = "/mnt/sdb/data/BRC_stereo"
    data_dir = osp.join(st_path, "processed_h5ad", file_dir)

    config = {
    "hidden_dim": 768,
    "gat_heads": 2,
    "dropout": 0.1,
    'train_ratio':0.7,
    "grad_accum_steps": 4,
    "num_neighbors": [8, 8]  # 两层采样的邻居数
    }
    batch_size = args.model2_batch_size
    log_frequency  = 10
    checkpoint_interval = 100
    max_checkpoints_to_keep = 5

    tx_emb = np.load(osp.join(args.emb_path,f"{args.sample}_batchSize{args.model1_batch_size}_tm_features.npy"))
    img_emb = np.load(osp.join(args.emb_path,f"{args.sample}_batchSize{args.model1_batch_size}_img_features.npy"))
    patch_name = pd.read_table(osp.join(args.emb_path,f"{args.sample}_batchSize{args.model1_batch_size}_patch_names.txt"),header=None)[0].str[:-3].tolist()
    all_patches = len(patch_name)
    all_idx = np.arange(all_patches)
    cat_emb = np.concatenate((tx_emb, img_emb), axis=1)

    if args.spatial:
        have_spatial_info = False
        if args.spatial_file is not None:
            if args.spatial_file.endwith(".h5ad")==False:
                have_spatial_info = True
                print("The spatial infomation is independent of h5ad file. You should store the information as text.")
        if have_spatial_info:
            spatial_info = torch.tensor(pd.read_table(args.spatial_file,sep="\t").values,dtype=torch.float)
        else:
            adata = ad.read_h5ad(osp.join(data_dir,f"{args.sample}.h5ad"))
            spatial_info = torch.tensor(adata.obs.loc[patch_name,['x','y']].values,dtype=torch.float)
        #构建整个WSI的graph
        edge_index_sp = knn_graph(spatial_info, 
                flow='target_to_source', k=args.k_neighbor, loop=True, num_workers=8)

    label_names,data = Build_global_graph(cat_emb,edge_index_sp)

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)  # 随机打乱节点索引

    train_size = int(config['train_ratio'] * num_nodes)
    val_size = int(((1-config['train_ratio'])/2) * num_nodes)
    test_size = int(((1-config['train_ratio'])/2) * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True

    print("start to get DataLoader")

    train_loader = NeighborLoader(
        data,
        num_neighbors=config["num_neighbors"],
        batch_size=1,
        input_nodes=data.train_mask
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=config["num_neighbors"],
        batch_size=1,
        input_nodes=data.val_mask
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=config["num_neighbors"],
        batch_size=1,
        input_nodes=data.test_mask
    )

    print(f"train_loader {len(train_loader)} , done.")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus}')
        print(f"use {device}")

    model = SubInfoGraph(num_features=3072, embedding_dim=256, num_gc_layers=1, prior=False,get_feat=False)
    Optimizer = Adam(learning_rate=args.lr)
    optimizer = Optimizer._create_pytorch_optimizer(model.parameters())

    if isinstance(Optimizer.learning_rate, LearningRateSchedule):
        lr_schedule = Optimizer.learning_rate._create_pytorch_schedule(optimizer)
    else:
        lr_schedule = None

    model.train()
    all_losses =[]
    avg_loss = 0.0
    last_avg_loss = 0.0
    averaged_batches = 0
    graph_list = []
    grobal_batches = 0
    time1 = time.time()

    # # Main training loop.
    for epoch in range(args.epochs):
        for step, sub_data in enumerate(train_loader):
            new_data = GraphData(node_features = sub_data.x.numpy(),edge_index = sub_data.edge_index.numpy())#node_pos_features=sub_data.y.numpy()
            graph_list.append(new_data)

            step+=1
            if step % batch_size==0:
                batch=BatchGraphData(graph_list).numpy_to_torch(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                if lr_schedule is not None:
                    lr_schedule.step()
                avg_loss += loss
                # g_enc, l_enc = model(batch)
                graph_list = []
                # print(step)

                averaged_batches += 1
                grobal_batches += 1
            
            should_log = averaged_batches ==log_frequency
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                print(f'Ending True Step {grobal_batches}, Average loss {avg_loss}')
                if all_losses is not None:
                    all_losses.append(avg_loss)

                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and grobal_batches % checkpoint_interval == checkpoint_interval - 1:
                save_checkpoint(model,optimizer,grobal_batches,max_checkpoints_to_keep,args.model_path)

        print(f'Ending epoch {epoch}, Average loss {last_avg_loss}')
        save_checkpoint(model,optimizer,grobal_batches,max_checkpoints_to_keep,args.model_path)

        time2 = time.time()
        print("TIMING: model fitting took %0.3f s" % (time2 - time1))



if __name__=='__main__':
    main()