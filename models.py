import timm
import sys 
import torch
from torch import einsum
import torch.nn.functional as F
from torch import nn
sys.path.append("/mnt/DATA-4/hx/Ruipath/MunchkinCat")
sys.path.append("/mnt/DATA-4/hx/Ruipath/scFoundation/model/")
from load import *

class RuiPathViT(nn.Module):

    def __init__(self, ckpt_path, device):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device

    def build(self):
        self.model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        embed_dim = self.model.num_features
        self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim*2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim*2, embed_dim)
            )
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device), strict=True)
        for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, x, *args, **kwargs):
        features = self.model(x, *args, **kwargs)
        return self.head(features)
    
    def construct_optimizers(self, head_lr=1e-3):
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=head_lr,
            weight_decay=1e-2,
        )
        return optimizer


class scFoundation(nn.Module):

    def __init__(self, ckpt_path, out_dim, device, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.out_dim = out_dim
        self.device = device

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            print('self.encoder.transformer_encoder ',na,' have grad')
            param.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], self.out_dim*2),
        nn.ReLU(),
        nn.Linear(self.out_dim*2, self.out_dim)
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, x, *args, **kwargs):
        
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        geneemb = self.encoder(x,x_padding)
        geneembmerge, _ = torch.max(geneemb, dim=1)
        
        return self.fc1(geneembmerge)

class RuiPathST(nn.Module):
    def __init__(self, vision_model, sc_model):
        super().__init__()
        self.vision_model = vision_model
        self.sc_model = sc_model
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.device = vision_model.device
    
    def embed_img(self, img):
        return self.vision_model(img)
    
    def embed_gene(self, gene_expr):
        return self.sc_model(gene_expr)

    def forward(self, img, gene_expr):
        batch_size = img.shape[0]
        img_features = self.embed_img(img)
        gene_features = self.embed_gene(gene_expr)
        ce = F.cross_entropy
        sim = einsum('i d, j d -> i j', img_features, gene_features)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch_size, device=self.device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        return contrastive_loss