from torch.utils.data import DataLoader
import sys 
import torch
from torch import einsum
from torch import nn
sys.path.append("/mnt/DATA-4/hx/Ruipath/MunchkinCat")
sys.path.append("/mnt/DATA-4/hx/Ruipath/scFoundation/model/")
# from model_finetune import *
from data_loader import *
from models import *

# slide_ids, slide_patch_ids = Get_hest_meta()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = HESTDataset(
    data_root="/mnt/DATA-4/hx/Ruipath/hest_data"
)

local_dir = "/mnt/DATA-4/hx/Ruipath/RuiPathViT/Ruipath_visionfoundation_v1.bin"
ruipathvit = RuiPathViT(local_dir, device)
ruipathvit.build()

ckpt_path = "/mnt/DATA-4/hx/Ruipath/scFoundation/model/models/models.ckpt"
out_dim = ruipathvit.model.num_features
sc_model = scFoundation(ckpt_path, out_dim, device)
sc_model.build()

import torch.optim as optim

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)   
model = RuiPathST(ruipathvit, sc_model).cuda()

# 使用自定义 criterion（这里不需要 nn.CrossEntropyLoss）
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    running_loss = 0.0
    for i, (images, genes) in enumerate(train_loader):  # 假设 dataloader 返回图像+基因
        images = images.cuda()
        genes = genes.cuda()

        optimizer.zero_grad()
        loss = model(images, genes)  # 假设模型这样设计
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches = i + 1
        avg_loss = running_loss / num_batches
        if i % 3 == 0:
            print(f'Epoch [{epoch+1}/{3}], Step [{i+1}/{len(train_loader)}], \
                  Total Loss: {running_loss:.4f}, Avg Loss: {avg_loss:.4f}')

    # 验证部分（可选，InfoNCE 通常不直接验证 accuracy）
    print(f"Loss on epoch {epoch+1}: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "/mnt/DATA-4/hx/Ruipath/RuiPathST_ckp/model_epoch3.pth")
print(f"Model saved to /mnt/DATA-4/hx/Ruipath/RuiPathST_ckp/mnt/DATA-4/hx/Ruipath/RuiPathST_ckp/model_epoch3.pth")