import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from functools import partial
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EEGDenoiser(pl.LightningModule):
    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()
        # 初始化编码器
        self.encoder = EEGTransformer(
            img_size=[58, 256*30],  # 使用标准58通道
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        # 加载预训练权重
        pretrain_ckpt = torch.load(load_path)
        encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                encoder_stat[k[15:]]=v
        self.encoder.load_state_dict(encoder_stat)
        
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 初始化重建器
        self.reconstructor = EEGTransformer(
            img_size=[58, 256*30],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        # 加载预训练重建器权重
        reconstructor_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("reconstructor."):
                reconstructor_stat[k[15:]]=v
        self.reconstructor.load_state_dict(reconstructor_stat)
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        # 1. 使用编码器提取特征
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)
            
        # 2. 使用重建器重建信号
        reconstructed = self.reconstructor(features)
        
        return reconstructed
        
    def training_step(self, batch, batch_idx):
        # 假设batch包含干净的数据作为目标
        noisy_data, clean_data = batch
        
        # 前向传播
        reconstructed = self.forward(noisy_data)
        
        # 计算损失
        loss = self.loss_fn(reconstructed, clean_data)
        
        # 记录损失
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        noisy_data, clean_data = batch
        reconstructed = self.forward(noisy_data)
        loss = self.loss_fn(reconstructed, clean_data)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        return loss
        
    def configure_optimizers(self):
        # 只优化重建器的参数
        optimizer = torch.optim.AdamW(
            self.reconstructor.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        return optimizer

# 数据加载器
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        # 加载数据
        data_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(data_path)
        
        # 转换为tensor
        data = torch.FloatTensor(data)
        
        # 这里需要根据实际情况添加噪声
        # 示例：添加高斯噪声
        noise = torch.randn_like(data) * 0.1
        noisy_data = data + noise
        
        return noisy_data, data

# 训练函数
def train_denoiser(data_dir, batch_size=32, max_epochs=100):
    # 设置随机种子
    seed_torch(7)
    
    # 创建数据集
    dataset = EEGDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = EEGDenoiser()
    
    # 创建训练器
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=max_epochs,
        logger=pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_DENOISER")
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    return model

if __name__ == "__main__":
    data_dir = "../datasets/your_data/"
    model = train_denoiser(data_dir) 