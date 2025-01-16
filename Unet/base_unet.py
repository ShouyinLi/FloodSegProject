import os
import torch
import numpy as np
import tifffile
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv
import torch
import logging
from datetime import datetime

def setup_logging():
    """
    设置日志记录
    """
    log_dir = 'base/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def save_checkpoint(model, optimizer, epoch, metrics, filepath='base/checkpoints'):
    """
    保存模型检查点
    
    Args:
        model (nn.Module): 要保存的模型
        optimizer (torch.optim): 优化器
        epoch (int): 当前训练轮次
        metrics (dict): 模型性能指标
        filepath (str): 检查点保存路径
    """
    os.makedirs(filepath, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(filepath, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(filepath, model, optimizer=None):
    """
    从检查点恢复训练
    
    Args:
        filepath (str): 检查点文件路径
        model (nn.Module): 模型
        optimizer (torch.optim, optional): 优化器
    
    Returns:
        tuple: 起始轮次和最佳指标
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metrics = checkpoint.get('metrics', None)
    
    return start_epoch, best_metrics

def log_metrics_to_csv(metrics, filepath='training_metrics.csv', mode='a'):
    """
    将训练指标记录到CSV文件
    
    Args:
        metrics (dict): 训练指标
        filepath (str): CSV文件路径
        mode (str): 文件写入模式
    """
    file_exists = os.path.exists(filepath)
    
    with open(filepath, mode, newline='') as csvfile:
        fieldnames = list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)

def normalize_channel(channel):
    """
    标准化单个通道
    Args:
        channel (np.ndarray): 输入通道数据
    Returns:
        np.ndarray: 归一化后的通道数据
    """
    min_val = channel.min()
    max_val = channel.max()
    # 使用安全的归一化方法，避免除零错误
    if min_val == max_val:
        return np.zeros_like(channel, dtype=np.uint8)
    return ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def save_prediction_images(images, masks, preds, epoch, batch_idx, save_dir='base-predictions'):
    """
    保存预测图像和真实图像的对比
    
    Args:
        images (torch.Tensor): 输入图像，期望形状为 (B, C, H, W)
        masks (torch.Tensor): 真实标签
        preds (torch.Tensor): 预测标签
        epoch (int): 当前轮次
        batch_idx (int): 批次索引
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 限制保存图像数量
    num_samples = min(4, images.size(0))
    
    for i in range(num_samples):
        try:
            # 安全地提取图像、掩码和预测
            img = images[i].cpu().numpy()
            mask = masks[i].cpu().numpy().squeeze()
            pred = preds[i].cpu().numpy().squeeze()
            
            # 处理多通道图像
            if img.ndim == 3:
                # 选择RGB通道（如果存在）
                if img.shape[0] >= 3:
                    rgb_image = img[1:4, :, :]
                else:
                    rgb_image = img[:3, :, :]
                
                # 归一化RGB通道
                rgb_normalized = np.stack([
                    normalize_channel(rgb_image[2]),
                    normalize_channel(rgb_image[1]),
                    normalize_channel(rgb_image[0])
                ], axis=2)
            else:
                rgb_normalized = img
            
            # 标准化掩码和预测
            mask = (mask * 255).clip(0, 255).astype(np.uint8)
            pred = (pred * 255).clip(0, 255).astype(np.uint8)
            
            # 创建子图
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title(f'Input Image (Epoch {epoch})')
            plt.imshow(rgb_normalized)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth')
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title('Prediction')
            plt.imshow(pred, cmap='gray')
            plt.axis('off')
            
            # 保存图像
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
        
        except Exception as e:
            print(f"Error saving image {i}: {e}")

# Metrics Calculation
def calculate_metrics(pred, target):
    """
    计算分割任务的评价指标
    
    Args:
        pred (torch.Tensor): 预测的分割图，二值化后的tensor
        target (torch.Tensor): 真实的分割图，二值化后的tensor
    
    Returns:
        dict: 包含Accuracy, Precision, Recall, mIoU, F1 Score的指标
    """
    # 确保预测和目标tensor是二值的
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # 展平tensor
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positives, False Positives, False Negatives
    tp = torch.sum((pred == 1) & (target == 1))
    fp = torch.sum((pred == 1) & (target == 0))
    fn = torch.sum((pred == 0) & (target == 1))
    tn = torch.sum((pred == 0) & (target == 0))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision (避免除零)
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    
    # Intersection over Union (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else torch.tensor(0.0)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'miou': iou.item(),
        'f1_score': f1.item()
    }

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7], reduction_ratio=4):
        super().__init__()
        # 引入注意力机制
        self.channel_attention = ChannelAttentionModule(dim_xl)
        
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        
        # 压缩通道数以减少计算复杂度
        reduced_dim = dim_xl // reduction_ratio
        # print("融合:", reduced_dim)
        # 多尺度特征提取模块
        self.multi_scale_module = nn.ModuleList()
        for d in d_list:
            scale_block = nn.Sequential(
                LayerNorm(normalized_shape=reduced_dim, data_format='channels_first'),
                nn.Conv2d(reduced_dim, reduced_dim, kernel_size=k_size, 
                        stride=1, padding=(k_size+(k_size-1)*(d))//2-1, 
                        dilation=d, groups=reduced_dim)
            )
            self.multi_scale_module.append(scale_block)
        
        # 融合模块
        self.fusion_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl, data_format='channels_first'),
            nn.Conv2d(dim_xl, dim_xl, 1),
            nn.GELU()  # 使用更先进的激活函数
        )
        
        # 残差连接
        self.residual_connection = nn.Conv2d(dim_xl, dim_xl, 1)
    
    def forward(self, xh, xl, mask):
        # 预处理
        # print("融合卷积后维度:", xh.shape)
        xh = self.pre_project(xh)
        # print("融合卷积后维度:", xh.shape)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        # print("融合卷积后维度:", xh.shape)
        
        # 通道注意力
        xl = self.channel_attention(xl)
        
        # 多尺度特征提取
        multi_scale_features = []
        for module in self.multi_scale_module:
            # print("维度:", torch.chunk(xl, 4, dim=1)[0].shape)
            feature = module(torch.chunk(xl, 4, dim=1)[0])
            multi_scale_features.append(feature)
            # print("融合卷积后维度:", feature.shape)
        
        # 特征融合
        
        x = torch.cat(multi_scale_features, dim=1)
        # print("融合卷积后维度:", x.shape)
        x = self.fusion_conv(x)
        # print("融合卷积后维度:", x.shape)
        # 残差连接
        # print("残差连接分支维度:", self.residual_connection(xl).shape)
        x = x + self.residual_connection(xl)
        
        return x

# 添加通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction_ratio, channel, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention

class SecondFloodNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=12, c_list=[16,32,48,64,96,128], 
                 bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        # 调整初始编码器以适应128*128输入
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[0]),
            nn.GELU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64*64

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[1]),
            nn.GELU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32*32

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[2]),
            nn.GELU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16*16

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[3]),
            nn.GELU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)  # 8*8

        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[4]),
            nn.GELU()
        )
        self.pool5 = nn.MaxPool2d(2, 2)  # 4*4

        self.encoder6 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[5]),
            nn.GELU()
        )  # 4*4

        # 桥接模块
        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        
        # 深度监督
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], 1, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], 1, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], 1, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], 1, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], 1, 1)
            print('gt deep supervision was used')
        
        # 解码器
        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[4]),
            nn.GELU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[3]),
            nn.GELU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[2]),
            nn.GELU()
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[1]),
            nn.GELU()
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(c_list[0]),
            nn.GELU()
        )

        # 最终输出卷积
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码过程
        x1 = self.encoder1(x)
        x1_pooled = self.pool1(x1)
        
        x2 = self.encoder2(x1_pooled)
        x2_pooled = self.pool2(x2)
        
        x3 = self.encoder3(x2_pooled)
        x3_pooled = self.pool3(x3)
        
        x4 = self.encoder4(x3_pooled)
        x4_pooled = self.pool4(x4)
        
        x5 = self.encoder5(x4_pooled)
        x5_pooled = self.pool5(x5)
        
        x6 = self.encoder6(x5_pooled)

        # 解码过程
        out5 = self.decoder1(x6)
        out5 = F.interpolate(out5, size=x5.shape[2:], mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(x6, x5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t5 = self.GAB5(x6, x5)
        out5 = out5 + t5

        out4 = self.decoder2(out5)
        out4 = F.interpolate(out4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(out5, x4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t4 = self.GAB4(out5, x4)
        out4 = out4 + t4

        out3 = self.decoder3(out4)
        out3 = F.interpolate(out3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(out4, x3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t3 = self.GAB3(out4, x3)
        out3 = out3 + t3

        out2 = self.decoder4(out3)
        out2 = F.interpolate(out2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(out3, x2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(out3, x2)
        out2 = out2 + t2

        out1 = self.decoder5(out2)
        out1 = F.interpolate(out1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(out2, x1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=1, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(out2, x1)
        out1 = out1 + t1

        # 最终输出
        out0 = F.interpolate(self.final(out1), scale_factor=1, mode='bilinear', align_corners=True)
        
        if self.gt_ds:
            return (
                torch.sigmoid(gt_pre5), 
                torch.sigmoid(gt_pre4), 
                torch.sigmoid(gt_pre3), 
                torch.sigmoid(gt_pre2), 
                torch.sigmoid(gt_pre1)
            ), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)
# Custom Dataset
class RemoteSensingDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像（12个通道）
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = tifffile.imread(image_path)  # 使用 tifffile 读取
        
        # 读取 PNG 标签
        label_path = os.path.join(self.label_folder, self.label_files[idx])
        label = np.array(Image.open(label_path))
        
        # 转换为 float32 并为标签添加通道维度
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        # 如果图像是 (H, W, C)，转换为 (C, H, W)
        if image.ndim == 3 and image.shape[-1] == 12:
            image = np.transpose(image, (2, 0, 1))
        
        # 处理标签
        # 对于二值分割，通常需要归一化
        if label.ndim == 2:
            label = (label > 0).astype(np.float32)  # 二值化
            label = np.expand_dims(label, axis=0)
        elif label.ndim == 3 and label.shape[-1] == 3:
            # 如果是RGB图像，转换为灰度
            label = label[:,:,0]  # 或使用更复杂的转换
            label = (label > 0).astype(np.float32)
            label = np.expand_dims(label, axis=0)
        
        # 可选：应用变换
        if self.transform:
            image, label = self.transform(image, label)
        
        return torch.from_numpy(image), torch.from_numpy(label)

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        intersection = (pred * target).sum(dim=(1,2))
        union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
        dice_coef = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coef.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        """
        带权重的交叉熵损失函数
        
        Args:
            weight (torch.Tensor, optional): 类别权重
            ignore_index (int, optional): 忽略的类别索引
            reduction (str, optional): 损失reduction方法
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        计算带权重的交叉熵损失
        
        Args:
            pred (torch.Tensor): 模型预测 (B, 1, H, W)
            target (torch.Tensor): 目标标签 (B, 1, H, W)
        
        Returns:
            torch.Tensor: 损失值
        """
        # 预处理输入
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        # 计算正负类别权重
        pos_weight = self.calculate_class_weight(target)
        
        # 使用带权重的二值交叉熵
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, 
            reduction=self.reduction
        )
        
        return criterion(pred, target)
    
    def calculate_class_weight(self, target):
        """
        计算类别权重，平衡正负样本
        
        Args:
            target (torch.Tensor): 目标标签
        
        Returns:
            torch.Tensor: 正类权重
        """
        num_neg = torch.sum(1 - target)
        num_pos = torch.sum(target)
        
        # 避免除零
        if num_pos == 0:
            return torch.tensor(1.0).to(target.device)
        
        # 计算权重：负类样本数 / 正类样本数
        return num_neg / (num_pos + 1e-8)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=3000, device='cuda:3', 
                resume_checkpoint=None, log_dir='logs'):
    """
    修改后的训练函数，支持从检查点恢复训练和日志记录
    """
    logger = setup_logging()
    
    # 如果提供检查点，从检查点恢复
    start_epoch = 0
    best_metrics = None
    
    if resume_checkpoint:
        start_epoch, best_metrics = load_checkpoint(
            resume_checkpoint, model, optimizer
        )
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {
            'accuracy': 0.0, 'precision': 0.0, 
            'recall': 0.0, 'miou': 0.0, 'f1_score': 0.0
        }
        
        # Mixed Precision Training
        scaler = GradScaler()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward with autocast
            with autocast():
                if model.gt_ds:
                    ds_outputs, output = model(images)
                    loss = 0
                    for ds_out in ds_outputs:
                        loss += criterion(ds_out, masks)
                    loss += criterion(output, masks)
                    loss /= (len(ds_outputs) + 1)
                else:
                    output = model(images)
                    loss = criterion(output, masks)
            
            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 计算训练指标
            batch_metrics = calculate_metrics(output.detach(), masks)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
        
        # 平均训练指标
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'accuracy': 0.0, 'precision': 0.0, 
            'recall': 0.0, 'miou': 0.0, 'f1_score': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                if model.gt_ds:
                    ds_outputs, output = model(images)
                    loss = 0
                    for ds_out in ds_outputs:
                        loss += criterion(ds_out, masks)
                    loss += criterion(output, masks)
                    loss /= (len(ds_outputs) + 1)
                else:
                    output = model(images)
                    loss = criterion(output, masks)
                
                val_loss += loss.item()
                
                # 计算验证指标
                batch_metrics = calculate_metrics(output, masks)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
                
                # 随机保存一些预测图像
                if batch_idx % 200 == 0:  # 每5个batch保存一次
                    save_prediction_images(images, masks, output, epoch, batch_idx)
        
        # 平均验证指标
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print('Train Metrics:', train_metrics)
        print('Val Metrics:', val_metrics)
        
        # Save best model based on mIoU
        if val_metrics['miou'] > (best_metrics['miou'] if best_metrics else 0):
            best_val_loss = val_loss
            best_metrics = val_metrics
            torch.save(model.state_dict(), 'normal_best_model.pth')
    
            # 记录训练和验证指标到CSV
        log_metrics_to_csv({
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader),
            **train_metrics,
            **val_metrics
        }, filepath=os.path.join(log_dir, 'metrics.csv'))
        if(epoch%50==0):
            save_checkpoint(model, optimizer, epoch, val_metrics)
        
        # 记录日志
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val Metrics: {val_metrics}')
    
    return best_metrics

# Main Execution
def main():
    # 设置设备
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    # 路径
    image_folder = 'data/images'
    label_folder = 'data/labels'
    
    # 创建数据集
    dataset = RemoteSensingDataset(image_folder, label_folder)
    
    # 拆分数据集
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    # 数据加载器
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)
    
    # 模型
    model = SecondFloodNet(num_classes=1, input_channels=12).to(device)
    
    # 损失和优化器
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    # 训练
        # 可选：从检查点恢复训练
    resume_checkpoint = None  # 如果要恢复训练，在此处指定检查点文件路径
    train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, 
        resume_checkpoint=resume_checkpoint
    )

if __name__ == '__main__':
    main()