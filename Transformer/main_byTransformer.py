import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import timm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,jaccard_score
import random
from matplotlib.colors import ListedColormap

# 定义超参数
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 128
TRAIN_RATIO = 0.8  # 训练集占比

# 定义数据集类
class ImageSegDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = os.listdir(img_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # 读取多通道tif图像
        with rasterio.open(img_path) as src:
            image = src.read()
        
        image = self.custom_normalize(image)

        # 读取png标签图像
        label = plt.imread(label_path)
        label[label > 0] = 1

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label
    
    def custom_normalize(self, image):
        channels, height, width = image.shape
        # 计算每个通道的均值和标准差
        mean = np.mean(image, axis=(1, 2)).reshape((channels, 1, 1))
        std = np.std(image, axis=(1, 2)).reshape((channels, 1, 1))

        # 避免除以零的情况
        std = np.where(std == 0, 1e-8, std)

        # 在原始形状上进行归一化
        normalized_image = (image - mean) / std

        return normalized_image

# 自定义基于Transformer架构的128*128输入尺寸的图像分割模型类
class CustomTransformerSegModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomTransformerSegModel, self).__init__()

        # 定义Patch Embedding层，适配128*128输入尺寸和12通道图像
        self.patch_embed = PatchEmbed(
            img_size=(128, 128),
            patch_size=(16, 16),
            in_channels=in_channels,
            embed_dim=768
        )

        # 定义Transformer Encoder层
        self.encoder = nn.ModuleList([
        timm.models.vision_transformer.Block(
            dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm
            ) for _ in range(12)
        ])
        
        self.norm = nn.LayerNorm(768)

         # 解码器部分
        self.lateral_conv = nn.Conv2d(768, 256, kernel_size=1)  # 1x1卷积调整通道数
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样到32x32
        self.fpn_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 3x3卷积进行特征融合
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样到128x128

        # 最终的分类卷积层
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        # 加入位置编码
        pos_embed = self.get_pos_embed(x)
        x = x + pos_embed
        # 通过Transformer Encoder
        for layer in self.encoder:
            x = layer(x)

        # 归一化
        x = self.norm(x)
        
        # 调整形状以匹配卷积层的输入
        batch_size, num_patches, embed_dim = x.shape
        height = int(num_patches ** 0.5)
        width = height
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, embed_dim, height, width)
        # 解码
        x = self.lateral_conv(x)  # 调整通道数
        x = self.upsample1(x)  # 上采样到32x32
        x = self.fpn_conv(x)  # 特征融合
        x = self.upsample2(x)  # 最终上采样到128x128

        # 最终的分类
        x = self.final_conv(x)
        return x

    def get_pos_embed(self, x):
        """
        生成位置编码
        """
        batch_size, num_patches, embed_dim = x.shape
        pe = torch.zeros((1, num_patches, embed_dim), device=x.device)
        position = torch.arange(0, num_patches).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


# 定义Patch Embedding层的类
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 计算Patch数量
        num_patches_h = img_size[0] // patch_size[0]
        num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = num_patches_h * num_patches_w

        # 定义卷积层进行Patch划分和特征嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # 进行Patch划分和特征嵌入
        x = self.proj(x)
        # 调整维度顺序，方便后续处理
        x = x.flatten(2).transpose(1, 2)
        return x

# 创建数据集和数据加载器
img_dir = 'images'
label_dir = 'labels'
dataset = ImageSegDataset(img_dir, label_dir)

# 划分训练集和测试集
total_size = len(dataset)
train_size = int(total_size * TRAIN_RATIO)
test_size = total_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 实例化模型、损失函数和优化器
model = CustomTransformerSegModel(in_channels=12, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 用于存储每个epoch的训练损失、测试准确率、精确率、召回率,F1分数以及IOU分数
train_losses = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []
test_iou_scores = []

# 训练循环
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_loss}')

    # 在测试集上进行评估
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds = preds.flatten()
            all_preds.extend(preds)

            labels = labels.cpu().numpy()
            labels = labels.flatten()
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    iou = jaccard_score(all_labels, all_preds, average='weighted')

    test_accuracies.append(accuracy)
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_f1_scores.append(f1)
    test_iou_scores.append(iou)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, IOU: {iou}')

# 保存模型（可选）
torch.save(model.state_dict(), 'transformer_seg_model.pth')

# 绘制训练损失变化曲线
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.clf()

# 绘制测试准确率、精确率、召回率、F1分数和IOU变化曲线
plt.plot(test_accuracies, label='Test Accuracy')
plt.plot(test_precisions, label='Precision')
plt.plot(test_recalls, label='Recall')
plt.plot(test_f1_scores, label='F1 Score')
plt.plot(test_iou_scores, label='IOU')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Accuracy, Precision, Recall, F1 Score and IOU')
plt.legend()
plt.savefig('metrics.png')
plt.clf()


# 从测试集中随机抽取四张图像进行可视化
random_indices = random.sample(range(len(test_dataset)), 4)

model = CustomTransformerSegModel(in_channels=12, num_classes=2)
model.load_state_dict(torch.load('transformer_seg_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
with torch.no_grad():
    for idx in random_indices:
        image_tensor, label = test_dataset[idx]

        # # 可视化原图（按照特定RGB组合合成真彩色图像）
        # rgb_combination = [3, 2, 1]
        # rgb_image = np.stack([image_tensor[:, :, i] for i in rgb_combination], axis=-1)
        # plt.figure(figsize=(15, 10))
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb_image / np.percentile(rgb_image, 98))
        # plt.title('Original Image')
        # plt.axis('off')

        # 可视化真实标签
        custom_cmap = ListedColormap(['yellow', 'blue'])
        plt.subplot(1, 2, 1)
        plt.imshow(label.cpu().numpy(), cmap=custom_cmap)
        plt.title('Ground Truth')
        plt.axis('off')

        # 可视化预测结果
        image_tensor = image_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap=custom_cmap)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.savefig(f'output/image_visualization_{idx}.png')
        
        plt.show()