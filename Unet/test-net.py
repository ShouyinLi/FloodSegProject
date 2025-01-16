from base_unet import SecondFloodNet
from improved_unet import ThirdFloodNet
from normal_unet import FirstFloodNet
import os
import numpy as np
import torch
import torch.nn as nn
import tifffile
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

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

def test_best_model(best_model_path, dataset, device, indices_to_test=[0, 50, 100, 150, 200, 250, 300]):
    """
    使用最佳模型测试指定索引的图像，并详细记录结果
    
    Args:
        best_model_path (str): 最佳模型的路径
        dataset (Dataset): 测试数据集
        device (torch.device): 运行设备
        indices_to_test (list): 要测试的图像索引列表
    """
    # 创建结果保存目录
    results_dir = 'test_results_improved'  # 修改为你的结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(results_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'predicts'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    
    # 初始化存储指标的列表
    all_metrics = []
    
    # 加载模型
    model = ThirdFloodNet(num_classes=1, input_channels=12).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 测试指定索引的图像
    with torch.no_grad():
        for idx in indices_to_test:
            # 获取图像和标签
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # 增加batch维度
            mask = mask.numpy()  # 转为numpy以便可视化
            
            # 模型预测
            if model.gt_ds:
                _, output = model(image)
            else:
                output = model(image)
            
            # 后处理预测结果
            pred = output.cpu().squeeze().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # 准备 BGR 图像
            # 使用第 2、3、4 个通道作为 BGR
            image_np = image.cpu().squeeze().numpy()
            bgr_image = np.transpose(image_np[[1,2,3]], (1,2,0))
            
            # 标准化 BGR 图像以便显示
            bgr_image = (bgr_image - bgr_image.min()) / (bgr_image.max() - bgr_image.min())
            
            # 保存 BGR 图像
            plt.imsave(
                os.path.join(results_dir, 'images', f'image_idx_{idx}.png'), 
                bgr_image
            )
            
            # 保存标签
            plt.imsave(
                os.path.join(results_dir, 'labels', f'label_idx_{idx}.png'), 
                mask.squeeze(), 
                cmap='gray'
            )
            
            # 保存预测结果
            plt.imsave(
                os.path.join(results_dir, 'predicts', f'predict_idx_{idx}.png'), 
                pred_binary.squeeze(), 
                cmap='gray'
            )
            
            # 可视化和保存结果
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title(f'Original Image (BGR)')
            plt.imshow(bgr_image)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title(f'Ground Truth Mask')
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title(f'Predicted Mask')
            plt.imshow(pred_binary.squeeze(), cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, 'visualizations', f'test_result_idx_{idx}.png'), 
                dpi=300
            )
            plt.close()
            
            # 计算评价指标
            metrics = calculate_metrics(
                torch.from_numpy(pred).unsqueeze(0), 
                torch.from_numpy(mask).unsqueeze(0)
            )
            
            # 添加索引信息
            metrics['index'] = idx
            all_metrics.append(metrics)
            
            print(f"Test Metrics for Index {idx}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print("\n")
    
    # 将指标保存为 CSV 和 Excel
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)
    metrics_df.to_excel(os.path.join(results_dir, 'test_metrics.xlsx'), index=False)
    
    # 计算和打印总体统计
    print("Overall Metrics Summary:")
    summary_metrics = {
        'mean_accuracy': metrics_df['accuracy'].mean(),
        'mean_precision': metrics_df['precision'].mean(),
        'mean_recall': metrics_df['recall'].mean(),
        'mean_miou': metrics_df['miou'].mean(),
        'mean_f1_score': metrics_df['f1_score'].mean(),
        
        'std_accuracy': metrics_df['accuracy'].std(),
        'std_precision': metrics_df['precision'].std(),
        'std_recall': metrics_df['recall'].std(),
        'std_miou': metrics_df['miou'].std(),
        'std_f1_score': metrics_df['f1_score'].std()
    }
    
    for metric, value in summary_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 返回指标数据框，以便进一步处理
    return metrics_df

def main():
    # 设置设备
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    # 数据集路径
    image_folder = 'data/images'
    label_folder = 'data/labels'
    
    # 创建数据集
    dataset = RemoteSensingDataset(image_folder, label_folder)
    
    # 测试最佳模型
    test_best_model(
        best_model_path='best_model.pth', # 修改为你的最佳模型路径 normal_best_model.pth, improved_best_model.pth,base_best_model.pth
        dataset=dataset, 
        device=device,
        indices_to_test=[0, 50, 100, 150, 200, 250, 300]
    )

if __name__ == '__main__':
    main()