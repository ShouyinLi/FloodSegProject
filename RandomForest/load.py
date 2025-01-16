import os
import rasterio
import numpy as np
import pandas as pd
import cv2

# 存储所有图像数据的列表
all_data = []

# 逐个读取文件
for i in range(0, 49):
    file_path = f'D:/kese/Track2/Track2/train/images/{i}.tif'
    if os.path.exists(file_path):  # 确保文件存在
        with rasterio.open(file_path) as src:
            channels_data = np.stack([src.read(j + 1) for j in range(12)], axis=0)
        data_flat = channels_data.reshape(channels_data.shape[0], -1).T
        all_data.append(data_flat)

# 创建 DataFrame 并保存为 CSV
df_images = pd.DataFrame(np.vstack(all_data))
df_images.columns = list('ABCDEFGHIJKL')
df_images.to_csv('train_0_to_49.csv', index=False, header=False)

# 逐个读取图像
all_labels = []

for i in range(0, 49):
    file_path = f'D:/kese/Track2/Track2/train/labels/{i}.png'
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:  # 检查图像是否成功读取
        data_flat = image.flatten()
        all_labels.append(data_flat)

# 创建 DataFrame 并保存为 CSV
df_labels = pd.DataFrame(all_labels)
df_labels.to_csv('label_0_to_49.csv', index=False, header=False)

# 将label_0_to_49.csv添加到train_0_to_49.csv的最后一列
df_images['label'] = df_labels.values
df_images.to_csv('combined_data.csv', index=False)