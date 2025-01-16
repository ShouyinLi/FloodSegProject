import os
import cv2
import rasterio
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score

# 定义字母标签
letter_labels = [chr(ord('A') + i) for i in range(12)]
letter_labels.append('label')

def process_image_data(image_num):
    # 读取图像数据
    image_data = []
    label_data = []

    # 读取图片数据
    image_file_path = f'D:/kese/Track2/Track2/train/images/{image_num}.tif'
    if os.path.exists(image_file_path):  # 确保文件存在
        with rasterio.open(image_file_path) as src:
            channels_data = np.stack([src.read(j + 1) for j in range(12)], axis=0)
        image_data = channels_data.reshape(channels_data.shape[0], -1).T

    # 读取标签数据
    label_file_path = f'D:/kese/Track2/Track2/train/labels/{image_num}.png'
    if os.path.exists(label_file_path):  # 确保文件存在
        label_image = cv2.imread(label_file_path, cv2.IMREAD_GRAYSCALE)
        if label_image is not None:  # 检查图像是否成功读取
            label_data = label_image.flatten()

    return image_data, label_data

# 读取数据、处理并构建 DataFrame
def read_process_save_data(start_image_num, end_image_num):
    all_image_data = []
    all_label_data = []

    for i in range(start_image_num, end_image_num + 1):
        image_data, label_data = process_image_data(str(i))
        if len(image_data) > 0 and len(label_data) > 0:
            all_image_data.append(image_data)
            all_label_data.append(label_data)

    all_image_data = np.concatenate(all_image_data)
    all_label_data = np.concatenate(all_label_data)
    image_data_with_label = np.column_stack([all_image_data, all_label_data])
    df = pd.DataFrame(image_data_with_label, columns=letter_labels)
    df.to_csv('images_50_to_300.csv', index=False)

start_image_num = 50
end_image_num = 300
read_process_save_data(start_image_num, end_image_num)