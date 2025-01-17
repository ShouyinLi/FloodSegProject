# FloodSegProject
使用了多种方法的来解决洪水卫星图像分割任务的仓库
以下是各个网络的框架图
RandomForest/diabetes.png
FloodSegProject/
│
├── DATA/             # 数据集管理
│   ├── raw/          # 原始图像数据
│   ├── processed/    # 预处理后的数据
│   └── annotations/  # 标注信息
│
├── Transformer/      # Transformer模型实现
│   ├── model.py      
│   └── train.py     
│
├── UNET/             # U-Net模型
│   ├── model.py
│   └── train.py
│
├── RandomForest/     # 随机森林方法
│   ├── model.py
│   └── diabetes.png  # 模型框架图
│
├── DataTransformer/  # 数据预处理
│   ├── augmentation.py
│   └── preprocessing.py
│
└── README.md
