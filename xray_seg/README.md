# X-Ray Segmentation Project

这个项目是一个X光图像分割项目，使用深度学习方法进行骨盆X光图像分割。
This repository contains an X-ray image segmentation project focusing on pelvis X-ray segmentation using deep learning approaches.

## 项目结构 Project Structure

```
xray_seg/
├── data/             
│   ├── raw/         
│   ├── processed/   
│   └── results/     
│
├── src/             
│   ├── models/      # 模型定义 model definitions
│   ├── predict/     # 预测代码 prediction code
│   └── utils/       # 工具函数 utility functions
│       └── pengwin_utils.py  # 数据处理工具 data processing utilities
│
├── notebooks/       
│   └── U_net.ipynb
│
└── README.md         
```

## 环境配置 Environment Setup

### 前置要求 Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU加速 for GPU acceleration)

### 安装步骤 Installation

1. 克隆仓库 Clone the repository:
   ```bash
   git clone <repository_url>
   cd xray_seg
   ```

2. 安装依赖 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明 Usage

### 数据准备 Data Preparation

1. 将原始X光图像放在 `data/raw` 目录下
   Place raw X-ray images in the `data/raw` directory

2. 运行数据预处理脚本
   Run data preprocessing script:
   ```bash
   python src/preprocess.py
   ```

### 训练模型 Training

使用 U-Net 模型进行训练：
Train using U-Net model:

```bash
jupyter notebook notebooks/U_net.ipynb
```

这个笔记本包含：
This notebook includes:
- 数据加载和预处理 Data loading and preprocessing
- U-Net 模型定义 Model definition
- 训练循环和可视化 Training loop and visualization

### 预测 Prediction

使用训练好的模型进行预测：
Use trained model for prediction:

```python
from src.predict import predict

# 加载模型 Load model
model = predict.load_model('path/to/model/weights')

# 进行预测 Make prediction
result = predict.segment_image('path/to/image')
```

## 项目结构说明 Directory Details

- `data/`: 存放数据集，包括原始数据、处理后的数据和结果
  Contains datasets, including raw data, processed data and results
  
- `src/models/`: 包含模型架构定义
  Contains model architecture definitions
  
- `src/predict/`: 包含用于预测的代码
  Contains code for making predictions
  
- `notebooks/`: 包含用于实验和训练的Jupyter notebooks
  Contains Jupyter notebooks for experiments and training

## 贡献 Contributing

欢迎提交 Pull Requests 来改进项目。
Feel free to submit Pull Requests to improve the project.