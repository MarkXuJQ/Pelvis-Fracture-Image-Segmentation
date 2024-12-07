# CT Segmentation Project

这个项目是一个CT图像分割项目，使用深度学习方法进行骨盆CT图像分割。
This repository contains a CT image segmentation project focusing on pelvis CT segmentation using deep learning approaches.

## 项目结构 Project Structure

```
ct_seg/
│
├── data/             
│   ├── raw/         # 原始CT图像数据 Raw CT image data
│   ├── processed/   # 预处理后的数据 Processed data
│   └── results/     # 模型输出结果 Model outputs
│
├── src/             
│   ├── models/      # 模型定义 Model definitions
│   └── predict/     # 预测代码 Prediction code
│
├── notebooks/       
│
└── README.md        
```

## 环境配置 Environment Setup

### 前置要求 Prerequisites

- Python 3.10+
- CUDA 11.8+ (用于GPU加速 for GPU acceleration)

### 安装步骤 Installation

1. 克隆仓库 Clone the repository:
   ```bash
   git clone <repository_url>
   cd ct_seg
   ```

2. 安装依赖 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明 Usage

### 训练模型 Training

使用notebooks目录下的训练脚本来训练模型：
Use the training scripts in the notebooks directory to train the model.


### 预测 Prediction

使用 `src/predict` 目录下的代码进行预测：
Use the code in `src/predict` directory for making predictions.


## 项目结构说明 Directory Details

- `data/`: 存放数据集，包括原始数据和处理后的数据
  Contains datasets, including both raw and processed data
  
- `src/models/`: 包含模型架构定义
  Contains model architecture definitions
  
- `src/predict/`: 包含用于预测的代码
  Contains code for making predictions
  
- `src/results/`: 存储模型输出结果
  Stores model output results
  
- `notebooks/`: 包含用于实验和训练的Jupyter notebooks
  Contains Jupyter notebooks for experiments and training

## 贡献 Contributing

欢迎提交 Pull Requests 来改进项目。
Feel free to submit Pull Requests to improve the project.