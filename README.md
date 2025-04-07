# Pelvis Fracture Image Segmentation

大创项目申请书：
[面向精准骨外科手术的多模态医学图像分析算法及系统](Project_plan.md)

 <img src="image/logo.png" width="300">

## 项目结构

```
pelvis_seg/
├── ct_seg/                 # CT分割模块
│   ├── data/               # CT数据目录
│   ├── tools/              # CT相关工具
│   └── training_code/      # 训练代码
│
├── xray_seg/               # X光分割模块
│   ├── data/               # X光数据目录
│   ├── training_code/          # Jupyter notebooks
│   └── src/                # 源代码
│       └── utils/          # 工具函数
│
├── system/                 # 系统核心模块
│   ├── config/             # 云主机配置文件
│   ├── database/           # 数据库相关
│   ├── medical_viewer/     # 医学图像查看器
│   │   └── segmenters/     # 分割器
│   │
│   ├── models/             # 模型定义
│   ├── ui/                 # 用户界面
│   └── utils/              # 通用工具
│
├── weights/                # 模型权重
│   ├── DeeplabV3/
│   ├── hub/
│   │   └── checkpoints/
│   ├── MedSAM/
│   └── U-net/
│
├── image/                  # 项目图片资源
│   └── plan/
│
├── requirements.txt        # pip依赖
├── environment.yml         # conda环境配置
├── README.md               # 项目说明
└── .gitignore             # Git忽略文件
```

## Installation Guide | 安装指南

### Prerequisites | 环境要求
- Python 3.10 (Required | 必需)
- CUDA 12.1 (Optional for GPU acceleration | 可选,用于GPU加速)
- Windows/Linux/MacOS

### Installation Steps | 安装步骤

1. Clone the repository | 克隆仓库
```bash
git clone https://github.com/your-username/pelvis_seg.git
cd pelvis_seg
```
2. Create and activate the conda environment | 创建并激活 conda 环境
```bash
# Create environment from yml file | 从 yml 文件创建环境
conda env create -f environment.yml

# Activate environment | 激活环境
conda activate pelvis_seg
```
or mannually install it | 手动安装
```bash
# Create and activate environment | 创建并激活环境
conda create -n pelvis_seg python=3.10
conda activate pelvis_seg

# Install PyTorch with CUDA support | 安装PyTorch(CUDA支持)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies | 安装其他依赖
pip install -r requirements.txt
```


### Using Mirror Sources (Optional) | 使用镜像源（可选）

For users in China | 中国用户可使用以下命令加速安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


### Troubleshooting | 常见问题

如遇安装问题，请查看我们的 [GitHub Issues](https://github.com/your-username/pelvis_seg/issues) 或创建新的问题。

### 支持的文件格式
- DICOM (.dcm)
- NIfTI (.nii, .nii.gz)
- NRRD (.nrrd)
- MetaImage (.mha, .mhd)

### 数据来源 | Data Sources

#### CT 图像数据集 | CT Image Dataset
来自 PENGWIN Task 1 骨盆骨折CT图像分割挑战赛的训练数据集，包含100例带有骨盆骨折的CT扫描及其分割标注。
From PENGWIN Task 1 Pelvic Fracture Segmentation Challenge training dataset, containing 100 CT scans with pelvic fractures and their segmentation labels.
- 数据集链接 | Dataset Link: [@Zenodo](https://doi.org/10.5281/zenodo.10927452)

#### X光图像数据集 | X-ray Image Dataset
来自 PENGWIN Task 2 骨盆骨折X光图像分割挑战赛的训练数据集，包含由CT数据通过DeepDRR生成的50,000张合成X光图像及其分割标注。
From PENGWIN Task 2 Pelvic Fragment Segmentation Challenge training dataset, containing 50,000 synthetic X-ray images and segmentations generated from CT data using DeepDRR.
- 数据集链接 | Dataset Link: [@Zenodo](https://doi.org/10.5281/zenodo.10913195)

### 注意事项
1. 建议使用独立的虚拟环境来避免包冲突
2. 确保系统有足够的磁盘空间（至少 5GB）
3. 需要稳定的网络连接来下载包
4. 某些功能可能需要较高的系统配置，特别是在处理大型 3D 图像时

### 技术支持
如果遇到安装问题，可以：
1. 查看项目 GitHub Issues
2. 在 Stack Overflow 上搜索相关问题