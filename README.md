# Pelvis Fracture Image Segmentation

[项目计划](Project_plan.md)

 <img src="image/logo.png" width="300">

## 项目结构

```
pelvis_seg/
├── system/ # 系统模块
│   ├── main.py # 主程序
│   ├── main_window.py # 主窗口
│   ├── ct_viewer.py # CT图像处理
│   ├── xray_viewer.py # X光图像处理
│   ├── patient_manager.py # 患者管理
│   ├── ui # 项目对应ui
├── ct_seg/ # CT图像分割
│   ├── ...
├── xray_seg/ # X光图像分割
│   ├── ...
```

## Medical Image Viewer Installation Guide

### 环境要求
- Python 3.10 (推荐)
- Windows/Linux/MacOS

### 配置基础环境

#### 方法一：使用 conda（推荐）

1. 安装 [Anaconda](https://www.anaconda.com/download) 或 Miniconda

2. 创建并激活新环境

打开Anaconda Prompt，输入以下命令创建环境并激活

```bash
conda create -n pelvis_seg python=3.10
conda activate pelvis_seg
```

3.下载所需要的安装包

```bash
conda install pyqt
conda install vtk=9.2.2
conda install simpleitk
conda install numpy
```
#### 方法二：使用 pip

如果你更倾向于使用 pip，可以按以下步骤安装：

```bash
pip install PyQt5
pip install vtk==9.2.2
pip install SimpleITK
pip install numpy
```

### 验证安装

创建一个 Python 文件来验证安装是否成功：

```python
import sys
import vtk
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import QApplication

# 打印版本信息
print(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")
print(f"SimpleITK version: {sitk.Version.VersionString()}")
print(f"NumPy version: {np.__version__}")
print(f"PyQt version: {QApplication.qt_version()}")
```

### 常见问题

1. VTK 安装失败
   - 尝试使用 conda 安装而不是 pip
   - 确保系统已安装必要的编译工具

2. PyQt5 导入错误
   - 确保安装了所有必要的 PyQt5 子模块
   - 在 macOS 上可能需要额外设置：`export QT_MAC_WANTS_LAYER=1`

3. 版本兼容性问题
   - 建议严格按照指定版本安装
   - 如果遇到兼容性问题，可以尝试使用 conda 环境

### 镜像源设置（可选）

如果下载速度较慢，可以使用国内镜像源：

#### pip 镜像
```bash
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### conda 镜像
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### 支持的文件格式
- DICOM (.dcm)
- NIfTI (.nii, .nii.gz)
- NRRD (.nrrd)
- MetaImage (.mha, .mhd)

### 注意事项
1. 建议使用独立的虚拟环境来避免包冲突
2. 确保系统有足够的磁盘空间（至少 5GB）
3. 需要稳定的网络连接来下载包
4. 某些功能可能需要较高的系统配置，特别是在处理大型 3D 图像时

### 技术支持
如果遇到安装问题，可以：
1. 查看项目 GitHub Issues
2. 在 Stack Overflow 上搜索相关问题
