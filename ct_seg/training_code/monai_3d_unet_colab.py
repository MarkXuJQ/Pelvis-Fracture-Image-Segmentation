!pip install -q gputil

import os
import sys
import torch
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, CropForegroundd, RandCropByPosNegLabeld, 
    RandAffined, ToTensord, RandGaussianNoised, RandGaussianSmoothd, 
    RandAdjustContrastd, Lambdad, Resized, RandFlipd, RandRotate90d,
    RandSpatialCropD, SpatialPadD, AsDiscreted, ScaleIntensityRanged,
    SpatialPadd, RandSpatialCropd
)
import numpy as np
import time
import psutil
import monai
from monai.networks.nets import UNet, UNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import nibabel as nib
import random
import torch.nn as nn
from GPUtil import showUtilization
import glob

# 打印CUDA状态
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 验证GPU功能
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU计算测试成功: 结果shape={z.shape}")
else:
    print("CUDA不可用")
    
# 检查nvidia-smi输出
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("\nnvidia-smi输出:")
    print(result.stdout)
except Exception as e:
    print(f"无法运行nvidia-smi: {e}")

# 安装必要的依赖
import subprocess
print("安装必要的依赖...")
subprocess.run(["pip", "install", "-q", "SimpleITK"], check=True)
subprocess.run(["pip", "install", "-q", "itk"], check=True)

# 验证依赖安装
try:
    import SimpleITK as sitk
    print(f"SimpleITK版本: {sitk.Version_MajorVersion()}.{sitk.Version_MinorVersion()}")
    
    import itk
    print(f"ITK版本: {itk.Version.GetITKVersion()}")
except ImportError as e:
    print(f"错误: {e}，请确保安装正确")

# 使用ITKReader
from monai.data.image_reader import ITKReader

# CUDA辅助函数
def force_cuda_init():
    """强制初始化CUDA"""
    if not torch.cuda.is_available():
        print("尝试强制初始化CUDA...")
        # 尝试解除环境变量限制
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"发现CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，尝试清除")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
        # 尝试强制加载CUDA库
        try:
            torch.zeros(1).cuda()
            return torch.cuda.is_available()
        except:
            print("强制初始化CUDA失败")
            return False
    return True

def is_cuda_available():
    """检查CUDA是否可用，包括多种检测方法"""
    # 基础检查
    basic_available = torch.cuda.is_available()
    if basic_available:
        return True
        
    # 如果基础检查失败，尝试强制初始化
    return force_cuda_init()

def reset_cuda():
    """尝试重置CUDA状态"""
    try:
        # 完全释放并重置CUDA
        if is_cuda_available():
            print("重置CUDA设备...")
            
            # 先清除缓存
            torch.cuda.empty_cache()
            
            # 同步所有CUDA流
            torch.cuda.synchronize()
            
            # 记录当前内存状态
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            print(f"重置后GPU内存: 已分配={allocated/1e9:.2f}GB, 保留={reserved/1e9:.2f}GB")
            
            return True
    except Exception as e:
        print(f"重置CUDA失败: {e}")
        return False

# 将种子初始化分离到独立函数
def safe_init_random(seed):
    """安全地初始化随机种子"""
    try:
        # 先设置CPU随机种子
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 然后尝试设置CUDA种子
        if is_cuda_available():
            try:
                # 尝试单独设置设备0的种子，避免设备间通信
                torch.cuda.manual_seed(seed)
                print("成功设置CUDA随机种子")
                return True
            except Exception as e:
                print(f"设置CUDA种子失败: {e}")
                return False
        return True
    except Exception as e:
        print(f"初始化随机种子失败: {e}")
        return False

# 打印系统信息
def print_system_info():
    print("\n======== 系统信息 ========")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MONAI版本: {monai.__version__}")
    print(f"可用CPU核心: {psutil.cpu_count(logical=True)}")
    print(f"可用内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # 安全地获取CUDA信息
    try:
        # 显示NVIDIA-SMI输出以验证系统GPU状态
        try:
            print("\n--- NVIDIA-SMI 输出 ---")
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            print(result.stdout)
        except Exception as e:
            print(f"运行nvidia-smi失败: {e}")
            
        if is_cuda_available():
            print(f"CUDA版本: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU: {torch.cuda.get_device_name(i)}")
                print(f"GPU显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    except Exception as e:
        print(f"无法获取GPU信息: {e}")

def calculate_num_classes(label_dir):
    """自动从标签文件中计算类别数"""
    max_label = 0
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]
    
    for label_file in tqdm(label_files, desc="分析标签文件"):
        file_path = os.path.join(label_dir, label_file)
        try:
            label_data = nib.load(file_path).get_fdata()
            current_max = int(np.max(label_data))
            if current_max > max_label:
                max_label = current_max
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
            raise
    
    # 处理不同情况
    if max_label == 0:
        num_classes = 1  # 只有背景
    else:
        num_classes = max_label + 1  # 假设标签从0开始连续
    
    print(f"\n自动检测结果:")
    print(f"最大标签值: {max_label}")
    print(f"计算类别数: {num_classes} (包含背景)")
    return num_classes

# 检查数据集
def check_dataset(data_dir):
    # 确保路径正确
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    # 添加路径存在性检查
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    
    # 获取完整文件路径
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
    
    # 添加文件可访问性检查
    for f in image_files[:3]:  # 检查前3个文件
        if not os.access(f, os.R_OK):
            raise PermissionError(f"无法读取图像文件: {f}")
    for f in label_files[:3]:
        if not os.access(f, os.R_OK):
            raise PermissionError(f"无法读取标签文件: {f}")
    
    # 打印验证通过的示例路径
    print("\n验证通过的示例路径:")
    print(f"图像: {image_files[0]}")
    print(f"标签: {label_files[0]}")
    
    # 自动计算类别数
    a100_optimized_config['num_classes'] = calculate_num_classes(label_dir)
    
    # 验证标签连续性
    all_labels = set()
    sample_files = random.sample(label_files, min(5, len(label_files)))
    for f in sample_files:
        label_data = nib.load(f).get_fdata()
        all_labels.update(np.unique(label_data).astype(int))
    
    expected_labels = set(range(a100_optimized_config['num_classes']))
    missing = expected_labels - all_labels
    if missing:
        print(f"警告: 缺少以下标签值: {missing}，可能影响训练效果")

# 其他代码保持不变 (CT3DDataModule, CT3DSegmentationModel等)
class CT3DDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.patch_size = config.get('patch_size', (128, 128, 128))
        self.num_workers = config.get('num_workers', 4)
        self.val_ratio = config.get('val_ratio', 0.2)
        self.cache_rate = config.get('cache_rate', 0.0)
        self.spatial_size = config.get('spatial_size', None)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # 查找所有图像和标签文件
        images_dir = os.path.join(self.data_dir, 'images')
        labels_dir = os.path.join(self.data_dir, 'labels')
        
        print(f"读取图像目录: {images_dir}")
        print(f"读取标签目录: {labels_dir}")
        
        # 查找所有.nii.gz文件
        image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
        
        # 验证匹配的文件对
        image_names = [os.path.basename(f) for f in image_files]
        label_names = [os.path.basename(f) for f in label_files]
        
        # 查找匹配的文件对
        matching_pairs = []
        for i, img_name in enumerate(image_names):
            if img_name in label_names:
                img_path = image_files[i]
                label_path = label_files[label_names.index(img_name)]
                matching_pairs.append((img_path, label_path))
        
        print(f"找到{len(matching_pairs)}对匹配的图像和标签文件")
        
        if len(matching_pairs) > 0:
            print("示例匹配对:")
            for i in range(min(3, len(matching_pairs))):
                img_path, label_path = matching_pairs[i]
                print(f"  图像: {os.path.basename(img_path)}")
                print(f"  标签: {os.path.basename(label_path)}")
        
        # 准备数据字典
        data_dicts = []
        for img_path, label_path in matching_pairs:
            data_dicts.append({
                "image": img_path,
                "label": label_path
            })
        
        print(f"找到{len(data_dicts)}对.nii.gz格式的匹配图像-标签文件")
        
        # 随机分割数据集
        n_val = int(len(data_dicts) * self.val_ratio)
        n_train = len(data_dicts) - n_val
        
        import random
        random.seed(42)  # 确保可重复性
        random.shuffle(data_dicts)
        
        self.train_files = data_dicts[:n_train]
        self.val_files = data_dicts[n_train:]
        
        print(f"随机分割数据集: 训练集 {len(self.train_files)}个样本, 验证集 {len(self.val_files)}个样本")
        
        # 定义数据转换
        self.train_transforms = self._get_transforms(train=True)
        self.val_transforms = self._get_transforms(train=False)
        
        # 创建数据集
        self.train_ds = CacheDataset(
            data=self.train_files, 
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            progress=True
        )
        
        self.val_ds = CacheDataset(
            data=self.val_files, 
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            progress=True
        )

    def _get_transforms(self, train=True):
        # 确保读取器可以处理.nii.gz文件
        reader = ITKReader()
        
        # 默认转换管道
        if self.spatial_size is None:
            # 原始空间大小的转换
            if train:
                return Compose([
                    LoadImaged(keys=["image", "label"], reader=reader, image_only=True),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                    ScaleIntensityd(keys=["image"]),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    
                    # 先填充到至少目标尺寸
                    SpatialPadd(
                        keys=['image', 'label'],
                        spatial_size=self.patch_size,
                        mode='constant'
                    ),
                    
                    # 再随机裁剪到精确尺寸
                    RandSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.patch_size,
                        random_center=True,
                        random_size=False
                    ),
                    
                    # 数据增强
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
                    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
                    RandGaussianSmoothd(
                        keys=["image"],
                        prob=0.15,
                        sigma_x=(0.5, 1.5),
                        sigma_y=(0.5, 1.5),
                        sigma_z=(0.5, 1.5),
                    ),
                    RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),
                    ToTensord(keys=["image", "label"]),
                    Lambdad(
                        keys=['label'],
                        func=lambda x: torch.where(x > 24, 0, x)  # 处理异常标签值
                    ),
                    Lambdad(
                        keys=['label'],
                        func=lambda x: torch.nn.functional.one_hot(
                            x.squeeze(1).long(),  # 关键：去除通道维度 [B,1,H,W,D] → [B,H,W,D]
                            num_classes=self.num_classes
                        ).permute(0, 4, 1, 2, 3)  # [B,H,W,D,C] → [B,C,H,W,D]
                    ),
                    EnsureChannelFirstd(keys=['label'], channel_dim=1),  # 确认通道位置
                    Resized(
                        keys=['image', 'label'],
                        spatial_size=(64, 64, 64),  # 下采样到64^3
                        mode=('trilinear', 'nearest')
                    )
                ])
            else:
                return Compose([
                    LoadImaged(keys=["image", "label"], reader=reader, image_only=True),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                    ScaleIntensityd(keys=["image"]),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    ToTensord(keys=["image", "label"]),
                ])
        else:
            # 使用固定空间大小的转换
            if train:
                return Compose([
                    LoadImaged(keys=["image", "label"], reader=reader, image_only=True),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                    ScaleIntensityd(keys=["image"]),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    Resized(keys=["image", "label"], spatial_size=self.spatial_size, mode=("trilinear", "nearest")),
                    
                    # 先填充到至少目标尺寸
                    SpatialPadd(
                        keys=['image', 'label'],
                        spatial_size=self.patch_size,
                        mode='constant'
                    ),
                    
                    # 再随机裁剪到精确尺寸
                    RandSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.patch_size,
                        random_center=True,
                        random_size=False
                    ),
                    
                    # 数据增强
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
                    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
                    RandGaussianSmoothd(
                        keys=["image"],
                        prob=0.15,
                        sigma_x=(0.5, 1.5),
                        sigma_y=(0.5, 1.5),
                        sigma_z=(0.5, 1.5),
                    ),
                    RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),
                    ToTensord(keys=["image", "label"]),
                    Lambdad(
                        keys=['label'],
                        func=lambda x: torch.where(x > 24, 0, x)  # 处理异常标签值
                    ),
                    Lambdad(
                        keys=['label'],
                        func=lambda x: torch.nn.functional.one_hot(
                            x.squeeze(1).long(),  # 关键：去除通道维度 [B,1,H,W,D] → [B,H,W,D]
                            num_classes=self.num_classes
                        ).permute(0, 4, 1, 2, 3)  # [B,H,W,D,C] → [B,C,H,W,D]
                    ),
                    EnsureChannelFirstd(keys=['label'], channel_dim=1),  # 确认通道位置
                    Resized(
                        keys=['image', 'label'],
                        spatial_size=(64, 64, 64),  # 下采样到64^3
                        mode=('trilinear', 'nearest')
                    )
                ])
            else:
                return Compose([
                    LoadImaged(keys=["image", "label"], reader=reader, image_only=True),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                    ScaleIntensityd(keys=["image"]),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    Resized(keys=["image", "label"], spatial_size=self.spatial_size, mode=("trilinear", "nearest")),
                    ToTensord(keys=["image", "label"]),
                ])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

class BTCVModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 自动检测设备
        self.device = config['device']
        print(f"模型将运行在: {self.device}")
        
        # 版本兼容的GradScaler
        if hasattr(torch.cuda.amp, 'GradScaler'):
            self.scaler = torch.cuda.amp.GradScaler()  # 新版本
        else:
            self.scaler = torch.amp.GradScaler(device_type='cuda')  # 旧版本兼容
            
        # 简化UNet结构
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=8,
            channels=(16, 32, 64),  # 减少层数
            strides=(2, 2),
            num_res_units=1
        ).to(self.device)
        self.loss_function = DiceCELoss(
            to_onehot_y=False,  # 标签已经是one-hot
            softmax=True,       # 模型输出需要softmax
            squared_pred=True
        )
        self.optimizer = AdamW(self.parameters(), lr=config['learning_rate'])

    def forward(self, x):
        return self.model(x)

def get_data_loaders(config):
    """创建训练和验证数据加载器 - 内存优化版本"""
    image_dir = os.path.join(config['data_dir'], 'images')
    label_dir = os.path.join(config['data_dir'], 'labels')
    
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                        if f.endswith('.nii.gz')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                        if f.endswith('.nii.gz')])
    
    data_dicts = [{'image': img, 'label': lbl} 
                for img, lbl in zip(image_files, label_files)]
    
    # 创建数据转换
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label'], reader=ITKReader(io_args={'mode': 'r'})),
        EnsureChannelFirstd(keys=['image', 'label']),
        # 提前降采样以减少内存使用
        Spacingd(
            keys=['image', 'label'],
            pixdim=(2.0, 2.0, 2.0),  # 增加体素大小来减少数据量
            mode=('bilinear', 'nearest')
        ),
        # 使用标准化替代one-hot编码来减少内存
        Lambdad(
            keys=['label'],
            func=lambda x: x.squeeze(1).long()  # 仅压缩通道，不进行one-hot
        ),
        # 减小crop size以减少内存使用
        RandSpatialCropd(
            keys=['image', 'label'], 
            roi_size=(64, 64, 64),  # 减小ROI大小
            random_size=False
        ),
        ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        ToTensord(keys=['image', 'label'])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(
            keys=['image', 'label'],
            pixdim=(2.0, 2.0, 2.0),
            mode=('bilinear', 'nearest')
        ),
        Lambdad(
            keys=['label'],
            func=lambda x: x.squeeze(1).long()
        ),
        ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        ToTensord(keys=['image', 'label'])
    ])
    
    # 修改1: 使用PersistentDataset而不是CacheDataset
    cache_dir = os.path.join(config['log_dir'], 'dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    from monai.data import PersistentDataset
    
    train_ds = PersistentDataset(
        data=data_dicts[:int(len(data_dicts) * 0.8)],
        transform=train_transforms,
        cache_dir=cache_dir  # 缓存到磁盘而不是内存
    )
    
    val_ds = PersistentDataset(
        data=data_dicts[int(len(data_dicts) * 0.8):],
        transform=val_transforms,
        cache_dir=cache_dir
    )
    
    # 修改2: 减小batch_size和workers数量
    reduced_batch_size = max(1, config['batch_size'] // 2)  # 减半batch_size
    reduced_workers = max(0, config['num_workers'] // 2)  # 减半worker数
    
    train_loader = DataLoader(
        train_ds,
        batch_size=reduced_batch_size,  # 减小batch_size 
        shuffle=True,
        num_workers=reduced_workers,    # 减少worker数量
        pin_memory=False                # 关闭pin_memory降低内存压力
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=reduced_batch_size,
        shuffle=False,
        num_workers=reduced_workers,
        pin_memory=False
    )
    
    # 修改3: 添加内存清理函数
    def cleanup_memory():
        """手动清理内存"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    cleanup_memory()  # 立即执行清理
    
    return train_loader, val_loader, cleanup_memory  # 返回清理函数供后续使用

def train_model(config):
    """内存优化版本的训练函数"""
    # 获取数据加载器和清理函数
    train_loader, val_loader, cleanup_memory = get_data_loaders(config)
    
    # 修改模型以支持非one-hot编码标签
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=config['num_classes'], 
        channels=(16, 32, 64, 128),     # 减少通道数
        strides=(2, 2, 2),              # 减少层数
        dropout=0.2                      # 添加dropout
    ).to(config['device'])
    
    # 修改损失函数以支持非one-hot编码标签 
    loss_function = DiceCELoss(
        to_onehot_y=True,               # 现在需要转换为one-hot
        softmax=True,
        squared_pred=True,
        include_background=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    
    # 使用混合精度训练减少内存使用
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        
        # 每个epoch结束后清理内存
        cleanup_memory()
        
        # 在训练循环中使用混合精度
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data['image'].to(config['device']), batch_data['label'].to(config['device'])
            
            optimizer.zero_grad()
            
            # 使用混合精度
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # 使用梯度缩放器
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # 每4个批次清理一次内存
            if batch_idx % 4 == 0:
                cleanup_memory()
                
            # 定期打印内存使用情况
            if batch_idx % 10 == 0:
                print(f"GPU内存: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB")
                
        print(f"Epoch {epoch+1}/{config['num_epochs']}, 平均损失: {epoch_loss/len(train_loader):.4f}")
        
        # 验证循环
        model.eval()
        with torch.no_grad():
            # ...验证代码...
            pass
            
        # 强制清理内存
        cleanup_memory()
        
    return model

def monitor_resources():
    """统一的资源监控函数"""
    try:
        from GPUtil import showUtilization
        showUtilization()
    except ImportError:
        print("GPU监控不可用，使用基础方案：")
        if torch.cuda.is_available():
            print(f"已分配内存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"保留内存: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    # 添加系统内存监控
    import psutil
    print(f"系统内存使用: {psutil.virtual_memory().percent}%")

# A100优化配置 - 使用NIfTI文件
a100_optimized_config = {
    'data_dir': '/content/drive/MyDrive/ct_segmentation/3DU-net',  # 指向包含images和labels子目录的目录
    'log_dir': '/content/drive/MyDrive/ct_segmentation/3DU-net/logs',
    'checkpoint_dir': '/content/drive/MyDrive/ct_segmentation/3DU-net/checkpoints',
    'experiment_name': '3d_unet_a100_nifti',
    'batch_size': 2,
    'patch_size': (128, 128, 128),
    'num_workers': 4,
    'learning_rate': 3e-4,
    'original_labels': [0,1,2,11,12,21,22,23],  # 实际存在的标签
    'include_background': True,
    'max_epochs': 200,
    'gpus': 1,
    'precision': 16,
    'accumulate_grad_batches': 1,
    'seed': 42,
    'val_ratio': 0.2,
    'cache_rate': 0.2,
    'resume_checkpoint': None,
    'spatial_size': (128, 128, 128),  # 设置统一的空间大小
    'weight_decay': 1e-5,  # 新增权重衰减参数
    'ignored_labels': [3,4,5,6,7,8,9,10,15,16,17,18,19,20,24],
    'valid_labels': [0,1,2,3,4,5,6,7],         # 重新映射后的标签索引
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 新增设备配置
    'label_remapping': {
        0:0, 1:1, 2:2, 11:3, 12:4, 21:5, 22:6, 23:7
    },
    'persistent_workers': True,
    'loss_function': DiceCELoss(
        to_onehot_y=False,  # 标签已经是one-hot
        softmax=True,       # 模型输出需要softmax
        squared_pred=True,
        include_background=True
    ),
    'dice_metric': DiceMetric(include_background=True),
    'num_classes': 25,  # 根据自动检测结果设置
    'in_channels': 1,
    'train_ratio': 0.8,  # 80%训练，20%验证
    'num_epochs': 200  # 新增num_epochs参数
}

# 强制训练前进行CUDA检测和初始化
if __name__ == "__main__":
    # 首先尝试强制初始化CUDA
    cuda_initialized = force_cuda_init()
    
    # 尝试安全地检查CUDA可用性
    gpu_available = is_cuda_available()
    
    # 如果CUDA不可用，尝试修复
    if not gpu_available:
        print("\n❌ CUDA检测失败，尝试解决方案...")
        print("1. 检查CUDA安装状态...")
        
        # 检查NVCC版本
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
            print(result.stdout)
        except:
            print("未找到nvcc")
            
        # 检查GPU驱动
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            print(result.stdout)
        except:
            print("未找到nvidia-smi")
            
        # 提示用户进行手动操作
        print("\n🔧 建议解决方案:")
        print("- 在Colab菜单中选择: Runtime > Factory reset runtime")
        print("- 运行时重新启动后，在设置中确认GPU已启用: Runtime > Change runtime type")
        print("- 重新运行此代码")
        print("\n尝试继续使用CPU训练，但效率会非常低...")
    
    if gpu_available:
        print("✅ CUDA可用，继续初始化...")
        try:
            # 尝试基本的CUDA设备检查
            device_count = torch.cuda.device_count()
            print(f"检测到 {device_count} 个CUDA设备")
            
            for i in range(device_count):
                print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
                
            # 尝试重置所有CUDA设备
            reset_cuda()
        except Exception as e:
            print(f"CUDA初始化检查失败: {e}")
            gpu_available = False
    
    # 打印系统信息
    print_system_info()
    
    # 检查数据集
    check_dataset('/content/drive/MyDrive/ct_segmentation/3DU-net')
    
    print("\n======== 开始3D U-Net训练 ========")
    start_time = time.time()
    
    try:
        model = train_model(a100_optimized_config)
        total_time = time.time() - start_time
        print(f"\n总训练时间: {total_time//3600}小时 {(total_time%3600)//60}分钟 {total_time%60:.1f}秒")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        import traceback
        log_path = os.path.join(a100_optimized_config['log_dir'], 'error_log.txt')
        print(f"\n训练失败: {e}")
        print(f"错误日志已保存到: {log_path}")
        
        # 确保日志目录存在
        os.makedirs(a100_optimized_config['log_dir'], exist_ok=True)
        
        # 写入错误信息
        with open(log_path, 'w') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        print("遇到错误，但允许继续执行")

# 修改auto_detect_classes函数参数
def auto_detect_classes(label_files, ignored_labels):
    all_labels = set()
    for f in tqdm(label_files, desc="分析标签文件"):
        img = nib.load(f).get_fdata()
        all_labels.update(np.unique(img).astype(int))
    return len(all_labels - set(ignored_labels))

# 数据路径处理
def prepare_datasets(config):
    # 获取所有配对文件
    images = sorted(glob.glob(os.path.join(config['data_dir'], 'images', '*.nii.gz')))
    labels = sorted(glob.glob(os.path.join(config['data_dir'], 'labels', '*.nii.gz')))
    
    # 验证文件配对
    assert len(images) == len(labels), "图像和标签数量不匹配"
    print(f"找到{len(images)}个有效数据对")
    
    # 创建数据字典列表
    data_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(images, labels)]
    
    # 随机分割训练集和验证集
    random.seed(42)
    random.shuffle(data_dicts)
    split = int(len(data_dicts) * config['train_ratio'])
    train_files = data_dicts[:split]
    val_files = data_dicts[split:]
    
    print(f"\n数据集分割结果:")
    print(f"训练集: {len(train_files)}个样本")
    print(f"验证集: {len(val_files)}个样本")
    print(f"示例训练样本: {train_files[0]['image']}")
    
    return train_files, val_files

# 调用数据准备
train_files, val_files = prepare_datasets(a100_optimized_config)

# 更新自动检测函数调用
num_classes = auto_detect_classes(
    [d['label'] for d in train_files],  # 仅使用训练集统计
    ignored_labels=a100_optimized_config.get('ignored_labels', [])
)

# 更新配置
a100_optimized_config['num_classes'] = num_classes 

# 添加内存监控
monitor_resources()  # 在每个epoch开始前调用 

# 极简化预处理流程（修复括号匹配）
train_transforms = Compose([
    LoadImaged(keys=['image', 'label'], reader=ITKReader()),
    EnsureChannelFirstd(keys=['image', 'label']),
    Lambdad(keys=['label'], func=lambda x: torch.from_numpy(np.vectorize(config['label_remapping'].get)(x.squeeze(0).numpy(), 0)).long()),
    AsDiscreted(keys=['label'], to_onehot=num_classes, num_classes=num_classes),
    RandSpatialCropd(keys=['image', 'label'], roi_size=(96, 96, 96), random_size=False),
    ToTensord(keys=['image', 'label'])
])

# 修改2：更新模型输出通道
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,  # 使用自动检测的类别数
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(config['device'])

# 修改3：调整损失函数配置 
a100_optimized_config['loss_function'] = DiceCELoss(
    to_onehot_y=False,  # 标签已编码
    softmax=True,
    squared_pred=True,
    include_background=True
)

# 验证步骤
sample = train_transforms(train_files[0])
print("\n预处理后维度验证:")
print(f"图像维度: {sample['image'].shape}")
print(f"标签维度: {sample['label'].shape}") 

# 逐步测试流程（避免复杂嵌套）
# 1. 加载测试样本
test_sample = train_files[0]
print(f"测试样本: {test_sample['image']}")

# 2. 测试基础转换  
basic_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    ToTensord(keys=['image', 'label'])
])

# 3. 应用转换测试
processed = basic_transforms(test_sample)
print(f"基础转换后: 图像={processed['image'].shape}, 标签={processed['label'].shape}")

# 4. 创建模型测试
test_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64),
    strides=(2, 2)
).to(config['device'])

print(f"模型创建成功! 参数数量: {sum(p.numel() for p in test_model.parameters())}") 