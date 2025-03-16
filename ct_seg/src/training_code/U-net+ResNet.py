# 检查并安装必要的库
import sys
import subprocess

# 安装必要的包
def install_packages():
    print("正在检查并安装必要的包...")
    required_packages = ['pytorch-lightning']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  - {package} 已安装")
        except ImportError:
            print(f"  - 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  - {package} 安装完成")

# 在导入之前安装
install_packages()

# 简化后的DeepLabV3实现用于CT图像分割
# 通过PyTorch Lightning简化训练代码

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 数据目录设置
data_base_dir = '/content/drive/MyDrive/ct_segmentation/data'
train_images_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_images')
train_labels_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_labels')
output_dir = '/content/drive/MyDrive/ct_segmentation/DeeplabV3'

# 若输出目录不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 配置对象
config = {
    'seed': 42,
    'image_size': (256, 256),  # 增大尺寸，更好利用GPU
    'batch_size': 32,  # 显著增大批量大小
    'val_ratio': 0.2,
    'max_epochs': 20,
    'learning_rate': 1e-3,  # 略微增大学习率
    'num_classes': 24,
    'use_mobilenet': True,  # 使用更轻量的MobileNet骨干
    'precision': 16,
}

# 修改设置随机种子函数，允许非确定性操作
def set_seed(seed=42):
    """设置随机种子确保可复现性，但允许某些非确定性操作"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 允许非确定性操作，但会发出警告而不是错误
        torch.use_deterministic_algorithms(False)
        # 使用以下设置可以获得较好的性能
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# CT切片数据集类
class CTSliceDataset(Dataset):
    """3D CT切片数据集"""
    def __init__(self, image_files, label_files, resize_to=None):
        self.image_files = image_files
        self.label_files = label_files
        self.resize_to = resize_to
        self.slice_indices = self._prepare_slices()
    
    def _prepare_slices(self):
        """准备有效切片的索引，但进行下采样以减少数据量"""
        indices = []
        print("准备切片索引...")
        for i, (img_file, lbl_file) in enumerate(tqdm(zip(self.image_files, self.label_files))):
            try:
                # 读取数据
                img_sitk = sitk.ReadImage(img_file)
                lbl_sitk = sitk.ReadImage(lbl_file)
                
                img_array = sitk.GetArrayFromImage(img_sitk)
                lbl_array = sitk.GetArrayFromImage(lbl_sitk)
                
                # 只保留每5个有标签的切片(下采样)
                for slice_idx in range(0, img_array.shape[0], 5):  # 每5个切片取1个
                    if np.max(lbl_array[slice_idx]) > 0:
                        indices.append({'vol_idx': i, 'slice_idx': slice_idx})
            except Exception as e:
                print(f"准备切片 {img_file} 时出错: {e}")
        
        print(f"总共准备了 {len(indices)} 个2D切片")
        return indices
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        try:
            # 获取切片信息
            slice_info = self.slice_indices[idx]
            vol_idx = slice_info['vol_idx']
            slice_idx = slice_info['slice_idx']
            
            # 读取图像和标签
            img_sitk = sitk.ReadImage(self.image_files[vol_idx])
            lbl_sitk = sitk.ReadImage(self.label_files[vol_idx])
            
            img_array = sitk.GetArrayFromImage(img_sitk)
            lbl_array = sitk.GetArrayFromImage(lbl_sitk)
            
            # 获取单个切片
            image = img_array[slice_idx].astype(np.float32)
            label = lbl_array[slice_idx].astype(np.int64)
            
            # 标准化图像
            if np.max(image) > 0:
                image = image / np.max(image)
            
            # 转换为张量
            image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
            label = torch.from_numpy(label)  # [H, W]
            
            # 调整大小
            if self.resize_to:
                image = F.interpolate(
                    image.unsqueeze(0), 
                    size=self.resize_to, 
                    mode='bilinear'
                ).squeeze(0)
                
                label = F.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(), 
                    size=self.resize_to, 
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            # 确保标签在有效范围内
            if torch.max(label) >= config['num_classes']:
                label = torch.clamp(label, 0, config['num_classes']-1)
            
            return image, label
        except Exception as e:
            print(f"加载样本 {idx} 出错: {e}")
            # 返回零填充的样本
            dummy_shape = self.resize_to if self.resize_to else (128, 128)
            return torch.zeros((1, *dummy_shape)), torch.zeros(dummy_shape, dtype=torch.long)

# 添加用于U-Net的组件
class DoubleConv(nn.Module):
    """U-Net中的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入是BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 修改为UNet+ResNet模型
class UNetResNet(nn.Module):
    """结合ResNet骨干网络的U-Net模型"""
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 加载预训练的ResNet50作为编码器
        resnet = torchvision.models.resnet50(pretrained=True)
        
        # 修改第一层以接受单通道输入
        self.firstconv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 复制ResNet权重的平均值到新的单通道层
        with torch.no_grad():
            self.firstconv.weight = nn.Parameter(
                resnet.conv1.weight.mean(dim=1, keepdim=True)
            )
        
        # ResNet编码器部分
        self.encoder1 = nn.Sequential(self.firstconv, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2  # 输出 512 通道
        self.encoder4 = resnet.layer3  # 输出 1024 通道
        self.encoder5 = resnet.layer4  # 输出 2048 通道
        
        factor = 2 if bilinear else 1
        
        # 解码器部分
        self.up1 = Up(2048 + 1024, 1024 // factor, bilinear)
        self.up2 = Up(1024 // factor + 512, 512 // factor, bilinear)
        self.up3 = Up(512 // factor + 256, 256 // factor, bilinear)
        self.up4 = Up(256 // factor + 64, 128, bilinear)
        self.up5 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器路径
        x1 = self.encoder1(x)           # 64 通道
        x2 = self.encoder2(x1)          # 256 通道
        x3 = self.encoder3(x2)          # 512 通道
        x4 = self.encoder4(x3)          # 1024 通道
        x5 = self.encoder5(x4)          # 2048 通道
        
        # 解码器路径 + 跳跃连接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x)
        logits = self.outc(x)
        
        return logits

# 修改Lightning模块类
class CTSegmentationModule(pl.LightningModule):
    """CT分割Lightning模块，使用UNet+ResNet"""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # 使用自定义的UNet+ResNet模型
        self.model = UNetResNet(
            n_channels=1,
            n_classes=config['num_classes'],
            bilinear=True
        )
        
        # 创建交叉熵损失函数 
        self.criterion = nn.CrossEntropyLoss()
        
        # 最大类别数
        self.max_class = config['num_classes'] - 1
        
        print(f"U-Net+ResNet模型已初始化，类别数: {config['num_classes']}")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    # 训练步骤不变
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # 记录训练损失
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    # 验证步骤不变
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # 记录验证损失
        self.log('val_loss', loss, prog_bar=True)
        
        # 计算Dice系数
        preds = torch.argmax(y_hat, dim=1)
        dice = self._dice_coefficient(preds, y)
        self.log('val_dice', dice, prog_bar=True)
        
        return loss
    
    def _dice_coefficient(self, pred, target):
        smooth = 1.0
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# 数据模块
class CTDataModule(pl.LightningDataModule):
    """CT数据模块"""
    def __init__(self, config):
        super().__init__()
        self.config = config.copy()  # 创建配置的副本以便修改
        self.train_dataset = None
        self.val_dataset = None
        self.setup_done = False  # 添加标志以防止重复执行
        
        # 明确指定完整路径并验证文件存在性
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        
        print(f"检查数据目录:")
        print(f"图像目录: {self.train_images_dir}")
        print(f"  - 存在: {os.path.exists(self.train_images_dir)}")
        
        # 直接查找.mha文件
        if os.path.exists(self.train_images_dir):
            img_files = glob.glob(os.path.join(self.train_images_dir, "*.mha"))
            print(f"  - 找到 {len(img_files)} 个.mha文件")
            if len(img_files) == 0:
                # 尝试不同的扩展名
                img_files = glob.glob(os.path.join(self.train_images_dir, "*.*"))
                print(f"  - 所有文件格式共有 {len(img_files)} 个文件")
                print(f"  - 文件示例: {img_files[:3] if img_files else '无'}")
                
        print(f"标签目录: {self.train_labels_dir}")
        print(f"  - 存在: {os.path.exists(self.train_labels_dir)}")
        if os.path.exists(self.train_labels_dir):
            lbl_files = glob.glob(os.path.join(self.train_labels_dir, "*.mha"))
            print(f"  - 找到 {len(lbl_files)} 个.mha文件")
            if len(lbl_files) == 0:
                # 尝试不同的扩展名
                lbl_files = glob.glob(os.path.join(self.train_labels_dir, "*.*"))
                print(f"  - 所有文件格式共有 {len(lbl_files)} 个文件")
                print(f"  - 文件示例: {lbl_files[:3] if lbl_files else '无'}")
                
        self.image_files = []
        self.label_files = []
    
    def setup(self, stage=None):
        """准备数据集，防止重复执行"""
        # 如果已经设置过，则跳过
        if self.setup_done:
            return
            
        # 将.mha扩展名放在首位
        image_extensions = ["*.mha", "*.nii.gz", "*.nii", "*.nrrd", "*.dcm"]
        label_extensions = ["*.mha", "*.nii.gz", "*.nii", "*.nrrd", "*.dcm"] 
        
        self.image_files = []
        for ext in image_extensions:
            found_files = sorted(glob.glob(os.path.join(self.train_images_dir, ext)))
            if found_files:
                print(f"使用扩展名 {ext} 找到 {len(found_files)} 个图像文件")
                print(f"图像文件示例: {found_files[0] if found_files else '无'}")
                self.image_files = found_files
                break
                
        self.label_files = []
        for ext in label_extensions:
            found_files = sorted(glob.glob(os.path.join(self.train_labels_dir, ext)))
            if found_files:
                print(f"使用扩展名 {ext} 找到 {len(found_files)} 个标签文件")
                print(f"标签文件示例: {found_files[0] if found_files else '无'}")
                self.label_files = found_files
                break
        
        if not self.image_files or not self.label_files:
            print("警告: 未找到数据文件!")
            print("尝试创建示例数据用于演示...")
            
            # 创建假数据集
            self.train_dataset = DummyCTDataset(num_samples=10, image_size=self.config['image_size'])
            self.val_dataset = DummyCTDataset(num_samples=5, image_size=self.config['image_size']) 
            
            print("已创建演示数据集")
            print(f"训练集大小: {len(self.train_dataset)}")
            print(f"验证集大小: {len(self.val_dataset)}")
            return
        
        # 确保图像和标签数量匹配
        if len(self.image_files) != len(self.label_files):
            print(f"警告: 图像文件 ({len(self.image_files)}) 和标签文件 ({len(self.label_files)}) 数量不匹配!")
            # 尝试排序并匹配文件名
            try:
                img_basenames = [os.path.basename(f) for f in self.image_files]
                lbl_basenames = [os.path.basename(f) for f in self.label_files]
                
                # 查找相同的基本文件名
                common_names = set(img_basenames).intersection(set(lbl_basenames))
                print(f"找到 {len(common_names)} 个具有相同文件名的图像和标签对")
                
                if common_names:
                    # 重建文件列表
                    matched_img_files = []
                    matched_lbl_files = []
                    
                    for name in common_names:
                        img_idx = img_basenames.index(name)
                        lbl_idx = lbl_basenames.index(name)
                        matched_img_files.append(self.image_files[img_idx])
                        matched_lbl_files.append(self.label_files[lbl_idx])
                    
                    self.image_files = matched_img_files
                    self.label_files = matched_lbl_files
                    print(f"成功匹配 {len(self.image_files)} 对图像和标签文件")
                else:
                    # 如果找不到匹配的文件名，则取最小值
                    min_count = min(len(self.image_files), len(self.label_files))
                    self.image_files = self.image_files[:min_count]
                    self.label_files = self.label_files[:min_count]
            except Exception as e:
                print(f"匹配文件时出错: {e}")
                min_count = min(len(self.image_files), len(self.label_files))
                self.image_files = self.image_files[:min_count]
                self.label_files = self.label_files[:min_count]
        
        # 继续原有的划分逻辑
        n_train = int((1 - self.config['val_ratio']) * len(self.image_files))
        
        train_img_files = self.image_files[:n_train]
        train_lbl_files = self.label_files[:n_train]
        val_img_files = self.image_files[n_train:]
        val_lbl_files = self.label_files[n_train:]
        
        print(f"划分数据集完成:\n  - 训练集: {len(train_img_files)} 样本\n  - 验证集: {len(val_img_files)} 样本")
        
        # 验证第一个图像和标签文件是否可读
        try:
            if train_img_files and train_lbl_files:
                first_img = sitk.ReadImage(train_img_files[0])
                first_lbl = sitk.ReadImage(train_lbl_files[0])
                img_array = sitk.GetArrayFromImage(first_img)
                lbl_array = sitk.GetArrayFromImage(first_lbl)
                print(f"示例图像尺寸: {img_array.shape}, 示例标签尺寸: {lbl_array.shape}")
                print(f"图像数据类型: {img_array.dtype}, 标签数据类型: {lbl_array.dtype}")
                print(f"图像值范围: [{img_array.min()}, {img_array.max()}], 标签值范围: [{lbl_array.min()}, {lbl_array.max()}]")
        except Exception as e:
            print(f"读取示例文件时出错: {e}")
            print("请确认文件格式正确，如果是特殊格式可能需要额外处理")
            
        # 创建数据集
        self.train_dataset = CTSliceDataset(
            train_img_files, 
            train_lbl_files, 
            resize_to=self.config['image_size']
        )
        
        self.val_dataset = CTSliceDataset(
            val_img_files, 
            val_lbl_files, 
            resize_to=self.config['image_size']
        )
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
        
        # 创建数据集后，检测实际类别数并更新配置
        if len(self.train_dataset) > 0:
            actual_num_classes = self._check_label_range()
            self.config['detected_num_classes'] = actual_num_classes
            print(f"检测到的类别数: {actual_num_classes}")
            
        # 标记设置已完成
        self.setup_done = True
    
    def _check_label_range(self):
        """检查实际标签类别并返回检测到的类别数"""
        print("检查实际标签类别...")
        unique_labels = set()
        max_label = 0
        
        # 遍历数据集样本获取所有唯一标签
        for i in range(min(100, len(self.train_dataset))):  # 仅检查前100个样本以节省时间
            _, labels = self.train_dataset[i]
            unique_vals = np.unique(labels.numpy())
            curr_max = np.max(unique_vals) if len(unique_vals) > 0 else 0
            
            if curr_max > max_label:
                print(f"  在样本 {i} 中发现新的最大类别ID: {curr_max}")
                max_label = curr_max
                
            unique_labels.update(unique_vals)
        
        print(f"数据集中发现的不同标签值: {sorted(list(unique_labels))}")
        print(f"最大标签值: {max_label}")
        
        # 返回类别数（最大标签值+1）
        return max_label + 1
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,  # 增加工作线程数
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # 保持工作线程活跃
            prefetch_factor=3  # 预取数据批次
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'] * 2,  # 验证时用更大批量
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3
        )

# 添加一个假数据集用于演示
class DummyCTDataset(Dataset):
    """生成随机CT数据用于演示"""
    def __init__(self, num_samples=10, image_size=(128, 128), num_classes=24):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 创建一个随机图像
        image = torch.rand(1, *self.image_size)
        
        # 创建一个随机分割掩码
        mask = torch.randint(0, self.num_classes, self.image_size)
        
        return image, mask

# 主函数
def main():
    """主函数"""
    try:
        # 设置随机种子
        set_seed(config['seed'])
        
        # 创建数据模块并手动运行setup
        print("=== 准备数据模块 ===")
        data_module = CTDataModule(config)
        data_module.setup()
        
        # 使用检测到的类别数更新配置 - 显示更清晰的信息
        if hasattr(data_module, 'config') and 'detected_num_classes' in data_module.config:
            detected_classes = data_module.config['detected_num_classes']
            config['num_classes'] = detected_classes
            print(f"=== 使用从数据集检测到的类别数: {detected_classes} ===")
        
        # 创建模型
        print("=== 初始化模型 ===")
        model = CTSegmentationModule(config)
        
        # 创建回调
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename='deeplabv3-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # 创建日志器
        logger = TensorBoardLogger(output_dir, name="logs")
        
        # 优化训练速度的训练器设置
        trainer = pl.Trainer(
            max_epochs=config['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=config['precision'],
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            log_every_n_steps=10,
            deterministic=False,  # 禁用确定性以提高速度
            # 重要性能优化
            gradient_clip_val=1.0,  # 避免梯度爆炸
            accumulate_grad_batches=4,  # 积累梯度以增加有效批量大小
            benchmark=True,  # 启用CUDNN基准测试
            inference_mode=True,  # 使用torch.inference_mode()加速
            num_sanity_val_steps=0,  # 跳过初始验证步骤
            enable_progress_bar=True,  # 显示进度条
            limit_train_batches=0.25  # 只使用25%的训练数据（加快调试）
        )
        
        # 训练模型
        trainer.fit(model, data_module)
        
        # 保存最终模型
        trainer.save_checkpoint(os.path.join(output_dir, "final_model.ckpt"))
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  