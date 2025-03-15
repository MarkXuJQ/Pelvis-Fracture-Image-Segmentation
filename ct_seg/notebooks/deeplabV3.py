from google.colab import drive
drive.mount('/content/drive')

# 设置工作目录
# Set working directory
import os

# 基础数据目录
# Base data directory  
data_base_dir = '/content/drive/MyDrive/ct_segmentation/data'

# 训练图像和标签目录
# Training images and labels directories
train_images_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_images')
train_labels_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_labels')

# 创建输出目录
# Create output directory
output_dir = '/content/drive/MyDrive/ct_segmentation/DeeplabV3'
os.makedirs(output_dir, exist_ok=True)

# 验证目录存在性
# Verify directories exist
print(f"训练图像目录: {train_images_dir}")
print(f"  - 存在: {os.path.exists(train_images_dir)}")
if os.path.exists(train_images_dir):
    image_files = [f for f in os.listdir(train_images_dir) 
                  if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))]
    print(f"  - 包含 {len(image_files)} 个图像文件")

print(f"训练标签目录: {train_labels_dir}")
print(f"  - 存在: {os.path.exists(train_labels_dir)}")
if os.path.exists(train_labels_dir):
    label_files = [f for f in os.listdir(train_labels_dir) 
                  if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))]
    print(f"  - 包含 {len(label_files)} 个标签文件")

print(f"输出目录: {output_dir}")
print(f"  - 存在: {os.path.exists(output_dir)}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 修复导入方式，使用auto以适应不同环境

class PelvicCTDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, patch_size=(128, 128, 128), 
                 transform=None, mode='train', stride=(64, 64, 64)):
        """
        3D CT扫描数据集
        
        参数:
            images_dir (str): CT扫描图像目录
            labels_dir (str, optional): 分割标注目录 (测试模式下可为None)
            patch_size (tuple): 训练时使用的3D patch大小
            transform (callable, optional): 数据增强变换
            mode (str): 'train', 'val', 或 'test'
            stride (tuple): 提取patch时的步长
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.stride = stride
        
        # 获取所有CT图像文件列表
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                  if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))])
        
        if labels_dir is not None and mode != 'test':
            self.label_files = sorted([f for f in os.listdir(labels_dir)
                                      if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))])
            assert len(self.image_files) == len(self.label_files), "Images and labels count mismatch"
        else:
            self.label_files = None
            
    def __len__(self):
        return len(self.image_files)
    
    def _load_volume(self, file_path):
        """加载体积数据"""
        if file_path.endswith(('.nii', '.nii.gz')):
            nii_img = nib.load(file_path)
            return np.array(nii_img.get_fdata())
        else:
            sitk_img = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(sitk_img)
    
    def _preprocess(self, image):
        """CT图像预处理"""
        # 裁剪HU值并归一化到0-1
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000
        return image
    
    def _extract_patches(self, image, label=None):
        """从体积数据中提取3D patches"""
        D, H, W = image.shape
        patches_img = []
        patches_label = []
        patch_locations = []  # 存储patch位置，用于测试时重建
        
        # 计算每个维度的步长
        d_steps = range(0, max(1, D - self.patch_size[0] + 1), self.stride[0])
        h_steps = range(0, max(1, H - self.patch_size[1] + 1), self.stride[1])
        w_steps = range(0, max(1, W - self.patch_size[2] + 1), self.stride[2])
        
        # 确保至少有一个patch
        if not d_steps:
            d_steps = [0]
        if not h_steps:
            h_steps = [0]
        if not w_steps:
            w_steps = [0]
            
        for d in d_steps:
            for h in h_steps:
                for w in w_steps:
                    # 计算patch结束位置
                    d_end = min(d + self.patch_size[0], D)
                    h_end = min(h + self.patch_size[1], H)
                    w_end = min(w + self.patch_size[2], W)
                    
                    # 提取patch
                    patch_img = image[d:d_end, h:h_end, w:w_end]
                    
                    # 如果patch尺寸小于目标尺寸，进行填充
                    if patch_img.shape != self.patch_size:
                        pad_d = self.patch_size[0] - patch_img.shape[0]
                        pad_h = self.patch_size[1] - patch_img.shape[1]
                        pad_w = self.patch_size[2] - patch_img.shape[2]
                        
                        patch_img = np.pad(patch_img, 
                                         ((0, pad_d), (0, pad_h), (0, pad_w)), 
                                         mode='constant')
                    
                    patches_img.append(patch_img)
                    patch_locations.append((d, h, w, d_end, h_end, w_end))
                    
                    # 如果有标签，也提取对应的标签patch
                    if label is not None:
                        patch_label = label[d:d_end, h:h_end, w:w_end]
                        
                        if patch_label.shape != self.patch_size:
                            patch_label = np.pad(patch_label, 
                                               ((0, pad_d), (0, pad_h), (0, pad_w)), 
                                               mode='constant')
                        
                        patches_label.append(patch_label)
        
        return np.array(patches_img), np.array(patches_label) if label is not None else None, patch_locations
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 加载CT图像
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = self._load_volume(image_path)
        image = self._preprocess(image)
        
        # 训练或验证模式：加载标签并提取patches
        if self.mode != 'test' and self.labels_dir is not None:
            label_path = os.path.join(self.labels_dir, self.label_files[idx])
            label = self._load_volume(label_path)
            label = (label > 0).astype(np.float32)  # 转换为二值掩码
            
            if self.mode == 'train':
                # 训练时提取patches并选择一个
                patches_img, patches_label, _ = self._extract_patches(image, label)
                if len(patches_img) > 0:
                    patch_idx = np.random.randint(len(patches_img))
                    image_patch = patches_img[patch_idx]
                    label_patch = patches_label[patch_idx]
                else:
                    # 如果没有patches（极少情况），直接裁剪或填充
                    image_patch = self._resize_volume(image, self.patch_size)
                    label_patch = self._resize_volume(label, self.patch_size)
                
                # 扩展通道维度
                image_patch = np.expand_dims(image_patch, axis=0)
                label_patch = np.expand_dims(label_patch, axis=0)
                
                # 数据增强
                if self.transform is not None:
                    image_patch, label_patch = self.transform(image_patch, label_patch)
                
                return torch.FloatTensor(image_patch), torch.FloatTensor(label_patch)
            else:
                # 验证模式：返回完整图像和标签
                image = np.expand_dims(image, axis=0)
                label = np.expand_dims(label, axis=0)
                return torch.FloatTensor(image), torch.FloatTensor(label), self.image_files[idx]
        
        # 测试模式：只返回图像
        else:
            image = np.expand_dims(image, axis=0)
            return torch.FloatTensor(image), self.image_files[idx]
    
    def _resize_volume(self, volume, target_shape):
        """调整体积大小"""
        # 如果体积太大，裁剪中心区域
        current_shape = volume.shape
        if all(c >= t for c, t in zip(current_shape, target_shape)):
            # 居中裁剪
            start = [(c - t) // 2 for c, t in zip(current_shape, target_shape)]
            end = [s + t for s, t in zip(start, target_shape)]
            slices = tuple(slice(s, e) for s, e in zip(start, end))
            return volume[slices]
        else:
            # 填充到目标大小
            result = np.zeros(target_shape, dtype=volume.dtype)
            # 计算放置位置（居中）
            start = [(t - c) // 2 for c, t in zip(current_shape, target_shape)]
            end = [s + c for s, c in zip(start, current_shape)]
            # 获取源和目标的切片
            target_slices = tuple(slice(s, e) for s, e in zip(start, end))
            source_slices = tuple(slice(0, c) for c in current_shape)
            # 填充
            result[target_slices] = volume[source_slices]
            return result

class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP3D, self).__init__()
        
        # 1x1 卷积
        self.aspp1 = nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 减少空洞卷积层数，仅保留必要的空洞率
        self.aspp2 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=rates[0], dilation=rates[0], bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.aspp3 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=rates[2], dilation=rates[2], bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 全局平均池化 - 简化为更高效的版本
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 合并减少的分支数量
        self.conv1 = nn.Conv3d(out_channels * 4, out_channels, 1, bias=False)  # 从5个分支减少到4个
        self.bn5 = nn.BatchNorm3d(out_channels)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        feature_size = x.size()[2:]
        
        x1 = self.relu1(self.bn1(self.aspp1(x)))
        x2 = self.relu2(self.bn2(self.aspp2(x)))
        x3 = self.relu3(self.bn3(self.aspp3(x)))
        
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=feature_size, mode='trilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu5(self.bn5(self.conv1(x)))
        return self.dropout(x)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class DeepLabV3Plus3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(DeepLabV3Plus3D, self).__init__()
        
        # 保持合理的基础通道数
        base_filters = 64
        
        # 输入卷积
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 主干网络 - 减少每层块数
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=1, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, blocks=1, stride=2)
        
        # ASPP模块
        self.aspp = ASPP3D(base_filters*8, 256)
        
        # 解码器低级特征处理
        self.low_level_conv = nn.Conv3d(base_filters*2, 48, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm3d(48)
        self.low_level_relu = nn.ReLU(inplace=True)
        
        # 解码器 - 简化
        self.decoder = nn.Sequential(
            nn.Conv3d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv3d(256, num_classes, kernel_size=1, stride=1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 保存输入尺寸用于最终上采样
        input_shape = x.shape[-3:]
        
        # 编码阶段
        x = self.conv1(x)           # 下采样2倍
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 下采样2倍 (总共4倍)
        
        x = self.layer1(x)
        low_level_features = self.layer2(x)  # 保存低级特征 (总共8倍)
        x = self.layer3(low_level_features)  # 总共16倍
        x = self.layer4(x)                   # 总共32倍
        
        # ASPP
        x = self.aspp(x)
        
        # 上采样到低级特征尺寸 (从32倍到8倍)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='trilinear', align_corners=True)
        
        # 处理低级特征
        low_level_features = self.low_level_conv(low_level_features)
        low_level_features = self.low_level_bn(low_level_features)
        low_level_features = self.low_level_relu(low_level_features)
        
        # 融合低级和高级特征
        x = torch.cat((x, low_level_features), dim=1)
        
        # 解码器
        x = self.decoder(x)
        
        # 上采样到原始大小 - 这里是问题所在
        # 将x上采样到输入尺寸，而不是使用固定的scale_factor
        x = F.interpolate(x, size=input_shape, mode='trilinear', align_corners=True)
        
        return x

# 添加Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # 将预测转换为概率
        predictions = torch.sigmoid(predictions)
        
        # 平滑预测和标签
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (predictions * targets).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # 返回Dice损失
        return 1 - dice

# 添加BCE+Dice组合损失
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=1.0, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss

# 添加Dice系数计算函数（用于评估）
def dice_coefficient(predictions, targets, smooth=1.0):
    """计算Dice系数，用于评估分割结果
    
    Args:
        predictions: 预测值
        targets: 目标值
        smooth: 平滑因子，避免除零
        
    Returns:
        dice: Dice系数
    """
    predictions = torch.sigmoid(predictions)
    
    # 平滑预测和标签
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # 计算交集
    intersection = (predictions * targets).sum()
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice

def hausdorff_distance(pred, gt, spacing=(1, 1, 1)):
    """计算Hausdorff距离"""
    from scipy.ndimage import distance_transform_edt
    
    # 转换为二值图像
    pred = pred.cpu().numpy() > 0.5
    gt = gt.cpu().numpy() > 0
    
    # 计算距离图
    if np.any(pred) and np.any(gt):
        pred_dist = distance_transform_edt(~pred, sampling=spacing)
        gt_dist = distance_transform_edt(~gt, sampling=spacing)
        
        # 计算Hausdorff距离
        hd95 = np.percentile(pred_dist[gt], 95)
        return hd95
    elif np.any(pred) or np.any(gt):
        return np.inf
    else:
        return 0.0

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch=None, num_epochs=None):
    """标准训练函数，支持可选的epoch参数"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    steps = 0
    
    # 添加条件判断，如果提供了epoch参数则显示
    if epoch is not None and num_epochs is not None:
        print(f"Epoch {epoch+1}/{num_epochs} - 训练中...")
    else:
        print("训练中...")
    
    for images, labels in dataloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        # 标准训练步骤
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 计算Dice系数
        with torch.no_grad():
            dice = dice_coefficient(outputs, labels)
        
        # 更新指标
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        
        # 简单输出每10步的进度
        if steps % 10 == 0:
            print(f"  步骤 {steps}/{len(dataloader)}, Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
    
    # 输出epoch平均值
    avg_loss = epoch_loss / steps
    avg_dice = epoch_dice / steps
    print(f"  训练完成 - Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice

def validate(model, dataloader, criterion, device, epoch=None, num_epochs=None):
    """验证函数，支持可选的epoch参数"""
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    steps = 0
    
    # 添加条件判断，如果提供了epoch参数则显示
    if epoch is not None and num_epochs is not None:
        print(f"Epoch {epoch+1}/{num_epochs} - 验证中...")
    else:
        print("验证中...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算Dice系数
            dice = dice_coefficient(outputs, labels)
            
            # 更新指标
            epoch_loss += loss.item()
            epoch_dice += dice.item()
    
    # 计算平均值
    avg_loss = epoch_loss / steps
    avg_dice = epoch_dice / steps
    print(f"  验证完成 - Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice

def print_gpu_memory_usage():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(f"GPU内存使用: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 最大使用={max_memory:.2f}GB")
    else:
        print("没有可用的GPU")

def train_epoch_amp(model, dataloader, criterion, optimizer, scheduler, device, scaler=None, epoch=0, num_epochs=0):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    steps = 0
    
    # 简单进度条
    print(f"Epoch {epoch+1}/{num_epochs} - 训练中...")
    for images, labels in dataloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 计算Dice系数
        with torch.no_grad():
            dice = dice_coefficient(outputs, labels)
        
        # 更新指标
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        
        # 简单输出每10步的进度
        if steps % 10 == 0:
            print(f"  步骤 {steps}/{len(dataloader)}, Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
    
    # 输出epoch平均值
    avg_loss = epoch_loss / steps
    avg_dice = epoch_dice / steps
    print(f"  训练完成 - Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               num_epochs, device, save_dir, train_epoch_fn=None, scaler=None):
    """训练模型（极简版本）"""
    
    # 初始化指标列表
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    # 记录最佳验证Dice系数
    best_val_dice = 0.0
    best_epoch = 0
    
    # 开始训练
    print(f"\n{'='*30} 开始训练 {'='*30}")
    print(f"总共训练 {num_epochs} 个Epochs\n")
    
    for epoch in range(num_epochs):
        # 训练一个epoch - 传递可选参数
        if train_epoch_fn:
            train_loss, train_dice = train_epoch_fn(
                model=model, 
                dataloader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                device=device, 
                scaler=scaler, 
                epoch=epoch, 
                num_epochs=num_epochs
            )
        else:
            train_loss, train_dice = train_epoch(
                model=model, 
                dataloader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                device=device, 
                epoch=epoch, 
                num_epochs=num_epochs
            )
        
        # 验证 - 传递可选参数
        val_loss, val_dice = validate(
            model=model, 
            dataloader=val_loader, 
            criterion=criterion, 
            device=device, 
            epoch=epoch, 
            num_epochs=num_epochs
        )
        
        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        # 打印epoch总结
        print(f"\n【Epoch {epoch+1}/{num_epochs}】")
        print(f"  训练: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
        print(f"  验证: Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        
        # 检查是否为最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ? 已保存最佳模型: {val_dice:.4f}")
        
        # 每5个epoch保存检查点
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_dices': train_dices,
                'val_dices': val_dices
            }, checkpoint_path)
            print(f"  * 已保存检查点: {checkpoint_path}")
        
        print("-" * 50)  # 分隔线
    
    print(f"\n{'='*30} 训练完成 {'='*30}")
    print(f"最佳模型: Epoch {best_epoch}/{num_epochs}, 验证Dice: {best_val_dice:.4f}")
    
    return model, train_losses, val_losses, train_dices, val_dices

def predict_segmentation(model, image_path, output_path=None, device='cuda'):
    """使用训练好的模型进行分割预测"""
    # 设置模型为评估模式
    model.eval()
    
    # 加载和预处理图像
    if image_path.endswith(('.nii', '.nii.gz')):
        nii_img = nib.load(image_path)
        image = np.array(nii_img.get_fdata())
        affine = nii_img.affine
    else:
        sitk_img = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(sitk_img)
        
    # 预处理
    orig_shape = image.shape
    image = np.clip(image, -1000, 1000)
    image = (image + 1000) / 2000
    
    # 添加批次和通道维度
    image = np.expand_dims(image, axis=(0, 1))
    image_tensor = torch.FloatTensor(image).to(device)
    
    # 使用滑动窗口进行预测
    patch_size = (128, 128, 128)
    D, H, W = orig_shape
    stride = (64, 64, 64)
    
    # 创建输出体积
    prediction = np.zeros(orig_shape, dtype=np.float32)
    weight = np.zeros(orig_shape, dtype=np.float32)
    
    # 创建3D高斯权重核
    def gaussian_kernel(size):
        sigma = size/8
        x = np.linspace(-size/2, size/2, size)
        x, y, z = np.meshgrid(x, x, x)
        kernel = np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2))
        return kernel/kernel.max()
    
    weight_kernel = gaussian_kernel(patch_size[0])
    
    # 计算滑动窗口的步长
    d_steps = range(0, max(1, D - patch_size[0] + 1), stride[0])
    h_steps = range(0, max(1, H - patch_size[1] + 1), stride[1])
    w_steps = range(0, max(1, W - patch_size[2] + 1), stride[2])
    
    # 确保至少有一个步长
    if not d_steps:
        d_steps = [0]
    if not h_steps:
        h_steps = [0]
    if not w_steps:
        w_steps = [0]
    
    with torch.no_grad():
        for d in tqdm(d_steps, desc="处理深度维度"):
            for h in h_steps:
                for w in w_steps:
                    # 计算patch结束位置
                    d_end = min(d + patch_size[0], D)
                    h_end = min(h + patch_size[1], H)
                    w_end = min(w + patch_size[2], W)
                    
                    # 提取patch
                    patch = image[:, :, d:d_end, h:h_end, w:w_end]
                    
                    # 如果patch尺寸小于目标尺寸，进行填充
                    pad_d = max(0, patch_size[0] - (d_end - d))
                    pad_h = max(0, patch_size[1] - (h_end - h))
                    pad_w = max(0, patch_size[2] - (w_end - w))
                    
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        patch = torch.nn.functional.pad(
                            patch, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0
                        )
                    
                    # 前向传播
                    output = model(patch)
                    pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                    
                    # 如果进行了填充，去除填充部分
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        pred = pred[:d_end-d, :h_end-h, :w_end-w]
                        kernel = weight_kernel[:d_end-d, :h_end-h, :w_end-w]
                    else:
                        kernel = weight_kernel
                    
                    # 添加到输出体积并应用权重
                    prediction[d:d_end, h:h_end, w:w_end] += pred * kernel
                    weight[d:d_end, h:h_end, w:w_end] += kernel
    
    # 计算最终预测，处理权重为0的情况
    prediction = np.divide(prediction, weight, out=np.zeros_like(prediction), where=weight!=0)
    
    # 后处理
    final_pred = post_process_prediction(prediction)
    
    # 保存结果
    if output_path:
        if output_path.endswith(('.nii', '.nii.gz')):
            result_img = nib.Nifti1Image(final_pred.astype(np.uint8), affine)
            nib.save(result_img, output_path)
        else:
            result_sitk = sitk.GetImageFromArray(final_pred.astype(np.uint8))

# 创建训练和验证集
# Create training and validation sets
def create_train_val_split(images_dir, labels_dir, val_ratio=0.2, seed=42):
    """从同一目录创建训练集和验证集分割
    
    Args:
        images_dir: 图像目录
        labels_dir: 标签目录
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_images, train_labels, val_images, val_labels: 训练和验证集文件列表
    """
    import random
    random.seed(seed)
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))])
    
    # 获取所有标签文件
    label_files = sorted([f for f in os.listdir(labels_dir)
                         if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))])
    
    # 确保图像和标签一一对应
    assert len(image_files) == len(label_files), "图像和标签数量不匹配"
    
    # 验证文件名匹配
    for img, lbl in zip(image_files, label_files):
        img_base = os.path.splitext(img)[0].replace('.nii', '')
        lbl_base = os.path.splitext(lbl)[0].replace('.nii', '')
        if img_base != lbl_base:
            print(f"警告: 图像 {img} 和标签 {lbl} 文件名不匹配")
    
    # 创建索引并打乱
    indices = list(range(len(image_files)))
    random.shuffle(indices)
    
    # 分割为训练集和验证集
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # 创建训练集和验证集文件列表
    train_images = [os.path.join(images_dir, image_files[i]) for i in train_indices]
    train_labels = [os.path.join(labels_dir, label_files[i]) for i in train_indices]
    val_images = [os.path.join(images_dir, image_files[i]) for i in val_indices]
    val_labels = [os.path.join(labels_dir, label_files[i]) for i in val_indices]
    
    print(f"划分数据集完成:")
    print(f"  - 训练集: {len(train_images)} 样本")
    print(f"  - 验证集: {len(val_images)} 样本")
    
    return train_images, train_labels, val_images, val_labels

# 修改FileListDataset类，确保始终返回固定大小的patch
class FileListDataset(Dataset):
    def __init__(self, image_files, label_files=None, patch_size=(192, 192, 192), 
                 transform=None, mode='train'):
        """
        基于文件列表的数据集，确保始终返回固定大小的patch
        
        Args:
            image_files: 图像文件路径列表
            label_files: 标签文件路径列表
            patch_size: 3D patch大小
            transform: 数据增强变换
            mode: 'train', 'val', 或 'test'
        """
        self.image_files = image_files
        self.label_files = label_files
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
            
    def __len__(self):
        return len(self.image_files)
    
    def _load_volume(self, file_path):
        """加载体积数据"""
        import SimpleITK as sitk
        import nibabel as nib
        import numpy as np
        
        if file_path.endswith(('.nii', '.nii.gz')):
            nii_img = nib.load(file_path)
            return np.array(nii_img.get_fdata())
        else:
            sitk_img = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(sitk_img)
    
    def _preprocess(self, image):
        """CT图像预处理"""
        # 裁剪HU值并归一化到0-1
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000
        return image
    
    def _extract_random_patch(self, image, label=None):
        """从体积数据中提取随机3D patch"""
        import numpy as np
        
        D, H, W = image.shape
        
        # 确保图像至少与patch_size一样大
        if D < self.patch_size[0] or H < self.patch_size[1] or W < self.patch_size[2]:
            # 对小于目标尺寸的维度进行填充
            pad_d = max(0, self.patch_size[0] - D)
            pad_h = max(0, self.patch_size[1] - H)
            pad_w = max(0, self.patch_size[2] - W)
            
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            if label is not None:
                label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            
            # 更新尺寸
            D, H, W = image.shape
        
        # 随机选择patch的起始位置
        d_start = np.random.randint(0, D - self.patch_size[0] + 1)
        h_start = np.random.randint(0, H - self.patch_size[1] + 1)
        w_start = np.random.randint(0, W - self.patch_size[2] + 1)
        
        # 提取patch
        patch_img = image[
            d_start:d_start + self.patch_size[0],
            h_start:h_start + self.patch_size[1],
            w_start:w_start + self.patch_size[2]
        ]
        
        if label is not None:
            patch_label = label[
                d_start:d_start + self.patch_size[0],
                h_start:h_start + self.patch_size[1],
                w_start:w_start + self.patch_size[2]
            ]
            return patch_img, patch_label
        
        return patch_img
    
    def __getitem__(self, idx):
        """获取数据样本"""
        import numpy as np
        
        # 获取图像文件路径
        image_path = self.image_files[idx]
        
        # 加载图像
        image = self._load_volume(image_path)
        image = self._preprocess(image)
        
        if self.mode == 'train' or (self.mode == 'val' and self.label_files is not None):
            # 加载标签
            label_path = self.label_files[idx]
            label = self._load_volume(label_path)
            
            # 二值化标签
            label = (label > 0).astype(np.float32)
            
            if self.mode == 'train':
                # 训练模式：随机提取patch
                image_patch, label_patch = self._extract_random_patch(image, label)
                
                # 扩展通道维度
                image_patch = np.expand_dims(image_patch, axis=0)
                label_patch = np.expand_dims(label_patch, axis=0)
                
                # 数据增强
                if self.transform is not None:
                    image_patch, label_patch = self.transform(image_patch, label_patch)
                
                return torch.FloatTensor(image_patch), torch.FloatTensor(label_patch)
            else:
                # 验证模式：提取中心patch
                # 计算中心位置
                D, H, W = image.shape
                d_center = max(0, (D - self.patch_size[0]) // 2)
                h_center = max(0, (H - self.patch_size[1]) // 2)
                w_center = max(0, (W - self.patch_size[2]) // 2)
                
                # 提取中心patch
                image_patch = self._extract_random_patch(image, None)
                label_patch = self._extract_random_patch(label, None)
                
                # 扩展通道维度
                image_patch = np.expand_dims(image_patch, axis=0)
                label_patch = np.expand_dims(label_patch, axis=0)
                
                return torch.FloatTensor(image_patch), torch.FloatTensor(label_patch)
        
        # 测试模式：返回整个图像
        # 因为我们只是训练，先简化处理
        image_patch = self._extract_random_patch(image, None)
        image_patch = np.expand_dims(image_patch, axis=0)
        return torch.FloatTensor(image_patch)

# 添加自定义的collate函数，处理不同尺寸的体积
def collate_fn_pad(batch):
    """处理不同尺寸的体积数据的批处理函数
    
    将每个样本都填充到批次中最大尺寸
    """
    # 确定批次中每个维度的最大尺寸
    if isinstance(batch[0], tuple) and len(batch[0]) >= 2:
        # 处理训练和验证数据 (图像,标签)
        max_size = tuple(max(s) for s in zip(*[img.shape for img, _ in batch]))
        
        # 填充图像和标签
        padded_imgs = []
        padded_labels = []
        
        for img, label in batch:
            # 计算需要填充的尺寸
            pad_size = tuple(max_s - s for max_s, s in zip(max_size, img.shape))
            # 对每个维度的前后都进行填充，所以每个维度需要除以2
            padding = []
            for p in reversed(pad_size):  # 反转是因为F.pad期望的顺序
                padding.extend([0, p])
                
            # 填充图像和标签
            padded_img = F.pad(img, padding, "constant", 0)
            padded_label = F.pad(label, padding, "constant", 0)
            
            padded_imgs.append(padded_img)
            padded_labels.append(padded_label)
            
        # 将填充后的样本堆叠为批次
        return torch.stack(padded_imgs), torch.stack(padded_labels)
    else:
        # 处理测试数据 (只有图像)
        max_size = tuple(max(s) for s in zip(*[img.shape for img in batch]))
        
        # 填充图像
        padded_imgs = []
        for img in batch:
            # 计算需要填充的尺寸
            pad_size = tuple(max_s - s for max_s, s in zip(max_size, img.shape))
            padding = []
            for p in reversed(pad_size):
                padding.extend([0, p])
                
            # 填充图像
            padded_img = F.pad(img, padding, "constant", 0)
            padded_imgs.append(padded_img)
            
        # 将填充后的样本堆叠为批次
        return torch.stack(padded_imgs)

def main():
    # 配置参数 - 优化训练速度和内存使用
    config = {
        'data_dir': data_base_dir,
        'train_images_dir': train_images_dir,
        'train_labels_dir': train_labels_dir,
        'output_dir': output_dir,
        'patch_size': (192, 192, 192),  # 适度的patch大小，平衡上下文和速度
        'batch_size': 8,                # 保持小批量以确保稳定性
        'learning_rate': 2e-4,          # 增加学习率以加速收敛
        'num_epochs': 100,              # 减少总训练轮数
        'num_workers': 4,               # 增加数据加载线程
        'val_ratio': 0.2,
        'seed': 42,
        'use_amp': True,                # 继续使用混合精度训练
        'save_interval': 5,             # 每5个epoch保存一次模型
        'val_interval': 1,              # 每个epoch进行一次验证
    }
    
    print("使用优化版DeepLabV3+模型(更快速、更高效)")
    
    # 创建训练集和验证集分割
    train_images, train_labels, val_images, val_labels = create_train_val_split(
        config['train_images_dir'], 
        config['train_labels_dir'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    # 创建训练和验证数据集 - 优化数据加载
    train_dataset = FileListDataset(
        image_files=train_images,
        label_files=train_labels,
        patch_size=config['patch_size'],
        transform=None,  # 简化 - 移除数据增强以加速训练
        mode='train'
    )
    
    val_dataset = FileListDataset(
        image_files=val_images,
        label_files=val_labels,
        patch_size=config['patch_size'],
        transform=None,
        mode='val'
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器 - 使用自定义collate函数处理不同尺寸的体积
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn_pad  # 添加自定义collate函数
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn_pad  # 添加自定义collate函数
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = DeepLabV3Plus3D(in_channels=1, num_classes=1).to(device)
    
    # 使用最新的Adam优化器变体和更优的学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate']*10,  # 最大学习率
        steps_per_epoch=len(train_loader),
        epochs=config['num_epochs'],
        pct_start=0.3,  # 前30%的时间用于预热
        div_factor=25,  # 初始学习率 = max_lr/div_factor
        final_div_factor=1000  # 最终学习率 = initial_lr/final_div_factor
    )
    
    # 损失函数 - 使用组合损失加速训练
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=1.0)
    
    # 混合精度训练设置
    scaler = GradScaler('cuda') if config['use_amp'] else None
    
    # 训练模型
    model, train_losses, val_losses, train_dices, val_dices = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # 确保传递scheduler
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=config['output_dir'],
        train_epoch_fn=train_epoch_amp,
        scaler=scaler
    )
    
    # 保存最终模型
    final_model_path = os.path.join(config['output_dir'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices
    }, final_model_path)
    
    print("训练完成!")

if __name__ == "__main__":
    main()