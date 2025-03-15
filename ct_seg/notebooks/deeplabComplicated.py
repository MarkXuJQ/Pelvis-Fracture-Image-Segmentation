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
        # 编码阶段
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        low_level_features = self.layer2(x)  # 保存低级特征
        x = self.layer3(low_level_features)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # 上采样到低级特征尺寸
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='trilinear', align_corners=True)
        
        # 处理低级特征
        low_level_features = self.low_level_conv(low_level_features)
        low_level_features = self.low_level_bn(low_level_features)
        low_level_features = self.low_level_relu(low_level_features)
        
        # 融合低级和高级特征
        x = torch.cat((x, low_level_features), dim=1)
        
        # 解码器
        x = self.decoder(x)
        
        # 上采样到原始大小
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=True)
        
        return x

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        
        # 组合损失
        return 0.5 * bce_loss + 0.5 * dice_loss

def dice_coefficient(y_pred, y_true, smooth=1.0):
    """计算Dice系数"""
    y_pred = torch.sigmoid(y_pred)
    y_pred_flat = y_pred.contiguous().view(-1)
    y_true_flat = y_true.contiguous().view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()
    return (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)

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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    steps = 0
    
    progress_bar = tqdm(dataloader)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算Dice系数
        dice = dice_coefficient(outputs, labels)
        
        # 更新指标
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        
        # 更新进度条
        progress_bar.set_description(f"Train - Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
    
    return epoch_loss / steps, epoch_dice / steps

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    steps = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader)
        for batch_idx, (images, labels, _) in enumerate(progress_bar):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算Dice系数
            dice = dice_coefficient(outputs, labels)
            
            # 更新指标
            val_loss += loss.item()
            val_dice += dice.item()
            
            # 更新进度条
            progress_bar.set_description(f"Val - Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
    
    return val_loss / steps, val_dice / steps

def print_gpu_memory_usage():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(f"GPU内存使用: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 最大使用={max_memory:.2f}GB")
    else:
        print("没有可用的GPU")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               num_epochs, device, save_dir, train_epoch_fn=None, scaler=None):
    """训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 设备
        save_dir: 保存目录
        train_epoch_fn: 自定义训练轮函数
        scaler: 混合精度训练的梯度缩放器
    
    Returns:
        model: 训练后的模型
        train_losses: 训练损失
        val_losses: 验证损失
        train_dices: 训练Dice系数
        val_dices: 验证Dice系数
    """
    best_dice = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果没有提供自定义训练函数，使用默认的train_epoch
    if train_epoch_fn is None:
        train_epoch_fn = train_epoch
    
    # 打印GPU内存使用情况
    print_gpu_memory_usage()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss, train_dice = train_epoch_fn(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # 如果提供了调度器，更新学习率
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        # 打印当前epoch结果
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证Dice: {val_dice:.4f}")
        
        # 打印GPU内存使用情况
        print_gpu_memory_usage()
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_dices': train_dices,
            'val_dices': val_dices
        }, os.path.join(save_dir, 'checkpoint.pth'))
        
        # 如果是epoch的倍数，保存图表
        if (epoch + 1) % 5 == 0:
            # 画损失和准确率曲线
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(train_dices, label='Train Dice')
            plt.plot(val_dices, label='Val Dice')
            plt.title('Dice Coefficient')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png'))
            plt.close()
    
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

# 基于文件列表的数据集类
# File list based dataset class
class FileListDataset(PelvicCTDataset):
    """基于文件列表的CT数据集"""
    def __init__(self, image_files, label_files=None, patch_size=(128, 128, 128), 
                transform=None, mode='train', stride=(64, 64, 64)):
        self.image_files = image_files
        self.label_files = label_files
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.stride = stride
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载CT图像
        image_path = self.image_files[idx]
        image = self._load_volume(image_path)
        image = self._preprocess(image)
        
        # 训练或验证模式：加载标签并提取patches
        if self.mode != 'test' and self.label_files is not None:
            label_path = self.label_files[idx]
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
                return torch.FloatTensor(image), torch.FloatTensor(label), os.path.basename(image_path)
        
        # 测试模式：只返回图像
        else:
            image = np.expand_dims(image, axis=0)
            return torch.FloatTensor(image), os.path.basename(image_path)

def main():
    # 使用混合精度训练
    # 在这里导入
    try:
        from torch.amp import autocast, GradScaler
        use_amp = True
        print("使用混合精度训练")
    except ImportError:
        use_amp = False
        print("PyTorch版本不支持自动混合精度，使用全精度训练")
    
    # 配置参数 - 进一步优化以充分利用A100 GPU
    config = {
        'data_dir': data_base_dir,
        'train_images_dir': train_images_dir,
        'train_labels_dir': train_labels_dir,
        'output_dir': output_dir,
        'patch_size': (288, 288, 288),  # 进一步增加3D patch大小
        'batch_size': 4,  # 保持批量大小适中以容纳更大的模型
        'learning_rate': 2e-4,  # 略微提高学习率以加快收敛
        'num_epochs': 150,
        'num_workers': 6,  # 增加worker数量以加快数据加载
        'val_ratio': 0.2,  # 验证集比例
        'seed': 42,
        'use_amp': use_amp,  # 使用混合精度训练
        'use_enhanced_model': True,  # 使用增强版模型
        'base_filters': 96  # 基础滤波器数量，决定模型复杂度
    }
    
    # 创建训练集和验证集分割
    train_images, train_labels, val_images, val_labels = create_train_val_split(
        config['train_images_dir'], 
        config['train_labels_dir'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    # 创建训练和验证数据集
    train_dataset = FileListDataset(
        image_files=train_images,
        label_files=train_labels,
        patch_size=config['patch_size'],
        mode='train'
    )
    
    val_dataset = FileListDataset(
        image_files=val_images,
        label_files=val_labels,
        patch_size=config['patch_size'],
        mode='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # 验证时通常使用batch_size=1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 初始化增强版模型
    if config['use_enhanced_model']:
        model = DeepLabV3Plus3D(in_channels=1, num_classes=1)
        print("使用增强版DeepLabV3+模型(更深、更宽、带注意力机制)")
    else:
        model = DeepLabV3Plus3D(in_channels=1, num_classes=1)
        print("使用标准DeepLabV3+模型")
    
    # 检查是否有可用的预训练权重
    checkpoint_path = os.path.join(config['output_dir'], 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从epoch {start_epoch}继续训练")
    else:
        start_epoch = 0
    
    # 损失函数和优化器
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # 使用OneCycleLR调度器代替余弦退火，可以更快收敛
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['learning_rate'], 
        total_steps=config['num_epochs'] * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 初始化混合精度训练的scaler
    scaler = GradScaler('cuda') if config['use_amp'] else None
    
    # 修改训练函数以支持混合精度训练
    def train_epoch_amp(model, dataloader, criterion, optimizer, device, scaler=None):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        steps = 0
        
        progress_bar = tqdm(dataloader)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            # 使用混合精度训练 - 更新autocast使用方式
            if scaler is not None:
                with autocast('cuda'):  # 修改这里为新的语法
                    # 前向传播
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # 反向传播和优化 - 使用scaler
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准全精度训练
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 计算Dice系数
            with torch.no_grad():
                dice = dice_coefficient(outputs, labels)
            
            # 更新指标
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            
            # 更新进度条
            progress_bar.set_description(
                f"Train - Loss: {loss.item():.4f}, Dice: {dice.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        return epoch_loss / steps, epoch_dice / steps
    
    # 使用增强的训练函数
    train_epoch_fn = train_epoch_amp if config['use_amp'] else train_epoch
    
    # 训练模型
    model, train_losses, val_losses, train_dices, val_dices = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # 我们在训练循环中手动调用scheduler.step()
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=config['output_dir'],
        train_epoch_fn=train_epoch_fn,
        scaler=scaler
    )
    
    # 保存最终模型
    final_model_path = os.path.join(config['output_dir'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices
    }, final_model_path)
    
    print("训练完成!")

if __name__ == "__main__":
    main()