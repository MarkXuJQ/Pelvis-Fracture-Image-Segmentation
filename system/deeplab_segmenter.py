import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Union, Tuple, Optional, List
import math

# 自定义DeepLabV3模型架构，与保存的权重匹配
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # ASPP部分
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv1 = nn.Conv2d(out_channels*4, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x1 = F.relu(self.bn1(self.aspp1(x)))
        x2 = F.relu(self.bn2(self.aspp2(x)))
        x3 = F.relu(self.bn3(self.aspp3(x)))
        
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv1(x)))
        return x

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDeepLabV3, self).__init__()
        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet层
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2)
        )
        
        # ASPP模块
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.low_level_conv = nn.Conv2d(128, 48, 1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        # 编码器部分
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.layer1(x)
        low_level_feat = self.layer2(x)  # 用于跳跃连接
        x = self.layer3(low_level_feat)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        low_level_feat = F.relu(self.low_level_bn(self.low_level_conv(low_level_feat)))
        
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder(x)
        
        # 恢复到原始分辨率
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        return {'out': x}

class ResBlock3D(nn.Module):
    """3D残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ASPP3D(nn.Module):
    """3D空洞空间金字塔池化模块"""
    def __init__(self, in_channels, out_channels):
        super(ASPP3D, self).__init__()
        # ASPP部分
        self.aspp1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.aspp2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.aspp3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        self.conv1 = nn.Conv3d(out_channels*4, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x1 = F.relu(self.bn1(self.aspp1(x)))
        x2 = F.relu(self.bn2(self.aspp2(x)))
        x3 = F.relu(self.bn3(self.aspp3(x)))
        
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv1(x)))
        return x

class CustomDeepLabV3_3D(nn.Module):
    """自定义3D DeepLabV3网络"""
    def __init__(self, num_classes=1, input_channels=1):
        super(CustomDeepLabV3_3D, self).__init__()
        # 初始层
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        
        # ResNet层
        self.layer1 = nn.Sequential(
            ResBlock3D(64, 64),
            # 注意：原始权重似乎没有第二个块
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(64, 128, stride=2),
            ResBlock3D(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(128, 256, stride=2),
            ResBlock3D(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResBlock3D(256, 512, stride=2)
        )
        
        # ASPP模块
        self.aspp = ASPP3D(512, 256)
        
        # 低级特征处理
        self.low_level_conv = nn.Conv3d(128, 48, 1, bias=False)
        self.low_level_bn = nn.BatchNorm3d(48)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(256, num_classes, 1)
        )
        
    def forward(self, x):
        # 编码器
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        
        # 主干网络
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # ASPP
        x = self.aspp(x4)
        
        # 上采样到原始大小的1/4
        x = F.interpolate(x, size=x2.size()[2:], mode='trilinear', align_corners=True)
        
        # 低级特征处理
        low_level_features = self.low_level_conv(x2)
        low_level_features = F.relu(self.low_level_bn(low_level_features))
        
        # 连接特征
        x = torch.cat((x, low_level_features), dim=1)
        
        # 解码器
        x = self.decoder(x)
        
        # 上采样到原始大小
        x = F.interpolate(x, scale_factor=8, mode='trilinear', align_corners=True)
        
        return {'out': x}

class DeeplabV3Segmenter:
    """DeepLabV3 3D分割器"""
    
    def __init__(self, checkpoint_path, device=None):
        """
        初始化DeepLabV3分割器
        
        参数:
            checkpoint_path: 权重文件路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        # 设置设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DeepLabV3使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, checkpoint_path):
        """加载DeepLabV3模型及权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"权重文件未找到: {checkpoint_path}")
        
        # 创建DeepLabV3 3D模型
        model = CustomDeepLabV3_3D(num_classes=1, input_channels=1)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        return model
    
    def preprocess_volume(self, volume):
        """预处理输入体积"""
        # 确保是浮点数类型
        if volume.dtype != np.float32:
            volume = volume.astype(np.float32)
        
        # 标准化
        if volume.max() > 0:
            volume = volume / volume.max()
        
        # 转换为张量
        volume_tensor = torch.from_numpy(volume).float()
        
        # 添加批次和通道维度 [B, C, D, H, W]
        if len(volume_tensor.shape) == 3:
            volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
            
        return volume_tensor.to(self.device)
    
    def segment(self, image, points=None, point_labels=None, box=None):
        """
        执行2D分割 (针对单个切片)
        
        参数:
            image: 输入图像(numpy数组)
            points: 点提示(忽略)
            point_labels: 点标签(忽略)
            box: 感兴趣区域(可选)
            
        返回:
            二值掩码(numpy数组)
        """
        # 为单个切片创建伪3D体积
        pseudo_volume = np.expand_dims(image, axis=0)
        
        if box is not None:
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            pseudo_volume = np.expand_dims(image[y1:y2, x1:x2], axis=0)
        
        # 预处理
        volume_tensor = self.preprocess_volume(pseudo_volume)
        
        # 分割
        with torch.no_grad():
            output = self.model(volume_tensor)['out']
            
            # 阈值处理
            pred_mask = (output > 0).squeeze().cpu().numpy()
        
        # 如果使用了边界框，恢复完整大小
        if box is not None:
            full_mask = np.zeros_like(image, dtype=bool)
            full_mask[y1:y2, x1:x2] = pred_mask[0]  # 使用第一个(唯一的)切片
            pred_mask = full_mask
        else:
            pred_mask = pred_mask[0]  # 使用第一个(唯一的)切片
            
        return pred_mask
    
    def segment_3d(self, volume, axis=0):
        """
        分割3D体积
        
        参数:
            volume: 3D体积数据(numpy数组)
            axis: 切片轴(0=axial, 1=coronal, 2=sagittal)
            
        返回:
            3D二值掩码(numpy数组)
        """
        # 准备体积数据
        volume_tensor = self.preprocess_volume(volume)
        
        # 分割
        with torch.no_grad():
            output = self.model(volume_tensor)['out']
            
            # 阈值处理
            pred_mask = (output > 0).squeeze().cpu().numpy()
            
        return pred_mask 