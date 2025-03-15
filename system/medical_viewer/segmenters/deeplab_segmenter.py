"""
DeepLabV3+ 分割模型实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Union, Tuple, Optional, List
import math
import torchvision.models.segmentation as segmentation

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
    """DeepLabV3+ 分割模型"""
    
    def __init__(self, model_path=None):
        """初始化DeepLabV3+分割器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载预训练的DeepLabV3+模型
        self.model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=2)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: No model path provided or file doesn't exist. Using untrained model.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def segment(self, image):
        """
        使用DeepLabV3+进行图像分割
        
        参数:
            image: 输入图像，2D或3D numpy数组
            
        返回:
            mask: 分割掩码
        """
        # 转换为3通道图像
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image_rgb = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
            
        # 确保图像数据在0-1范围内
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0
            
        # 预处理图像
        input_tensor = self.transform(image_rgb.astype(np.float32))
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(input_batch)['out']
            output = F.softmax(output, dim=1)
            
        # 获取前景掩码
        mask = output[0, 1].cpu().numpy()
        
        return mask

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