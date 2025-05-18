import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
import sys

# 复制UNet2D结构（来自xray_seg/training_code/train_3stage_pipline.py）
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AttentionGate2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate2D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        def conv_block(in_ch, out_ch):
            class SEBlock2D(nn.Module):
                def __init__(self, channel, reduction=16):
                    super(SEBlock2D, self).__init__()
                    self.avg_pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Sequential(
                        nn.Linear(channel, channel // reduction, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Linear(channel // reduction, channel, bias=False),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    b, c, _, _ = x.size()
                    y = self.avg_pool(x).view(b, c)
                    y = self.fc(y).view(b, c, 1, 1)
                    return x * y.expand_as(x)
            return nn.Sequential(
                DepthwiseSeparableConv2d(in_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv2d(out_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                SEBlock2D(out_ch)
            )
        self.encoder1 = conv_block(in_channels, 8)
        self.encoder2 = conv_block(8, 16)
        self.encoder3 = conv_block(16, 32)
        self.encoder4 = conv_block(32, 64)
        self.attention1 = AttentionGate2D(64, 64, 32)
        self.attention2 = AttentionGate2D(32, 32, 16)
        self.attention3 = AttentionGate2D(16, 16, 8)
        self.attention4 = AttentionGate2D(8, 8, 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(64, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = conv_block(128, 64)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder3 = conv_block(64, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = conv_block(32, 16)
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = conv_block(16, 8)
        self.output = nn.Sequential(
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.dropout = nn.Dropout2d(p=0.3)
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.dropout(self.pool(e1)))
        e3 = self.encoder3(self.dropout(self.pool(e2)))
        e4 = self.encoder4(self.dropout(self.pool(e3)))
        b = self.bottleneck(self.dropout(self.pool(e4)))
        d4 = self.upconv4(b)
        e4 = self.attention1(d4, e4)
        d4 = self.decoder4(torch.cat((e4, d4), dim=1))
        d3 = self.upconv3(d4)
        e3 = self.attention2(d3, e3)
        d3 = self.decoder3(torch.cat((e3, d3), dim=1))
        d2 = self.upconv2(d3)
        e2 = self.attention3(d2, e2)
        d2 = self.decoder2(torch.cat((e2, d2), dim=1))
        d1 = self.upconv1(d2)
        e1 = self.attention4(d1, e1)
        d1 = self.decoder1(torch.cat((e1, d1), dim=1))
        return self.output(d1)

class XrayUnet2dSegmenter:
    """X光Unet2D分割器，适用于X光片2D分割"""
    def __init__(self, weights_path=None, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet2D(in_channels=1, out_channels=4)  # 0=背景, 1=骶骨, 2=左髋骨, 3=右髋骨
        if weights_path is not None and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image):
        # 输入: numpy数组 (H, W) 或 (H, W, 1)
        if len(image.shape) == 3:
            image = image[..., 0]
        img = image.astype(np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, axis=0)  # (1, 512, 512)
        img = np.expand_dims(img, axis=0)  # (1, 1, 512, 512)
        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
        return img_tensor

    def segment(self, image, *args, **kwargs):
        # image: numpy数组 (H, W) 或 (H, W, 1)
        img_tensor = self.preprocess(image)
        with torch.no_grad():
            output = self.model(img_tensor)  # (1, 4, 512, 512)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]  # (512, 512)
        return pred.astype(np.uint8)

    def get_colored_segmentation(self, mask):
        # mask: (H, W), 0=背景, 1=骶骨, 2=左髋骨, 3=右髋骨
        color_map = {
            0: [0, 0, 0, 0],
            1: [255, 0, 0, 180],    # 红色 骶骨
            2: [0, 255, 0, 180],    # 绿色 左髋骨
            3: [0, 0, 255, 180],    # 蓝色 右髋骨
        }
        h, w = mask.shape
        colored = np.zeros((h, w, 4), dtype=np.uint8)
        for k, v in color_map.items():
            colored[mask == k] = v
        return colored

    def get_color_legend(self):
        return {
            '骶骨 (1)': [255, 0, 0],
            '左髋骨 (2)': [0, 255, 0],
            '右髋骨 (3)': [0, 0, 255]
        } 