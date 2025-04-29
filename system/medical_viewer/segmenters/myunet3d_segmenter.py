import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import traceback
from monai.inferers import sliding_window_inference

# Define the DepthwiseSeparableConv3d class first
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Define the AttentionGate class
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Now define the UNet3D class
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                # First depthwise separable convolution
                DepthwiseSeparableConv3d(in_ch, out_ch),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                # Second depthwise separable convolution
                DepthwiseSeparableConv3d(out_ch, out_ch),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                # SE block for channel attention
                SEBlock(out_ch)
            )

        # Squeeze and Excitation Block for channel attention
        class SEBlock(nn.Module):
            def __init__(self, channel, reduction=16):
                super(SEBlock, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1, 1)
                return x * y.expand_as(x)

        # Encoder
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        # Attention Gates
        self.attention1 = AttentionGate(512, 512, 256)
        self.attention2 = AttentionGate(256, 256, 128)
        self.attention3 = AttentionGate(128, 128, 64)
        self.attention4 = AttentionGate(64, 64, 32)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        # Encoder path with residual connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.dropout(self.pool(e1)))
        e3 = self.encoder3(self.dropout(self.pool(e2)))
        e4 = self.encoder4(self.dropout(self.pool(e3)))

        # Bottleneck
        b = self.bottleneck(self.dropout(self.pool(e4)))

        # Decoder path with attention and skip connections
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

class MyUNet3DSegmenter:
    def __init__(self, weights_path, device='cuda'):
        self.NUM_CLASSES = 31
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = UNet3D(in_channels=1, out_channels=self.NUM_CLASSES).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def segment(self, image_3d: np.ndarray) -> np.ndarray:
        img = np.clip(image_3d, -1000, 1000)
        img = (img + 1000) / 2000
        with torch.no_grad():
            input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            # 使用滑窗推理
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(64, 128, 128),  # 你可以根据实际显存调整
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5
            )
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        return pred.astype(np.uint8)

    def get_colored_segmentation(self, segmentation):
        # segmentation: [D, H, W]
        color_map = {
            'background': [0, 0, 0, 0],
            'sacrum': [255, 0, 0, 180],
            'left_hip': [0, 255, 0, 180],
            'right_hip': [0, 0, 255, 180]
        }
        colored = np.zeros((*segmentation.shape, 4), dtype=np.uint8)
        colored[(segmentation >= 1) & (segmentation <= 10)] = color_map['sacrum']
        colored[(segmentation >= 11) & (segmentation <= 20)] = color_map['left_hip']
        colored[(segmentation >= 21) & (segmentation <= 30)] = color_map['right_hip']
        return colored

    def save_mask(self, mask, file_path):
        sitk_mask = sitk.GetImageFromArray(mask)
        sitk.WriteImage(sitk_mask, file_path)

    def get_probability_map(self, image_3d):
        img = np.clip(image_3d, -1000, 1000)
        img = (img + 1000) / 2000
        with torch.no_grad():
            input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # [C, D, H, W]
        return probs

    def get_color_legend(self):
        return {
            '骶骨 (1-10)': [255, 0, 0],
            '左髋骨 (11-20)': [0, 255, 0],
            '右髋骨 (21-30)': [0, 0, 255]
        }
