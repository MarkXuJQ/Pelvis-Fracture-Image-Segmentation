# ct_seg/models/unet_model.py
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # Encoder
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((e4, d4), dim=1))
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((e3, d3), dim=1))
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((e2, d2), dim=1))
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((e1, d1), dim=1))
        
        return self.output(d1)