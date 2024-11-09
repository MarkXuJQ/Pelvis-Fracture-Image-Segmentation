import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CTScanDataset
import SimpleITK as sitk

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block
        
        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
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
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.output(d1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
data_dir = '../../data'  # Adjust this path if necessary
model_dir = '.'  # Set to the current folder to save model checkpoints
output_dir = '../../outputs'

# Hyperparameters
learning_rate = 1e-4
num_epochs = 50
batch_size = 4

# Dataset and DataLoader
train_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CTScanDataset(
    os.path.join(data_dir, 'PENGWIN_CT_train_images_part1'),
    os.path.join(data_dir, 'PENGWIN_CT_train_images_part2'),
    os.path.join(data_dir, 'PENGWIN_CT_train_labels'),
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = UNet(in_channels=1, out_channels=1).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(model_dir, f'unet_epoch_{epoch+1}.pth'))
    print(f'Epoch [{epoch+1}/{num_epochs}] completed with average loss: {running_loss/len(train_loader):.4f}')

print('Training finished.')

# Save final model
torch.save(model.state_dict(), os.path.join(model_dir, 'unet_final.pth'))
