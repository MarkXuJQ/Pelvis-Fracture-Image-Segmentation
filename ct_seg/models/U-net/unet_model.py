import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CTScanDataset
import SimpleITK as sitk

# Define the 3D UNet model
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        def conv_block(in_ch, out_ch):
            block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block
        
        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
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

# Paths
data_dir = '../../data'
images_path = os.path.join(data_dir, 'PENGWIN_CT_train_images')  # Combined image directory
labels_path = os.path.join(data_dir, 'PENGWIN_CT_train_labels')
model_dir = '.'  # Path to save model checkpoints

# Transformations
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset and DataLoader
train_dataset = CTScanDataset(
    images_path,  # Use single path for images
    labels_path,
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Hyperparameters
learning_rate = 0.001  # Define the learning rate
num_epochs = 50
batch_size = 2

# Model
model = UNet3D(in_channels=1, out_channels=1).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')  # Initialize best loss for model checkpointing

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # Reset running loss at the start of each epoch

    for i, (images, masks) in enumerate(train_loader):
        # Convert to float32 and send to the correct device
        images = images.float().to(device)
        masks = masks.float().to(device)

        # Diagnostic print statements
        print("Image shape:", images.shape, "Mask shape:", masks.shape)
        print("Image device:", images.device, "Mask device:", masks.device)
        print("Image values (sample):", images[0, 0, :5, :5, :5])
        print("Mask values (sample):", masks[0, 0, :5, :5, :5])

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # Print every 10 steps for better tracking
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}')

    # Save checkpoint if average loss improves
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_unet_model.pth'))
        print(f'Saved improved model with average loss: {avg_loss:.4f}')


print('Training finished.')

