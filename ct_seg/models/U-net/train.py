import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from unet_model import UNet
from dataset import CTScanDataset
import SimpleITK as sitk

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

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor(),  # Convert NumPy array to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
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
