# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset import CTScanDataset
from unet_model import UNet3D  # Import your custom UNet3D model
from torch.cuda.amp import autocast, GradScaler  # For mixed-precision training

# Paths
images_path = '../../data/PENGWIN_CT_train_images'  # Combined image directory
labels_path = '../../data/PENGWIN_CT_train_labels'
model_dir = './model_checkpoints'

# Dataset and DataLoader
transform = T.Compose([
    T.Normalize([0.5], [0.5])
])
dataset = CTScanDataset(images_path, labels_path, transform=transform)  # Use single image path
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Train-validation split (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
target_shape = (1, 64, 128, 128)  # Define a uniform shape for all images and labels
train_dataset = CTScanDataset(images_path_part1, images_path_part2, labels_path, transform=transform, target_shape=target_shape)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model
model = UNet3D(in_channels=1, out_channels=1)  # Using your custom UNet3D model

# Loss, Optimizer, and Scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Mixed-precision training setup
scaler = GradScaler()  # Mixed precision scaling

# Training and validation loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training loop
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward and backward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
    
    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Save model checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, f'best_unet_model.pth'))
        print(f'Saved improved model at epoch {epoch+1} with validation loss: {val_loss:.4f}')

print('Training complete. Best validation loss:', best_val_loss)