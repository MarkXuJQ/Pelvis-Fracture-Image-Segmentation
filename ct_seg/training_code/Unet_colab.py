!pip install simpleitk
# 1. 创建父目录
import os
os.makedirs('/content/data/images', exist_ok=True)
os.makedirs('/content/data/labels', exist_ok=True)

# 2. 解压
!unzip -q /content/drive/MyDrive/ct_segmentation/data/PENGWIN_CT_train_images.zip -d /content/data/images
!unzip -q /content/drive/MyDrive/ct_segmentation/data/PENGWIN_CT_train_labels.zip -d /content/data/labels

# 3. 用本地路径
images_path = '/content/data/images/PENGWIN_CT_train_images'
labels_path = '/content/data/labels/PENGWIN_CT_train_labels'

# 后面再import、初始化数据集、DataLoader、训练等
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from google.colab import drive
from google.colab import files


# Print versions for reproducibility
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"SimpleITK version: {sitk.__version__}")
print(f"NumPy version: {np.__version__}")

# Define paths
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/data'
BASE_DIR = '/content/drive/MyDrive/ct_segmentation'  # Create a dedicated project folder

# Create organized subdirectories
PATHS = {
    'data': f'{BASE_DIR}/data',
    'models': f'{BASE_DIR}/models',
    'checkpoints': f'{BASE_DIR}/checkpoints',
    'results': f'{BASE_DIR}/results',
    'logs': f'{BASE_DIR}/logs'
}

# Create all directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# Update model save paths
MODEL_SAVE_PATH = os.path.join(PATHS['models'], 'best_unet_model.pth')
CHECKPOINT_DIR = PATHS['checkpoints']

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

def print_model_summary(model):
    """Print model architecture summary"""
    print("Model Architecture:")
    print("------------------")
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: {list(param.shape)} ({param_count:,} parameters)")
    print(f"\nTotal parameters: {total_params:,}")

# Initialize model and print summary
model = UNet3D(in_channels=1, out_channels=1)
print_model_summary(model)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 4  # 改回原来的配置
ACCUMULATION_STEPS = 8  # 改回原来的配置
PATCH_SIZE = (128, 128, 128)
STRIDE = (26, 26, 26)

# Dataset and DataLoader setup
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Normalize((0.5,), (0.5,))
])

class CTScanDataset(Dataset):
    def __init__(self, images_path, labels_path, patch_size=(128, 128, 128), stride=(26, 26, 26)):
        self.image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.mha')])
        self.label_paths = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.mha')])
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = 31  # 0=background, 1-10=sacrum, 11-20=left hip, 21-30=right hip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[idx])).astype(np.float32)
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_paths[idx])).astype(np.int64)  # Change to int64 for labels

        # Preprocessing
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000
        
        # No need to binarize labels anymore, keep original values (0-30)
        label = np.clip(label, 0, self.num_classes - 1)  # Ensure labels are in valid range

        # Extract random patch
        D, H, W = image.shape
        d = np.random.randint(0, max(1, D - self.patch_size[0] + 1))
        h = np.random.randint(0, max(1, H - self.patch_size[1] + 1))
        w = np.random.randint(0, max(1, W - self.patch_size[2] + 1))

        image_patch = image[d:d+self.patch_size[0], h:h+self.patch_size[1], w:w+self.patch_size[2]]
        label_patch = label[d:d+self.patch_size[0], h:h+self.patch_size[1], w:w+self.patch_size[2]]

        # Add channel dimension for image only
        image_patch = np.expand_dims(image_patch, axis=0)

        return torch.tensor(image_patch), torch.tensor(label_patch)

# Initialize dataset with patch-based approach
train_dataset = CTScanDataset(
    images_path=images_path,
    labels_path=labels_path,
    patch_size=PATCH_SIZE,
    stride=STRIDE
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    pin_memory=True,  # Keep memory optimizations
    num_workers=2     # Keep parallel loading
)

# Model initialization
NUM_CLASSES = 31  # 0=background, 1-10=sacrum, 11-20=left hip, 21-30=right hip
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss for multi-class
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
# 切换为ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True
)

class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.learning_rates = []
        self.current_epoch = 0

    def update(self, epoch_loss, learning_rate):
        self.train_losses.append(epoch_loss)
        self.learning_rates.append(learning_rate)
        self.current_epoch += 1

    def plot_progress(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot learning rate
        ax2.plot(self.learning_rates, label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def print_stats(self):
        print(f"Current epoch: {self.current_epoch}")
        print(f"Best loss: {min(self.train_losses):.4f}")
        print(f"Current loss: {self.train_losses[-1]:.4f}")
        print(f"Current learning rate: {self.learning_rates[-1]:.6f}")

# Initialize training monitor
monitor = TrainingMonitor()

# Update save paths at the start
best_model_dice_path = os.path.join(PATHS['models'], 'best_model_dice.pth')
best_model_loss_path = os.path.join(PATHS['models'], 'best_model_loss.pth')

def get_fragment_color(label):
    """
    Returns normalized color for different fragment types
    """
    if label == 0:  # Background
        return [0, 0, 0]
    elif 1 <= label <= 10:  # Sacrum fragments
        intensity = min(1.0, 0.3 + (label - 1) * 0.07)
        return [intensity, 0, 0]
    elif 11 <= label <= 20:  # Left hip fragments
        intensity = min(1.0, 0.3 + (label - 11) * 0.07)
        return [0, intensity, 0]
    elif 21 <= label <= 30:  # Right hip fragments
        intensity = min(1.0, 0.3 + (label - 21) * 0.07)
        return [0, 0, intensity]
    return [0, 0, 0]

def validate(model, val_image_path, val_label_path, patch_size=(128, 128, 128)):
    """
    Validate model performance on a single validation image
    Returns mean Dice score across all classes
    """
    model.eval()
    device = next(model.parameters()).device

    # Load validation image and label
    val_image = sitk.GetArrayFromImage(sitk.ReadImage(val_image_path)).astype(np.float32)
    val_label = sitk.GetArrayFromImage(sitk.ReadImage(val_label_path)).astype(np.int64)

    # Preprocess
    val_image = np.clip(val_image, -1000, 1000)
    val_image = (val_image + 1000) / 2000
    val_label = np.clip(val_label, 0, NUM_CLASSES - 1)

    # Predict using sliding window
    prediction = predict_volume(model, val_image, patch_size=patch_size)
    
    # Calculate Dice score for each class
    dice_scores = []
    for class_idx in range(1, NUM_CLASSES):  # Skip background class
        pred_class = (prediction == class_idx).astype(np.float32)
        true_class = (val_label == class_idx).astype(np.float32)
        
        intersection = np.sum(pred_class * true_class)
        dice_score = (2. * intersection) / (np.sum(pred_class) + np.sum(true_class) + 1e-7)
        dice_scores.append(dice_score)
    
    return np.mean(dice_scores)

def predict_volume(model, image, patch_size=(128, 128, 128), stride=(26, 26, 26)):
    """Predict segmentation using sliding window approach"""
    model.eval()
    device = next(model.parameters()).device

    D, H, W = image.shape
    output = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)
    weight = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)

    # Calculate steps
    d_steps = range(0, D - patch_size[0] + 1, stride[0])
    h_steps = range(0, H - patch_size[1] + 1, stride[1])
    w_steps = range(0, W - patch_size[2] + 1, stride[2])

    # Handle edge cases
    if D < patch_size[0]:
        d_steps = [0]
    if H < patch_size[1]:
        h_steps = [0]
    if W < patch_size[2]:
        w_steps = [0]

    with torch.no_grad():
        for d in d_steps:
            for h in h_steps:
                for w in w_steps:
                    # Extract patch
                    d_end = min(d + patch_size[0], D)
                    h_end = min(h + patch_size[1], H)
                    w_end = min(w + patch_size[2], W)

                    patch = image[d:d_end, h:h_end, w:w_end]

                    # Initialize padding values before potential assignment
                    pad_d, pad_h, pad_w = 0, 0, 0

                    # Pad if necessary
                    if patch.shape != patch_size:
                        pad_d = patch_size[0] - patch.shape[0]
                        pad_h = patch_size[1] - patch.shape[1]
                        pad_w = patch_size[2] - patch.shape[2]

                        patch = np.pad(patch,
                                     ((0, pad_d), (0, pad_h), (0, pad_w)),
                                     mode='constant')

                    # Predict
                    patch = torch.tensor(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                    with torch.amp.autocast('cuda'):
                        pred = model(patch)
                        pred = torch.softmax(pred, dim=1).cpu().numpy()[0]  # Apply softmax for probabilities

                    # Unpad if necessary
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        pred = pred[:, :d_end-d, :h_end-h, :w_end-w]

                    # Add to output with weight
                    output[:, d:d_end, h:h_end, w:w_end] += pred
                    weight[:, d:d_end, h:h_end, w:w_end] += 1

    # Average overlapping predictions
    output = np.divide(output, weight, where=weight!=0)
    
    # Get final class predictions
    final_prediction = np.argmax(output, axis=0)
    return final_prediction

def visualize_prediction(prediction):
    """
    Convert prediction to RGB image for visualization
    """
    D, H, W = prediction.shape
    rgb_image = np.zeros((D, H, W, 3), dtype=np.float32)
    
    for label in range(NUM_CLASSES):
        mask = prediction == label
        color = get_fragment_color(label)
        for c in range(3):
            rgb_image[mask, c] = color[c]
    
    return rgb_image

def convert_weights(checkpoint_dict, num_classes):
    """
    Convert weights from binary classification to multi-class classification
    Only converts the final layer weights, keeps other layers unchanged
    """
    new_state_dict = checkpoint_dict['model_state_dict']
    
    # Get the old output layer weights
    old_output_weight = new_state_dict['output.0.weight']  # Shape: [1, 64, 1, 1, 1]
    old_output_bias = new_state_dict['output.0.bias']      # Shape: [1]
    
    # Create new output layer weights
    new_output_weight = torch.zeros((num_classes, 64, 1, 1, 1), device=old_output_weight.device)
    new_output_bias = torch.zeros(num_classes, device=old_output_bias.device)
    
    # Initialize with old weights (for the first class)
    new_output_weight[0] = old_output_weight[0]
    new_output_bias[0] = old_output_bias[0]
    
    # Random initialize other classes (optional)
    if num_classes > 1:
        nn.init.xavier_uniform_(new_output_weight[1:])
        nn.init.zeros_(new_output_bias[1:])
    
    # Update the state dict
    new_state_dict['output.0.weight'] = new_output_weight
    new_state_dict['output.0.bias'] = new_output_bias
    
    # Initialize BatchNorm parameters for multi-class
    new_state_dict['output.1.weight'] = torch.ones(num_classes)
    new_state_dict['output.1.bias'] = torch.zeros(num_classes)
    new_state_dict['output.1.running_mean'] = torch.zeros(num_classes)
    new_state_dict['output.1.running_var'] = torch.ones(num_classes)
    
    checkpoint_dict['model_state_dict'] = new_state_dict
    return checkpoint_dict

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Convert weights if necessary
    if 'model_state_dict' in checkpoint:
        output_shape = checkpoint['model_state_dict']['output.0.weight'].shape
        if output_shape[0] != NUM_CLASSES:
            print(f"Converting weights from {output_shape[0]} classes to {NUM_CLASSES} classes...")
            checkpoint = convert_weights(checkpoint, NUM_CLASSES)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    
    # 重新计算remaining_epochs并创建新的scheduler
    remaining_epochs = NUM_EPOCHS - start_epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # 如果checkpoint中包含scheduler状态，则恢复它
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    best_loss = checkpoint['loss'] if 'loss' in checkpoint else float('inf')
    best_dice = checkpoint['best_dice'] if 'best_dice' in checkpoint else 0
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Best loss so far: {best_loss:.4f}")
    if 'best_dice' in checkpoint:
        print(f"Best dice score so far: {best_dice:.4f}")
    
    return start_epoch, best_loss, best_dice, scheduler

def train_model(model, train_loader, val_image_path, val_label_path, criterion, optimizer, 
                num_epochs, device, monitor, resume_from=None):
    best_loss = float('inf')
    best_dice = 0
    start_epoch = 0
    # scheduler 已在外部初始化
    scaler = torch.amp.GradScaler('cuda')
    total_steps = len(train_loader) * num_epochs
    current_step = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(pbar):
            torch.cuda.empty_cache()
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                current_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': loss.item() * ACCUMULATION_STEPS,
                'lr': f'{current_lr:.6f}'
            })

            running_loss += loss.item() * ACCUMULATION_STEPS
            del images, masks, outputs
            torch.cuda.empty_cache()

        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        monitor.update(avg_loss, current_lr)
        # 在每个epoch结束后用avg_loss调整学习率
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:  # 改为每20个epoch验证一次
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                dice_score = validate(model, val_image_path, val_label_path, patch_size=PATCH_SIZE)
            print(f"Validation Dice score: {dice_score:.4f}")
            model.train()

            if dice_score > best_dice:
                best_dice = dice_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'dice_score': dice_score,
                }, best_model_dice_path)
                print(f"Saved new best model with Dice score: {dice_score:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_loss_path)
            print(f"Saved new best model with loss: {best_loss:.4f}")

        # 每20个epoch保存检查点
        if (epoch + 1) % 20 == 0:  # 同样改为每20个epoch
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch,
                'loss': avg_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dice_score': dice_score if 'dice_score' in locals() else None,
                'best_dice': best_dice,
                'best_loss': best_loss
            }, checkpoint_path)

            monitor.plot_progress()
            monitor.print_stats()
            
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

# Split your data into training and validation sets
all_image_files = sorted(os.listdir(images_path))
all_label_files = sorted(os.listdir(labels_path))

# Use last image for validation
val_image_path = os.path.join(images_path, all_image_files[-1])
val_label_path = os.path.join(labels_path, all_label_files[-1])

# Start training
print("Starting fresh training for multi-class segmentation...")
print(f"Number of classes: {NUM_CLASSES} (0=background, 1-10=sacrum, 11-20=left hip, 21-30=right hip)")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Using device: {device}")

# 从头开始训练，不使用checkpoint
train_model(model, train_loader, val_image_path, val_label_path,
           criterion, optimizer, NUM_EPOCHS, device, monitor, resume_from=None)

# Prediction and Visualization after training
def visualize_single_slice(image_path, label_path, model_path, save_dir, slice_idx=None):
    """
    可视化单个切片的对比图：原始图像、真实标签、预测结果
    Args:
        image_path: CT图像路径
        label_path: 标签图像路径
        model_path: 模型路径
        save_dir: 保存目录
        slice_idx: 指定切片索引，如果为None则使用中间切片
    """
    # 加载图像和标签
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int64)
    
    # 预处理图像
    image_normalized = np.clip(image, -1000, 1000)
    image_normalized = (image_normalized + 1000) / 2000
    
    # 加载模型和预测
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    prediction = predict_volume(model, image_normalized)
    
    # 选择切片
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    # 创建图像
    plt.figure(figsize=(15, 5))
    
    # 原始CT图像
    plt.subplot(131)
    plt.imshow(image[slice_idx], cmap='gray')
    plt.title('Original CT')
    plt.axis('off')
    
    # 真实标签
    plt.subplot(132)
    plt.imshow(label[slice_idx], cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(133)
    plt.imshow(prediction[slice_idx], cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Prediction')
    plt.axis('off')
    
    # 添加颜色条
    plt.colorbar(plt.cm.ScalarMappable(cmap='tab20', 
                                      norm=plt.Normalize(vmin=0, vmax=NUM_CLASSES-1)),
                 ax=plt.gca())
    
    # 保存图像
    save_path = os.path.join(save_dir, f'slice_{slice_idx}_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparison saved to: {save_path}")
    return slice_idx

# 使用函数
slice_idx = visualize_single_slice(
    image_path=val_image_path,
    label_path=val_label_path,
    model_path=best_model_loss_path,  # 或者使用 best_model_dice_path
    save_dir=os.path.join(PATHS['results'], 'predictions')
)

def calculate_class_weights(dataset):
    """计算每个类别的权重"""
    class_counts = torch.zeros(NUM_CLASSES)
    for _, label in dataset:
        for i in range(NUM_CLASSES):
            class_counts[i] += (label == i).sum()
    
    # 使用逆频率作为权重
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts * NUM_CLASSES)
    # 将背景类的权重降低
    class_weights[0] *= 0.5
    return class_weights.to(device)

def visualize_comparison_three_views(image_path, label_path, model_path, save_dir):
    """显示三个视角的对比图"""
    # 加载数据
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int64)
    
    # 预处理
    image_normalized = np.clip(image, -1000, 1000)
    image_normalized = (image_normalized + 1000) / 2000
    
    # 加载模型和预测
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    prediction = predict_volume(model, image_normalized)
    
    # 获取中间切片索引
    D, H, W = image.shape
    mid_d, mid_h, mid_w = D//2, H//2, W//2
    
    # 创建三行三列的图像布局
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    views = ['Axial', 'Coronal', 'Sagittal']
    
    for i, (view, slice_idx) in enumerate(zip(views, [mid_d, mid_h, mid_w])):
        # 原始图像
        if view == 'Axial':
            img_slice = image[slice_idx]
            lbl_slice = label[slice_idx]
            pred_slice = prediction[slice_idx]
        elif view == 'Coronal':
            img_slice = image[:, slice_idx, :]
            lbl_slice = label[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
        else:  # Sagittal
            img_slice = image[:, :, slice_idx]
            lbl_slice = label[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
            
        # 显示原始图像
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'{view} View - Original')
        axes[i, 0].axis('off')
        
        # 显示真实标签
        axes[i, 1].imshow(lbl_slice, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 1].set_title(f'{view} View - Ground Truth')
        axes[i, 1].axis('off')
        
        # 显示预测结果
        axes[i, 2].imshow(pred_slice, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 2].set_title(f'{view} View - Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'three_views_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparison saved to: {save_path}")

def visualize_three_views_prediction(image_path, model_path, save_dir, title=None):
    """
    展示三视图（Sagittal, Coronal, Axial）的预测结果。
    Args:
        image_path: CT图像路径
        model_path: 模型权重路径
        save_dir: 保存目录
        title: 图像标题
    """
    import os
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
    image_normalized = np.clip(image, -1000, 1000)
    image_normalized = (image_normalized + 1000) / 2000

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    prediction = predict_volume(model, image_normalized)

    D, H, W = prediction.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    plt.figure(figsize=(18, 6))
    if title is not None:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f"Prediction using {os.path.basename(model_path)}", fontsize=16)

    # Sagittal
    plt.subplot(1, 3, 1)
    plt.imshow(prediction[:, :, mid_w], cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Sagittal View')
    plt.axis('off')

    # Coronal
    plt.subplot(1, 3, 2)
    plt.imshow(prediction[:, mid_h, :], cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Coronal View')
    plt.axis('off')

    # Axial
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[mid_d, :, :], cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Axial View')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'three_views_prediction.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Three-view prediction saved to: {save_path}")
    return save_path

# 在训练结束后添加三视图预测展示
visualize_three_views_prediction(
    image_path=val_image_path,
    model_path=best_model_dice_path,  # 或者best_model_loss_path
    save_dir=os.path.join(PATHS['results'], 'predictions'),
    title=None
)