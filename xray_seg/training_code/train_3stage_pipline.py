import os
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------
# 数据集定义
# ----------------------
class XRayPelvisDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = tiff.imread(self.image_paths[idx]).astype(np.float32)
        label = tiff.imread(self.label_paths[idx]).astype(np.uint32)
        # 归一化
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        # 缩放到512x512
        image = np.resize(image, (512, 512))
        label = np.resize(label, (512, 512))
        # 三通道标签：1=骶骨, 2=左髋骨, 3=右髋骨
        label_3ch = np.zeros((3, 512, 512), dtype=np.float32)
        label_3ch[0] = (np.isin(label, np.arange(1, 11))).astype(np.float32)  # 骶骨
        label_3ch[1] = (np.isin(label, np.arange(11, 21))).astype(np.float32) # 左髋骨
        label_3ch[2] = (np.isin(label, np.arange(21, 31))).astype(np.float32) # 右髋骨
        image = np.expand_dims(image, axis=0)  # (1, 512, 512)
        if self.transform:
            image = self.transform(image)
            label_3ch = self.transform(label_3ch)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label_3ch, dtype=torch.float32)

class XRayFractureDataset(Dataset):
    """
    针对某一骨（骶骨/左髋骨/右髋骨）分割碎片的数据集
    mask_type: 1=骶骨, 2=左髋骨, 3=右髋骨
    """
    def __init__(self, image_paths, label_paths, mask_type, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mask_type = mask_type
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = tiff.imread(self.image_paths[idx]).astype(np.float32)
        label = tiff.imread(self.label_paths[idx]).astype(np.uint32)
        # 归一化
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        image = np.resize(image, (512, 512))
        label = np.resize(label, (512, 512))
        # 只保留该骨的mask区域
        if self.mask_type == 1:
            mask = np.isin(label, np.arange(1, 11))
            frag_label = label * mask
            frag_label = np.where(mask, label, 0)
        elif self.mask_type == 2:
            mask = np.isin(label, np.arange(11, 21))
            frag_label = label * mask
            frag_label = np.where(mask, label-10, 0)  # 变为1-10
        elif self.mask_type == 3:
            mask = np.isin(label, np.arange(21, 31))
            frag_label = label * mask
            frag_label = np.where(mask, label-20, 0)  # 变为1-10
        # 输出为单通道，类别0-10
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(frag_label, dtype=torch.long)

# ----------------------
# 2. 网络结构定义（UNet2D，DepthwiseSeparableConv2d，AttentionGate2D，SEBlock2D）
# ----------------------
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

# ----------------------
# 训练与验证函数
# ----------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = target.float()
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def train_one_epoch(model, loader, optimizer, criterion, dice_loss, device):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) + dice_loss(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, dice_loss, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) + dice_loss(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)

# ----------------------
# 主流程
# ----------------------
def main():
    # 路径
    image_dir = 'xray_seg/data/train/input/images/x-ray'
    label_dir = 'xray_seg/data/train/output/images/x-ray'
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.tif')])
    # 划分训练集和验证集
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(image_files, label_files, test_size=0.1, random_state=42)
    # 数据集
    train_dataset = XRayPelvisDataset(train_imgs, train_lbls)
    val_dataset = XRayPelvisDataset(val_imgs, val_lbls)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=4)
    # 网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(1, 3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    # 训练
    best_val_loss = float('inf')
    for epoch in range(1, 101):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, dice_loss, device)
        val_loss = validate(model, val_loader, criterion, dice_loss, device)
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('xray_seg/data/result', 'best_unet2d_stage1.pth'))
    print('Stage 1 training finished!')

    # 阶段2：分别训练三个骨的碎片分割
    train_fracture_stage(1, train_imgs, train_lbls, val_imgs, val_lbls, device)  # 骶骨
    train_fracture_stage(2, train_imgs, train_lbls, val_imgs, val_lbls, device)  # 左髋骨
    train_fracture_stage(3, train_imgs, train_lbls, val_imgs, val_lbls, device)  # 右髋骨

def train_fracture_stage(mask_type, train_imgs, train_lbls, val_imgs, val_lbls, device):
    # mask_type: 1=骶骨, 2=左髋骨, 3=右髋骨
    train_dataset = XRayFractureDataset(train_imgs, train_lbls, mask_type)
    val_dataset = XRayFractureDataset(val_imgs, val_lbls, mask_type)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    model = UNet2D(1, 11).to(device)  # 0-10类
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        # 验证
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader)
        print(f'[Fracture Mask {mask_type}] Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('xray_seg/data/result', f'best_unet2d_fracture_mask{mask_type}.pth'))
    print(f'Stage 2 (mask {mask_type}) training finished!')

def cascade_inference(image_path, device):
    # 1. 加载骨盆分割模型
    model_pelvis = UNet2D(1, 3).to(device)
    model_pelvis.load_state_dict(torch.load(os.path.join('xray_seg/data/result', 'best_unet2d_stage1.pth'), map_location=device))
    model_pelvis.eval()
    # 2. 加载骨折分割模型
    model_fracture1 = UNet2D(1, 11).to(device)
    model_fracture1.load_state_dict(torch.load(os.path.join('xray_seg/data/result', 'best_unet2d_fracture_mask1.pth'), map_location=device))
    model_fracture1.eval()
    model_fracture2 = UNet2D(1, 11).to(device)
    model_fracture2.load_state_dict(torch.load(os.path.join('xray_seg/data/result', 'best_unet2d_fracture_mask2.pth'), map_location=device))
    model_fracture2.eval()
    model_fracture3 = UNet2D(1, 11).to(device)
    model_fracture3.load_state_dict(torch.load(os.path.join('xray_seg/data/result', 'best_unet2d_fracture_mask3.pth'), map_location=device))
    model_fracture3.eval()
    # 3. 读取并预处理图像
    image = tiff.imread(image_path).astype(np.float32)
    image = (image - np.mean(image)) / (np.std(image) + 1e-8)
    image = np.resize(image, (256, 256))
    image_tensor = torch.tensor(image[None, None, :, :], dtype=torch.float32).to(device)
    # 4. 骨盆分割
    with torch.no_grad():
        pelvis_pred = torch.sigmoid(model_pelvis(image_tensor)).cpu().numpy()[0]
    # 5. 骨折分割（对每个骨mask区域分别推理）
    final_mask = np.zeros((256, 256), dtype=np.uint32)
    for i, (fracture_model, offset) in enumerate(zip(
        [model_fracture1, model_fracture2, model_fracture3], [0, 10, 20])):
        mask = pelvis_pred[i] > 0.5
        if mask.sum() == 0:
            continue
        input_tensor = image_tensor.clone()
        with torch.no_grad():
            frag_pred = torch.argmax(fracture_model(input_tensor), dim=1).cpu().numpy()[0]
        frag_pred = frag_pred * mask
        frag_pred[frag_pred > 0] += offset
        final_mask[mask] = frag_pred[mask]
    tiff.imwrite(os.path.join('xray_seg/data/result', f'result_{os.path.basename(image_path)}'), final_mask.astype(np.uint32))
    return final_mask

def relabel_pelvis(label):
    new_label = np.zeros_like(label, dtype=np.int64)
    new_label[np.isin(label, np.arange(1,11))] = 1
    new_label[np.isin(label, np.arange(11,21))] = 2
    new_label[np.isin(label, np.arange(21,31))] = 3
    return new_label

if __name__ == '__main__':
    main()