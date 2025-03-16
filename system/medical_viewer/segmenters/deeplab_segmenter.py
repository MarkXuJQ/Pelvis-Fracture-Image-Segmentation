import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap
import cv2
import traceback

# 添加项目根目录到Python路径 - 修复导入问题
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 从评估代码直接复制的模型定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不同，添加1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 创建4个残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 不同膨胀率的空洞卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # 全局平均池化分支
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        # 输出卷积
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        size = x.size()[2:]
        
        # 1x1卷积分支
        x1 = F.relu(self.bn1(self.conv1(x)))
        
        # 空洞卷积分支
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        
        # 池化分支
        x5 = self.pool(x)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        
        # 合并所有分支
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        
        # 最终1x1卷积
        out = F.relu(self.bn_out(self.conv_out(x_cat)))
        
        return out

class SimpleDeepLabV3(nn.Module):
    def __init__(self, in_channels=1, num_classes=24):
        super().__init__()
        self.backbone = SimpleBackbone(in_channels)
        self.aspp = ASPP(512, 256)
        
        # 简单解码器 - 匹配评估代码的结构
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # 上采样到原始图像大小
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        input_size = x.shape[-2:]  # 记录输入尺寸
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        
        # 确保上采样到与输入完全相同的尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

# Lightning模块包装类 - 匹配评估代码
class CTSegmentationModule(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {'num_classes': 24}
        
        # 使用与评估代码相同的模型结构
        self.model = SimpleDeepLabV3(in_channels=1, num_classes=config['num_classes'])
        
    def forward(self, x):
        return self.model(x)

# 定义颜色映射，从评估代码复制
class_colors = [
    [0, 0, 0],       # 0: 背景 (黑色)
    [255, 0, 0],     # 1: 骨盆 (红色)
    [0, 255, 0],     # 2: 股骨头 (绿色)
    [0, 0, 255],     # 3: 股骨颈 (蓝色)
    [255, 255, 0],   # 4: 股骨干 (黄色)
    # 其他类别...
]

# 类别名称，从评估代码复制
class_names = ["背景", "骨盆", "股骨头", "股骨颈", "股骨干"]

# DeepLabV3分割器
class DeeplabV3Segmenter:
    """与评估代码完全匹配的DeepLabV3分割器"""
    
    def __init__(self, weights_path=None, device=None, output_dir=None):
        """初始化 DeepLabV3 分割器"""
        print("==== 初始化DeepLabV3分割器 ====")
        
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置输出目录
        self.output_dir = output_dir if output_dir is not None else os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建模型
        try:
            # 创建模型，24类分割
            self.model = self._create_model(num_classes=24)
            print(f"模型创建成功 - 类型: {type(self.model)}")
            
            # 尝试加载权重
            if weights_path is not None and os.path.exists(weights_path):
                print(f"尝试加载权重: {weights_path}")
                print(f"权重文件存在: {os.path.exists(weights_path)}")
                
                try:
                    # 尝试加载模型
                    checkpoint = torch.load(weights_path, map_location=self.device)
                    
                    # 检查加载的内容
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # 加载模型状态
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"成功加载模型权重 (带state_dict键)")
                    elif isinstance(checkpoint, dict):
                        # 直接尝试加载整个字典
                        try:
                            self.model.load_state_dict(checkpoint)
                            print(f"成功加载模型权重 (直接状态字典)")
                        except Exception as e:
                            print(f"使用状态字典加载失败: {e}")
                            # 尝试修复模块名不匹配的问题
                            fixed_state_dict = {}
                            for k, v in checkpoint.items():
                                # 移除可能导致问题的前缀
                                if k.startswith('deeplabV3.') or k.startswith('module.'):
                                    fixed_k = k.split('.', 1)[1]
                                else:
                                    fixed_k = k
                                fixed_state_dict[fixed_k] = v
                            
                            try:
                                self.model.load_state_dict(fixed_state_dict)
                                print(f"使用修复的状态字典成功加载模型权重")
                            except Exception as e2:
                                print(f"使用修复的状态字典加载失败: {e2}")
                                raise
                except Exception as e:
                    print(f"加载模型时出错: {e}")
                    traceback.print_exc()
                    print("使用未训练的默认模型")
            else:
                print(f"未提供权重路径或文件不存在，使用未初始化模型")
            
            # 将模型移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"模型已准备好进行推理，设备: {self.device}")
        except Exception as e:
            print(f"初始化模型时出错: {e}")
            traceback.print_exc()
            # 创建一个伪模型进行占位
            self.model = None
    
    def preprocess(self, image):
        """完全按照评估代码的预处理方法"""
        # 确保图像是2D或3D的
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError(f"不支持的图像形状: {image.shape}")
        
        # 处理3D图像：取中间切片
        original_shape = image.shape
        if len(image.shape) == 3:
            slice_idx = image.shape[0] // 2
            image = image[slice_idx]
        
        # 保存原始图像作为调试
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title('原始输入图像')
        plt.savefig(os.path.join(self.output_dir, 'original_input.png'))
        plt.close()
        
        # 图像预处理：调整大小到256x256
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = resize(image, (256, 256), order=1, preserve_range=True)
        
        # 归一化到[0,1]区间
        if image.max() > 1.0:
            image = image / 255.0
        
        # 保存预处理后的图像作为调试
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title('预处理后图像 (256x256)')
        plt.savefig(os.path.join(self.output_dir, 'preprocessed_input.png'))
        plt.close()
        
        # 转换为张量，添加批次和通道维度
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        return image_tensor, original_shape
    
    def create_colored_mask(self, pred):
        """创建彩色掩码，准确匹配评估代码的可视化方式"""
        # 创建彩色掩码
        colored_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        
        # 为每个类别应用颜色
        for i in range(min(len(class_colors), self.num_classes)):
            colored_mask[pred == i] = class_colors[i]
            
        return colored_mask
    
    def segment(self, image, points=None, point_labels=None, box=None, raw_output=False):
        """对图像进行分割
        
        参数:
            image: 输入图像
            points: 点标记 (DeepLabV3 不使用)
            point_labels: 点标记的标签 (DeepLabV3 不使用)
            box: 边界框 (DeepLabV3 不使用)
            raw_output: 是否返回原始多类别输出
            
        返回:
            分割掩码
        """
        try:
            print("==== 开始DeepLabV3分割 ====")
            
            # 检查模型是否成功初始化
            if self.model is None:
                print("错误: DeepLabV3模型未成功初始化，无法执行分割")
                QMessageBox.critical(None, "模型错误", 
                    "DeepLabV3模型未能正确加载。请检查权重文件是否存在且格式正确。")
                # 返回空掩码
                if raw_output:
                    # 对于多类别输出，返回全0的具有多个通道的掩码
                    h, w = image.shape[:2]
                    return np.zeros((h, w), dtype=np.uint8)
                else:
                    # 对于二值掩码，返回全0的掩码
                    return np.zeros_like(image, dtype=np.uint8)
            
            # 预处理图像
            h, w = image.shape[:2]
            
            # 转换图像到PyTorch张量
            img_tensor = self._preprocess_image(image)
            print(f"预处理后张量形状: {img_tensor.shape}")
            
            # 执行推理
            with torch.no_grad():
                self.model.eval()
                output = self.model(img_tensor)
                
                if isinstance(output, dict):
                    output = output['out']
                elif isinstance(output, list) or isinstance(output, tuple):
                    output = output[0]
            
            # 获取预测结果
            if raw_output:
                # 返回完整的多类别预测
                pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                print(f"多类别分割完成，类别范围: {np.min(pred)} - {np.max(pred)}")
                print(f"分割中的唯一类别: {np.unique(pred)}")
                
                # 验证模型是否有效地分割了图像
                if np.max(pred) == 0:
                    print("警告: 分割结果只包含背景 (类别0)")
                    
                    # 创建一个样例分割结果用于调试
                    if image is not None:
                        test_mask = np.zeros_like(image, dtype=np.uint8)
                        h, w = image.shape[:2]
                        # 创建一些简单的几何图形作为测试掩码
                        cv2.circle(test_mask, (w//2, h//2), min(h,w)//4, 1, -1)
                        cv2.rectangle(test_mask, (w//4, h//4), (3*w//4, 3*h//4), 2, 2)
                        print("已创建测试掩码进行调试")
                        
                        # 仅在调试模式下返回测试掩码
                        debug_mode = False  # 设置为True以返回测试掩码
                        if debug_mode:
                            return test_mask
                
                return pred
            else:
                # 返回二值掩码（非背景为前景）
                pred = output.argmax(1).squeeze(0).cpu().numpy()
                binary_mask = (pred > 0).astype(np.uint8)
                print(f"二值分割完成，前景像素数: {np.sum(binary_mask)}")
                return binary_mask
            
        except Exception as e:
            print(f"分割时出错: {e}")
            traceback.print_exc()
            
            # 返回空掩码
            return np.zeros_like(image, dtype=np.uint8)

    def get_probability_map(self, image, points=None, box=None, boxes=None, prompt_type=None):
        """获取概率图"""
        try:
            # 预处理输入
            input_tensor, _ = self.preprocess(image)
            
            # 添加到设备
            input_tensor = input_tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                
                # 使用softmax获取类别概率
                probs = F.softmax(outputs, dim=1)
                
                # 获取所有非背景类的概率总和
                foreground_prob = torch.zeros_like(probs[:, 0])
                for cls in range(1, self.num_classes):
                    foreground_prob += probs[:, cls]
                
                # 保存前景概率图
                fg_prob_np = foreground_prob.squeeze().cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(fg_prob_np, cmap='jet', vmin=0, vmax=1)
                plt.colorbar()
                plt.title('前景概率图(所有非背景类)')
                plt.savefig(os.path.join(self.output_dir, 'foreground_probability.png'))
                plt.close()
            
            # 返回前景概率图
            return foreground_prob.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"获取概率图时出错: {str(e)}")
            if len(image.shape) == 2:
                return np.zeros(image.shape, dtype=np.float32)
            else:
                return np.zeros(image.shape[1:] if len(image.shape) > 2 else image.shape, dtype=np.float32)

    def _create_model(self, num_classes=24):
        """创建DeepLabV3模型"""
        try:
            # 尝试使用torchvision的实现
            print("尝试加载torchvision的DeepLabV3模型")
            
            # 通道检查 - 我们使用的是医学图像，通常是单通道
            in_channels = 1  # 默认为单通道灰度图像
            
            try:
                # 导入相关模块
                from torchvision.models.segmentation import deeplabv3_resnet50
                
                # 创建具有指定输入通道的模型
                model = deeplabv3_resnet50(
                    weights=None,  # 不使用预训练权重
                    num_classes=num_classes
                )
                
                # 修改输入层以支持单通道图像
                if in_channels != 3:  # 默认是3通道
                    # 获取原始卷积层
                    original_conv = model.backbone.conv1
                    
                    # 创建新的卷积层，保持其他参数相同但改变输入通道数
                    new_conv = nn.Conv2d(
                        in_channels, 
                        original_conv.out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding,
                        bias=(original_conv.bias is not None)
                    )
                    
                    # 初始化新层的权重
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)
                        
                    # 替换卷积层
                    model.backbone.conv1 = new_conv
                    
                    print(f"已将输入通道从3修改为{in_channels}")
                
                print("成功创建torchvision DeepLabV3模型")
                return model
                
            except ImportError as e:
                print(f"无法导入torchvision DeepLabV3模型: {e}")
                # 尝试备选方案
            
            # 如果上面失败，尝试使用自定义实现
            print("尝试使用自定义DeepLabV3实现")
            
            # 这里可以添加自定义的DeepLabV3实现代码
            # ...
            
            # 如果所有尝试都失败，抛出异常
            raise ImportError("无法创建DeepLabV3模型 - 请确保安装了正确的依赖")
            
        except Exception as e:
            print(f"创建模型时发生错误: {e}")
            traceback.print_exc()
            
            # 在模型创建失败时，显示一个警告但不终止程序
            QMessageBox.warning(None, "模型加载警告", 
                f"无法创建DeepLabV3模型: {str(e)}\n程序将继续运行，但分割功能可能不可用。")
            
            # 返回None表示模型创建失败
            return None