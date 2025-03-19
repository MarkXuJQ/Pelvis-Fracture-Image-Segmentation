import os
import torch
import numpy as np
import SimpleITK as sitk
import traceback
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord, CropForegroundd, ScaleIntensityRangePercentilesd, Resized,
    EnsureType, ScaleIntensityd, NormalizeIntensityd
)
from monai.data import ITKReader
from PyQt5.QtWidgets import QMessageBox
import time
from scipy.ndimage import zoom
import nibabel as nib

class UNet3DSegmenter:
    """使用UNETR模型的3D分割器"""
    
    def __init__(self, weights_path=None, device=None):
        """初始化UNet3D分割器
        
        参数:
            weights_path: 模型权重文件路径
            device: 运行设备 (CPU或CUDA)
        """
        print("==== 初始化UNETR分割器 ====")
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"使用设备: {self.device}")
        
        # 设置标签映射 - 与训练代码保持一致
        self.original_labels = [0, 1, 2, 11, 12, 13, 21, 22, 23]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.original_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 设置配置与训练代码一致
        self.config = {
            'num_classes': len(self.original_labels),  # 9类(包括背景)
            'patch_size': (64, 64, 64),  # 与训练代码一致，不再是96³
            'spatial_dims': 3
        }
        
        # 创建模型 - 参数与训练代码完全一致
        self.model = UNETR(
            in_channels=1,
            out_channels=self.config['num_classes'],
            img_size=self.config['patch_size'],  # 使用64³
            feature_size=16,
            hidden_size=384,  # 与训练代码一致
            mlp_dim=1536,    # 与训练代码一致
            num_heads=6,     # 与训练代码一致
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(self.device)
        
        # 加载预训练权重
        if weights_path and os.path.exists(weights_path):
            print(f"加载权重: {weights_path}")
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                
                # 尝试不同的加载方式以适应不同的checkpoint格式
                if "model" in checkpoint:
                    # 如果checkpoint中有model键
                    self.model.load_state_dict(checkpoint["model"])
                elif "model_state_dict" in checkpoint:
                    # 如果checkpoint中有model_state_dict键(训练代码格式)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    # 如果checkpoint中有state_dict键
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "module.conv_0.weight" in checkpoint:
                    # 如果是DataParallel保存的权重
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint.items():
                        name = k.replace("module.", "") if k.startswith("module.") else k
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                else:
                    # 直接尝试加载，适用于仅包含模型权重的checkpoint
                    self.model.load_state_dict(checkpoint)
                    
                print("权重加载成功")
                
            except Exception as e:
                print(f"加载权重失败: {str(e)}")
                traceback.print_exc()
        else:
            print("警告: 未提供有效权重路径，使用未初始化模型")
        
        # 设置评估模式
        self.model.eval()
        
        # 设置预处理和后处理转换
        self.transform = self._create_transforms()
        
        # 设置输出目录
        self.output_dir = os.path.join(os.path.dirname(__file__), 'unetr_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_transforms(self):
        """创建数据预处理和后处理转换"""
        transforms = Compose([
            EnsureType(data_type="tensor"),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Resized(keys=["image"], spatial_size=(96, 96, 96)),
        ])
        return transforms

    def _preprocess_volume(self, image):
        """预处理3D图像体积，与训练代码保持一致
        
        参数:
            image: numpy数组，3D图像数据
            
        返回:
            预处理后的张量，适合输入模型
        """
        try:
            print("预处理3D图像...")
            
            # 检查图像
            if image is None:
                print("错误: 输入图像为空")
                return None
            
            # 获取输入图像信息
            print(f"输入图像形状: {image.shape}, 类型: {image.dtype}")
            
            # 转换为float32类型
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # 与训练代码一致的预处理 - ScaleIntensityRanged
            # 训练代码使用 a_min=-175, a_max=250, b_min=0.0, b_max=1.0
            min_val, max_val = -175, 250
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val)
            
            # 添加通道和批次维度 [B,C,H,W,D]
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            
            # 移动到正确的设备
            image_tensor = image_tensor.to(self.device)
            
            print(f"预处理后的张量形状: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            print(f"预处理图像时出错: {str(e)}")
            traceback.print_exc()
            return None

    def segment(self, image, **kwargs):
        """
        执行3D图像分割
        
        参数:
            image: 3D图像数据 (numpy数组)
            
        返回:
            分割掩码
        """
        try:
            print("执行UNET 3D分割...")
            
            # 定义开始时间
            start_time = time.time()
            
            # 检查输入图像
            if image is None or not isinstance(image, np.ndarray):
                print("错误: 输入必须是有效的numpy数组")
                return None
                
            print(f"输入图像形状: {image.shape}")
            
            # 预处理图像
            processed_data = self._preprocess_volume(image)
            if processed_data is None:
                return None
                
            # 记录设备信息
            print(f"模型设备: {next(self.model.parameters()).device}")
            
            # 推理 - 使用与训练代码相同的参数
            with torch.no_grad():
                outputs = sliding_window_inference(
                    inputs=processed_data, 
                    roi_size=self.config['patch_size'],  # 使用(64,64,64)
                    sw_batch_size=1, 
                    predictor=self.model,
                    overlap=0.5  # 与训练代码一致
                )
                
            print(f"推理完成，输出形状: {outputs.shape}")
            
            # 后处理 - 获取类别索引
            mask = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            mask = mask[0]  # 移除批次维度
            
            # 还原为原始标签值 - 使用标签映射
            result_mask = np.zeros_like(mask, dtype=np.uint8)
            for idx, label in self.idx_to_label.items():
                result_mask[mask == idx] = label
            
            # 如果需要调整掩码的形状以匹配原始图像
            if result_mask.shape != image.shape:
                print(f"掩码形状 {result_mask.shape} 与原始图像形状 {image.shape} 不匹配")
                # 如果需要调整尺寸，可以在这里添加代码
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"分割完成，耗时 {elapsed:.2f} 秒")
            
            return result_mask
            
        except Exception as e:
            print(f"分割时出错: {str(e)}")
            traceback.print_exc()
            return None

    def _postprocess(self, output):
        """后处理输出"""
        # 转换为概率图
        probs = torch.softmax(output, dim=1)
        
        # 获取最终分割结果
        segmentation = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # 将索引映射回原始标签值
        result = np.zeros_like(segmentation, dtype=np.uint8)
        for idx, label in self.idx_to_label.items():
            result[segmentation == idx] = label
        
        return result

    def save_mask(self, mask, file_path):
        """保存分割结果"""
        try:
            sitk_mask = sitk.GetImageFromArray(mask)
            
            # 复制原始元数据
            sitk_mask.CopyInformation(self.original_image)
            
            sitk.WriteImage(sitk_mask, file_path)
            print(f"3D分割结果已保存至: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            traceback.print_exc()
            return False

    def get_probability_map(self, image):
        """获取3D概率图"""
        try:
            # 预处理图像
            processed_data = self._preprocess_volume(image)
            if processed_data is None:
                return None
            
            # 推理
            with torch.no_grad():
                outputs = sliding_window_inference(
                    inputs=processed_data, 
                    roi_size=self.config['patch_size'],
                    sw_batch_size=1, 
                    predictor=self.model,
                    overlap=0.5
                )
            
            # 转换为概率(使用softmax)
            probs = torch.softmax(outputs, dim=1)
            
            # 转换为numpy数组并返回
            return probs.cpu().numpy()[0]  # 移除批次维度
            
        except Exception as e:
            print(f"获取概率图时出错: {str(e)}")
            traceback.print_exc()
            return None 