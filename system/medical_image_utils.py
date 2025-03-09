import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import torch
from typing import Union, Tuple, Optional, List
import cv2
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的分割模型
from system.models import MedSamSegmenter, DeepLabV3Segmenter, SegmentationAPI


class MedicalImageProcessor:
    """医学图像处理类，提供加载、显示和分割功能"""
    
    SUPPORTED_3D_FORMATS = ['.mha', '.nii', '.nii.gz']
    SUPPORTED_2D_FORMATS = ['.tif', '.jpg', '.png', '.bmp', '.jpeg']
    
    def __init__(self):
        self.image_data = None
        self.image_path = None
        self.is_3d = False
        self.metadata = {}
        self.segmenter = None
        
    def load_image(self, image_path: str) -> bool:
        """
        加载医学图像文件
        
        参数:
            image_path: 图像文件路径
            
        返回:
            bool: 成功加载返回True，否则返回False
        """
        if not os.path.exists(image_path):
            print(f"错误: 文件 {image_path} 不存在")
            return False
        
        file_ext = os.path.splitext(image_path)[1].lower()
        
        try:
            # 处理3D图像格式
            if file_ext in self.SUPPORTED_3D_FORMATS:
                self._load_3d_image(image_path)
                
            # 处理2D图像格式
            elif file_ext in self.SUPPORTED_2D_FORMATS:
                self._load_2d_image(image_path)
                
            else:
                print(f"错误: 不支持的文件格式 {file_ext}")
                return False
                
            self.image_path = image_path
            return True
            
        except Exception as e:
            print(f"加载图像时出错: {str(e)}")
            return False
    
    def _load_3d_image(self, image_path: str) -> None:
        """加载3D医学图像"""
        # 使用SimpleITK加载3D图像
        sitk_image = sitk.ReadImage(image_path)
        
        # 保存元数据
        for key in sitk_image.GetMetaDataKeys():
            self.metadata[key] = sitk_image.GetMetaData(key)
        
        # 获取其他属性
        self.metadata['spacing'] = sitk_image.GetSpacing()
        self.metadata['origin'] = sitk_image.GetOrigin()
        self.metadata['direction'] = sitk_image.GetDirection()
        
        # 转换为numpy数组
        self.image_data = sitk.GetArrayFromImage(sitk_image)
        self.is_3d = True
        
        print(f"已加载3D图像: 形状={self.image_data.shape}")
    
    def _load_2d_image(self, image_path: str) -> None:
        """加载2D医学图像"""
        # 使用PIL加载2D图像
        pil_image = Image.open(image_path)
        
        # 保存元数据
        self.metadata['format'] = pil_image.format
        self.metadata['mode'] = pil_image.mode
        self.metadata['size'] = pil_image.size
        
        # 转换为numpy数组
        self.image_data = np.array(pil_image)
        self.is_3d = False
        
        print(f"已加载2D图像: 形状={self.image_data.shape}")
    
    def display_image(self, slice_index: int = None) -> None:
        """
        显示医学图像
        
        参数:
            slice_index: 3D图像的切片索引，如果为None则显示中间切片
        """
        if self.image_data is None:
            print("错误: 没有加载图像")
            return
        
        plt.figure(figsize=(10, 8))
        
        if self.is_3d:
            if slice_index is None:
                slice_index = self.image_data.shape[0] // 2  # 默认显示中间切片
            
            # 确保slice_index在有效范围内
            if slice_index < 0 or slice_index >= self.image_data.shape[0]:
                print(f"错误: 切片索引 {slice_index} 超出范围 (0-{self.image_data.shape[0]-1})")
                return
            
            plt.imshow(self.image_data[slice_index], cmap='gray')
            plt.title(f"3D图像切片 #{slice_index}/{self.image_data.shape[0]-1}")
            
        else:
            # 对于2D图像，直接显示
            if len(self.image_data.shape) == 3 and self.image_data.shape[2] == 3:  # RGB图像
                plt.imshow(self.image_data)
            else:  # 灰度图像
                plt.imshow(self.image_data, cmap='gray')
            
            plt.title("2D图像")
        
        plt.colorbar(label='像素值')
        plt.axis('on')
        plt.tight_layout()
        plt.show()
    
    def display_3d_slices(self, num_slices: int = 4) -> None:
        """
        显示3D图像的多个切片
        
        参数:
            num_slices: 要显示的切片数量
        """
        if not self.is_3d or self.image_data is None:
            print("错误: 没有加载3D图像")
            return
        
        total_slices = self.image_data.shape[0]
        indices = np.linspace(0, total_slices-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(1, num_slices, figsize=(16, 4))
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.image_data[idx], cmap='gray')
            axes[i].set_title(f"切片 #{idx}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def set_segmentation_model(self, model_name, **kwargs):
        """
        设置分割模型
        
        参数:
            model_name: 模型类型 ('medsam', 'deeplabv3_resnet50' 或 'deeplabv3_resnet101')
            **kwargs: 模型特定的参数
        """
        try:
            models = list_available_models()
            if model_name not in models:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
            model_info = models[model_name]
            checkpoint_path = kwargs.get('checkpoint_path', model_info['weights_path'])
            
            # 检查模型文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
            
            if model_name == 'medsam':
                self.segmenter = MedSamSegmenter(
                    model_type=kwargs.get('model_type', 'vit_b'),
                    checkpoint_path=checkpoint_path,
                    device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                )
                
            elif model_name.startswith('deeplabv3'):
                # 创建模型并明确禁用预训练下载
                self.segmenter = DeepLabV3Segmenter(
                    num_classes=kwargs.get('num_classes', 21),
                    pretrained=False,
                    backbone=model_info.get('backbone', 'resnet50'),  # 使用模型信息中指定的骨干网络
                    device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                )
                
                # 加载本地权重
                print(f"正在加载自定义DeepLabV3权重: {checkpoint_path}")
                self.segmenter.load_weights(checkpoint_path)
                
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
        except Exception as e:
            print(f"设置分割模型时出错: {str(e)}")
            raise e
    
    def segment_image(self, **kwargs):
        """
        分割图像
        
        参数:
            **kwargs: 分割特定参数（例如points, point_labels, target_class等）
            
        返回:
            mask: 分割结果
        """
        if self.segmenter is None:
            raise ValueError("请先使用set_segmentation_model设置分割模型")
        
        if self.is_3d:
            print("对3D图像进行分割，每个切片单独处理")
            masks = []
            for i in range(self.image_data.shape[0]):
                # 确保图像为浮点型并且范围在0-1之间
                slice_img = self.image_data[i].astype(np.float32)
                if slice_img.max() > 1.0:
                    slice_img = slice_img / 255.0
                    
                try:
                    mask = self.segmenter.segment(slice_img, **kwargs)
                    masks.append(mask)
                except Exception as e:
                    print(f"分割切片 {i} 时出错: {e}")
                    # 填充空白掩码
                    masks.append(np.zeros_like(self.image_data[i], dtype=bool))
                
            return np.stack(masks)
        else:
            # 确保图像为浮点型并且范围在0-1之间
            img = self.image_data.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            
            return self.segmenter.segment(img, **kwargs)
    
    def display_segmentation_result(self, mask: np.ndarray, slice_index: int = None) -> None:
        """
        显示分割结果
        
        参数:
            mask: 分割掩码
            slice_index: 3D图像的切片索引，默认显示中间切片
        """
        if self.image_data is None:
            print("错误: 没有加载图像")
            return
        
        if mask is None:
            print("错误: 没有分割掩码")
            return
        
        plt.figure(figsize=(15, 5))
        
        if self.is_3d:
            if slice_index is None:
                slice_index = self.image_data.shape[0] // 2
            
            # 显示原图
            plt.subplot(1, 3, 1)
            plt.imshow(self.image_data[slice_index], cmap='gray')
            plt.title(f"原始图像 (切片 #{slice_index})")
            plt.axis('on')
            
            # 显示掩码
            plt.subplot(1, 3, 2)
            plt.imshow(mask[slice_index], cmap='viridis')
            plt.title(f"分割掩码 (切片 #{slice_index})")
            plt.axis('on')
            
            # 显示叠加图
            plt.subplot(1, 3, 3)
            plt.imshow(self.image_data[slice_index], cmap='gray')
            plt.imshow(mask[slice_index], alpha=0.5, cmap='viridis')
            plt.title("叠加图")
            plt.axis('on')
            
        else:
            # 显示原图
            plt.subplot(1, 3, 1)
            if len(self.image_data.shape) == 3 and self.image_data.shape[2] == 3:
                plt.imshow(self.image_data)
            else:
                plt.imshow(self.image_data, cmap='gray')
            plt.title("原始图像")
            plt.axis('on')
            
            # 显示掩码
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='viridis')
            plt.title("分割掩码")
            plt.axis('on')
            
            # 显示叠加图
            plt.subplot(1, 3, 3)
            if len(self.image_data.shape) == 3 and self.image_data.shape[2] == 3:
                plt.imshow(self.image_data)
            else:
                plt.imshow(self.image_data, cmap='gray')
            plt.imshow(mask, alpha=0.5, cmap='viridis')
            plt.title("叠加图")
            plt.axis('on')
        
        plt.tight_layout()
        plt.show()
    
    def save_segmentation_result(self, mask: np.ndarray, output_path: str) -> bool:
        """
        保存分割结果
        
        参数:
            mask: 分割掩码
            output_path: 输出文件路径
            
        返回:
            bool: 成功保存返回True，否则返回False
        """
        if mask is None:
            print("错误: 没有分割掩码")
            return False
        
        try:
            if self.is_3d:
                # 保存3D分割结果
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                
                # 复制原始图像的元数据（如果有）
                if 'spacing' in self.metadata:
                    mask_sitk.SetSpacing(self.metadata['spacing'])
                if 'origin' in self.metadata:
                    mask_sitk.SetOrigin(self.metadata['origin'])
                if 'direction' in self.metadata:
                    mask_sitk.SetDirection(self.metadata['direction'])
                
                sitk.WriteImage(mask_sitk, output_path)
                
            else:
                # 保存2D分割结果
                Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
            
            print(f"已成功保存分割结果到: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存分割结果时出错: {str(e)}")
            return False


def list_available_models() -> dict:
    """
    列出系统中可用的分割模型
    
    返回:
        dict: 可用模型的字典，包含模型信息
    """
    models = {
        'medsam': {
            'description': 'Medical SAM 模型 (基于SAM的医学图像分割)',
            'weights_path': 'weights/MedSAM/medsam_vit_b.pth',
            'class': MedSamSegmenter
        },
        'deeplabv3_resnet50': {
            'description': 'DeepLabV3 ResNet50 模型 (语义分割)',
            'weights_path': 'weights/DeeplabV3/best_deeplabv3_resnet50_voc_os16.pth',
            'class': DeepLabV3Segmenter,
            'backbone': 'resnet50'
        },
        'deeplabv3_resnet101': {
            'description': 'DeepLabV3 ResNet101 模型 (语义分割)',
            'weights_path': 'weights/DeeplabV3/best_deeplabv3_resnet101_voc_os16.pth',
            'class': DeepLabV3Segmenter,
            'backbone': 'resnet101'
        }
    }
    
    # 检查权重文件是否存在
    for name, info in models.items():
        if os.path.exists(info['weights_path']):
            info['status'] = '已安装'
        else:
            info['status'] = '未安装'
    
    return models


def get_model_weights_path(model_type: str) -> str:
    """
    获取指定模型类型的权重文件路径
    
    参数:
        model_type: 模型类型 ('medsam' 或 'deeplabv3')
        
    返回:
        str: 权重文件路径
    """
    models = list_available_models()
    
    if model_type in models:
        return models[model_type]['weights_path']
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 使用示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = MedicalImageProcessor()
    
    # 列出可用模型
    print("可用的分割模型:")
    for name, info in list_available_models().items():
        print(f"- {name}: {info['description']} ({info['status']})")
    
    # 加载图像示例（取消注释使用）
    # processor.load_image("path/to/image.nii")
    # processor.display_image()
    
    # 设置分割模型示例（取消注释使用）
    # processor.set_segmentation_model("medsam", checkpoint_path="weights/MedSAM/medsam_vit_b.pth")
    
    # 分割图像示例（取消注释使用）
    # mask = processor.segment_image(points=[[100, 100]], point_labels=[1])
    # processor.display_segmentation_result(mask) 