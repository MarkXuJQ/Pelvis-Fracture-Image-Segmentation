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

# 导入分割模型
from system.medsam_segmenter import MedSAMSegmenter

class MedicalImageProcessor:
    """医学图像处理类，提供加载、显示和基础处理功能"""
    
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
            
            plt.imshow(self.image_data[slice_index], cmap='gray')
            plt.title(f"3D图像切片 #{slice_index}")
        else:
            plt.imshow(self.image_data, cmap='gray' if len(self.image_data.shape) == 2 else None)
            plt.title("2D图像")
            
        plt.colorbar()
        plt.show()
    
    def save_mask(self, mask: np.ndarray, file_path: str) -> bool:
        """
        保存分割掩码
        
        参数:
            mask: 分割掩码数组
            file_path: 保存路径
            
        返回:
            bool: 成功保存返回True，否则返回False
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.nii.gz' or file_ext == '.nii':
                # 保存为NIFTI格式
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                
                # 如果原图是3D图像，复制原图的元数据
                if self.is_3d and 'spacing' in self.metadata:
                    mask_sitk.SetSpacing(self.metadata['spacing'])
                if self.is_3d and 'origin' in self.metadata:
                    mask_sitk.SetOrigin(self.metadata['origin'])
                if self.is_3d and 'direction' in self.metadata:
                    mask_sitk.SetDirection(self.metadata['direction'])
                    
                sitk.WriteImage(mask_sitk, file_path)
                
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                # 保存为图像格式
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(file_path, mask_uint8)
                
            else:
                print(f"不支持的保存格式: {file_ext}")
                return False
                
            print(f"掩码已保存到: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存掩码时出错: {str(e)}")
            return False
    
    def set_segmentation_model(self, model_name, checkpoint_path):
        """
        设置分割模型
        
        参数:
            model_name: 模型类型 ('medsam')
            checkpoint_path: 权重文件路径
        """
        if model_name == 'medsam':
            self.segmenter = MedSAMSegmenter(
                model_type='vit_b',
                checkpoint_path=checkpoint_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"MedSAM分割器已设置，权重文件: {checkpoint_path}")
            return True
        else:
            print(f"不支持的模型类型: {model_name}")
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
            'class': MedSAMSegmenter
        }
    }
    
    # 检查权重文件是否存在
    for name, info in models.items():
        if os.path.exists(info['weights_path']):
            info['status'] = '已安装'
        else:
            info['status'] = '未安装'
    
    return models 