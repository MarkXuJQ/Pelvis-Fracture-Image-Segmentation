import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import torch
from typing import Union, Tuple, Optional, List
import cv2
import sys
import vtk
from vtk.util import numpy_support
import traceback

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入分割模型
from system.medical_viewer.segmenters.medsam_segmenter import MedSAMSegmenter
from system.medical_viewer.segmenters.deeplab_segmenter import DeeplabV3Segmenter
from .segmenters.unet_3d_segmenter import UNet3DSegmenter

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
        
        # 检查是否为X光图像（基于文件扩展名）
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in self.SUPPORTED_2D_FORMATS:
            self._process_xray_if_needed()
        
        print(f"已加载2D图像: 形状={self.image_data.shape}")
    
    def _process_xray_if_needed(self) -> None:
        """处理X光图像，应用适当的gamma校正"""
        # 检测是否为X光图像（基于图像特性）
        if self._is_likely_xray():
            print("检测到X光图像，应用gamma校正")
            # 应用gamma校正（典型X光gamma值约为1.5-2.2）
            gamma = 1.8
            self.apply_gamma_correction(gamma)
    
    def _is_likely_xray(self) -> bool:
        """基于图像特性判断是否为X光图像"""
        # 简单启发式方法：X光图像通常有高对比度、灰度特性
        if len(self.image_data.shape) == 2 or (len(self.image_data.shape) == 3 and self.image_data.shape[2] == 1):
            # 灰度图
            mean_val = np.mean(self.image_data)
            std_val = np.std(self.image_data)
            # X光通常有中等平均亮度和较高对比度
            return 30 < mean_val < 200 and std_val > 40
        elif len(self.image_data.shape) == 3:
            # 检查RGB图像是否近似灰度（如果是X光的RGB表示）
            r, g, b = self.image_data[:, :, 0], self.image_data[:, :, 1], self.image_data[:, :, 2]
            if np.abs(np.mean(r-g)) < 5 and np.abs(np.mean(r-b)) < 5 and np.abs(np.mean(g-b)) < 5:
                return True
        return False
    
    def apply_gamma_correction(self, gamma: float) -> None:
        """
        应用gamma校正到图像
        
        参数:
            gamma: gamma校正值，<1为增亮，>1为变暗
        """
        # 确保图像数据在0-1范围内
        if self.image_data.max() > 1.0:
            img_normalized = self.image_data / 255.0
        else:
            img_normalized = self.image_data.copy()
            
        # 应用gamma校正
        corrected = np.power(img_normalized, 1.0/gamma)
        
        # 转换回原始范围
        if self.image_data.max() > 1.0:
            self.image_data = (corrected * 255.0).astype(self.image_data.dtype)
        else:
            self.image_data = corrected
    
    def display_image(self, slice_index: int = None, use_vtk: bool = False) -> None:
        """
        显示医学图像
        
        参数:
            slice_index: 3D图像的切片索引，如果为None则显示中间切片
            use_vtk: 是否使用VTK进行渲染（推荐用于X光图像）
        """
        if self.image_data is None:
            print("错误: 没有加载图像")
            return
        
        if use_vtk:
            self._display_with_vtk(slice_index)
        else:
            # 原始matplotlib显示方法
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
            
    def _display_with_vtk(self, slice_index: int = None) -> None:
        """使用VTK显示图像，更适合X光图像显示"""
        if self.is_3d:
            if slice_index is None:
                slice_index = self.image_data.shape[0] // 2
            img_data = self.image_data[slice_index].copy()
        else:
            img_data = self.image_data.copy()
        
        # 确保图像为灰度图
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            # 转换RGB为灰度
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        
        # 确保数据类型正确
        if img_data.dtype != np.uint8:
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)
        
        # 创建VTK图像数据
        height, width = img_data.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        # 填充VTK图像数据
        for y in range(height):
            for x in range(width):
                vtk_image.SetScalarComponentFromDouble(x, y, 0, 0, img_data[y, x])
        
        # 创建颜色映射
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)  # 黑色
        color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)  # 白色
        
        # 创建不透明度映射
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0, 0.0)
        opacity_func.AddPoint(255, 1.0)
        
        # 创建图像属性和映射器
        image_property = vtk.vtkImageProperty()
        image_property.SetColorWindow(255)
        image_property.SetColorLevel(127.5)
        image_property.SetAmbient(1.0)
        image_property.SetInterpolationTypeToLinear()
        
        # 创建图像slice
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputData(vtk_image)
        actor.SetProperty(image_property)
        
        # 创建渲染器和渲染窗口
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        
        # 创建交互器
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # 设置相机
        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(width/2, height/2, 100)
        camera.SetFocalPoint(width/2, height/2, 0)
        camera.SetParallelScale(height/2)
        
        # 初始化并开始
        interactor.Initialize()
        render_window.Render()
        interactor.Start()
    
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
    
    def set_segmentation_model(self, model_name, **kwargs):
        """设置分割模型"""
        if model_name == 'medsam':
            # 强制要求提供checkpoint_path
            if 'checkpoint_path' not in kwargs:
                # 尝试从预定义模型列表获取
                model_info = list_available_models().get('medsam')
                if model_info and model_info.get('exists'):
                    kwargs['checkpoint_path'] = model_info['weights_path']
                else:
                    raise ValueError("必须提供checkpoint_path参数或正确配置模型列表")
            
            # 创建实例并加载模型
            self.segmenter = MedSAMSegmenter(checkpoint_path=kwargs['checkpoint_path'])
            
            # 设备设置
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.segmenter.device = torch.device(device)
            
            print(f"MedSAM模型已成功初始化，使用设备: {self.segmenter.device}")
        elif model_name == 'deeplabv3':
            self.segmenter = DeeplabV3Segmenter(checkpoint_path=kwargs.get('checkpoint_path'), device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            print(f"已设置DeepLabV3分割器，使用设备: {self.segmenter.device}")
        elif model_name == 'unet3d':
            # 创建UNet3D分割器并传递权重路径和设备
            self.segmenter = UNet3DSegmenter(weights_path=kwargs.get('checkpoint_path'), device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            print("已创建UNet3D分割器")
            
            # 检查UNet3D分割器配置
            if hasattr(self.segmenter, 'config'):
                print(f"UNet3D配置: {self.segmenter.config}")
            
            # 显示标签映射(如果存在)
            if hasattr(self.segmenter, 'original_labels'):
                print("标签映射:")
                for idx, label in self.segmenter.idx_to_label.items():
                    print(f"  索引 {idx} -> 标签值 {label}")
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def segment_image(self, prompt_points=None, prompt_box=None):
        """
        根据用户提示执行分割
        
        参数:
            prompt_points: 点提示，格式为[((x1,y1), label1), ((x2,y2), label2), ...]
            prompt_box: 框提示，格式为[x1, y1, x2, y2]
        
        返回:
            分割掩码
        """
        if not hasattr(self, 'segmenter'):
            print("错误: 未设置分割器")
            return None
        
        if self.image_data is None:
            print("错误: 未加载图像")
            return None
        
        # 获取分割器类型
        segmenter_type = type(self.segmenter).__name__
        
        # 根据分割器类型执行不同的分割逻辑
        if segmenter_type == 'MedSAMSegmenter':
            # MedSAM需要点或框提示
            if self.is_3d:
                # 对当前切片进行分割
                if self.current_view == 'axial':
                    slice_data = self.image_data[self.current_slice]
                elif self.current_view == 'coronal':
                    slice_data = self.image_data[:, self.current_slice, :]
                    slice_data = np.rot90(slice_data, k=2)
                elif self.current_view == 'sagittal':
                    slice_data = self.image_data[:, :, self.current_slice]
                    slice_data = np.rot90(slice_data, k=2)
                
                # 执行分割
                self.mask = self.segmenter.segment(slice_data, prompt_points, prompt_box)
                
                # 修改返回的掩码形状以匹配3D数据
                if self.mask is not None and len(self.mask.shape) == 2:
                    # 创建全零3D掩码
                    full_mask = np.zeros_like(self.image_data, dtype=np.uint8)
                    
                    # 将2D掩码插入3D掩码的相应切片
                    if self.current_view == 'axial':
                        full_mask[self.current_slice] = self.mask
                    elif self.current_view == 'coronal':
                        mask_rot = np.rot90(self.mask, k=2)  # 反向旋转
                        full_mask[:, self.current_slice, :] = mask_rot
                    elif self.current_view == 'sagittal':
                        mask_rot = np.rot90(self.mask, k=2)  # 反向旋转
                        full_mask[:, :, self.current_slice] = mask_rot
                    
                    self.mask = full_mask
            else:
                # 2D图像直接分割
                self.mask = self.segmenter.segment(self.image_data, prompt_points, prompt_box)
        
        elif segmenter_type == 'DeepLabSegmenter':
            # DeepLabV3不需要提示，直接分割
            if self.is_3d:
                # 对当前切片进行分割
                if self.current_view == 'axial':
                    slice_data = self.image_data[self.current_slice]
                elif self.current_view == 'coronal':
                    slice_data = self.image_data[:, self.current_slice, :]
                    slice_data = np.rot90(slice_data, k=2)
                elif self.current_view == 'sagittal':
                    slice_data = self.image_data[:, :, self.current_slice]
                    slice_data = np.rot90(slice_data, k=2)
                
                # 执行分割
                self.mask = self.segmenter.segment(slice_data)
                
                # 修改返回的掩码形状以匹配3D数据
                if self.mask is not None and len(self.mask.shape) == 2:
                    # 创建全零3D掩码
                    full_mask = np.zeros_like(self.image_data, dtype=np.uint8)
                    
                    # 将2D掩码插入3D掩码的相应切片
                    if self.current_view == 'axial':
                        full_mask[self.current_slice] = self.mask
                    elif self.current_view == 'coronal':
                        mask_rot = np.rot90(self.mask, k=2)  # 反向旋转
                        full_mask[:, self.current_slice, :] = mask_rot
                    elif self.current_view == 'sagittal':
                        mask_rot = np.rot90(self.mask, k=2)  # 反向旋转
                        full_mask[:, :, self.current_slice] = mask_rot
                    
                    self.mask = full_mask
            else:
                # 2D图像直接分割
                self.mask = self.segmenter.segment(self.image_data)
        
        elif segmenter_type == 'UNet3DSegmenter':
            # UNet3D分割 - 3D体积分割，不需要提示点或框
            print("执行UNet3D分割...")
            
            # 对3D体积进行分割
            if self.is_3d:
                # 直接将完整3D体积传递给分割器
                try:
                    print(f"传递3D体积给分割器，形状: {self.image_data.shape}")
                    self.mask = self.segmenter.segment(self.image_data)
                    
                    if self.mask is not None:
                        print(f"分割成功，掩码形状: {self.mask.shape}")
                        
                        # 检查标签值
                        unique_values = np.unique(self.mask)
                        print(f"分割结果包含以下标签值: {unique_values}")
                        
                        return self.mask
                    else:
                        print("分割失败：返回了空掩码")
                        return None
                except Exception as e:
                    print(f"UNet3D分割过程中出错: {str(e)}")
                    traceback.print_exc()
                    return None
            else:
                print("错误: UNet3D分割器需要3D体积数据")
                return None
        
        else:
            print(f"错误: 不支持的分割器类型 {segmenter_type}")
            return None
        
        return self.mask

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
            'class': MedSAMSegmenter,
            'is_3d_capable': True
        },
        'deeplabv3': {
            'description': 'DeepLabV3 模型 (医学图像分割)',
            'weights_path': 'weights/DeeplabV3/complete_deeplabv3_model.pt',  # 更新为新的权重文件路径
            'class': DeeplabV3Segmenter,
            'is_3d_capable': False  # 明确标记不支持3D
        },
        'unet3d': {
            'description': '3D U-Net (适用于CT/MRI体积分割)',
            'weights_path': 'weights/U-net/final_model.pth',
            'class': UNet3DSegmenter,
            'is_3d_capable': True
        }
    }
    
    # 检查权重文件是否存在
    for name, info in models.items():
        if 'weights_path' in info and os.path.exists(info['weights_path']):
            info['status'] = '已安装'
        else:
            info['status'] = '未安装'
    
    return models 