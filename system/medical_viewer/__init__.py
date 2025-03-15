"""
医学图像查看器模块
提供医学图像加载、显示、分割等功能
"""

# 修复导入路径 - 使用绝对导入
from system.medical_viewer.medical_image_utils import MedicalImageProcessor
from system.medical_viewer.vtk_3d_viewer import VTK3DViewer
from system.medical_viewer.ct_viewer import CTViewer
from system.medical_viewer.xray_viewer import XRayViewer
from system.medical_viewer.image_viewer_window import ImageViewerWindow

# 导入分割器
from system.medical_viewer.segmenters import DeeplabV3Segmenter, MedSAMSegmenter

__all__ = [
    'MedicalImageViewer',
    'MedicalImageProcessor',
    'VTK3DViewer',
    'CTViewer',
    'XRayViewer',
    'ImageViewerWindow',
    'DeeplabV3Segmenter',
    'MedSAMSegmenter'
]