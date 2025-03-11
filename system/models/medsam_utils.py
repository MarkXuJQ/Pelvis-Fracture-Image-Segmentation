import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os

class MedSamSegmenter:
    def __init__(self, model_type='vit_b', checkpoint_path=None, device='cuda'):
        """
        初始化Medical SAM分割器
        
        参数:
            model_type: 模型类型 ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: 权重文件路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # 延迟加载以避免不必要的导入
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            print(f"加载Medical SAM模型 ({model_type})...")
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            print("Medical SAM模型加载完成!")
            
        except ImportError:
            print("警告: segment_anything库未安装，无法使用MedSAM模型")
            self.sam = None
            self.predictor = None
            
        except Exception as e:
            print(f"加载Medical SAM模型时出错: {str(e)}")
            self.sam = None
            self.predictor = None
    
    def segment(self, image, points=None, point_labels=None, box=None):
        """
        使用Medical SAM模型分割图像
        
        参数:
            image: 输入图像
            points: 点提示，格式为[[x1, y1], [x2, y2], ...]
            point_labels: 点标签，1表示前景，0表示背景
            box: 边界框提示，格式为[x1, y1, x2, y2]
            
        返回:
            mask: 分割掩码
        """
        if self.predictor is None:
            raise ValueError("Medical SAM模型未正确加载")
        
        # 图像预处理 - 确保图像是正确的格式
        print(f"原始图像形状: {image.shape}, 数据类型: {image.dtype}, 值范围: [{image.min()}-{image.max()}]")
        
        # 确保图像的值范围在[0,1]之间
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # 对于CT图像，常常需要窗宽窗位调整
        if image.min() < 0:
            # 这可能是CT图像，进行窗宽窗位调整
            wl, ww = 40, 400  # 骨窗的常用窗宽窗位
            image = np.clip((image - (wl - ww/2)) / ww + 0.5, 0, 1)
            print("应用了CT图像窗宽窗位调整")
        
        # 确保图像是3通道的
        if len(image.shape) == 2:
            # 转换灰度图为RGB
            image = np.stack([image, image, image], axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
            
        # 确保图像为uint8类型
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 设置输入图像
        self.predictor.set_image(image_uint8)
        
        # 准备点提示
        input_points = np.array(points) if points is not None else None
        input_labels = np.array(point_labels) if point_labels is not None else None
        
        # 打印调试信息
        print(f"分割提示 - 点: {input_points}, 标签: {input_labels}, 框: {box}")
        
        # 生成掩码
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box,
            multimask_output=True  # 返回多个候选掩码
        )
        
        print(f"生成了 {len(masks)} 个掩码，分数: {scores}")
        
        # 返回最高分数的掩码
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx] 