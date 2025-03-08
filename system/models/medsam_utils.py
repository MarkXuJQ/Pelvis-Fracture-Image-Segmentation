import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os

class MedSamSegmenter:
    def __init__(self, model_type="vit_b", checkpoint_path="weights\MedSAM\medsam_vit_b.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化 MedSam 分割器
        
        参数:
            model_type: SAM 模型类型 (vit_h, vit_l, vit_b)
            checkpoint_path: 预训练权重路径
            device: 运行设备 (cuda 或 cpu)
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # 检查模型文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到 MedSam 模型权重文件: {checkpoint_path}")
        
        # 初始化 SAM 模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        # 创建 SAM 预测器
        self.predictor = SamPredictor(self.sam)
        
    def segment(self, image, points=None, point_labels=None, boxes=None, multimask_output=False):
        """
        使用 MedSam 执行分割
        
        参数:
            image: 输入图像 (numpy 数组)
            points: 提示点坐标, shape: (N, 2)
            point_labels: 提示点标签 (1 表示前景, 0 表示背景), shape: (N,)
            boxes: 边界框坐标 [x1, y1, x2, y2]
            multimask_output: 是否输出多个候选掩码
            
        返回:
            掩码: numpy 二值数组
        """
        # 预处理图像
        if len(image.shape) == 2:  # 灰度图
            # 归一化到0-255并转换为3通道
            image_normalized = ((image - image.min()) * 255 / (image.max() - image.min())).astype(np.uint8)
            image_rgb = np.stack([image_normalized] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB图
            image_rgb = image
        else:
            raise ValueError(f"不支持的图像格式: {image.shape}")
        
        # 设置图像
        self.predictor.set_image(image_rgb)
        
        # 处理输入提示
        if points is not None:
            points = np.array(points)
            if point_labels is None:
                # 默认所有点都是前景
                point_labels = np.ones(points.shape[0])
                
            # 添加输入点
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=boxes,
                multimask_output=multimask_output
            )
        else:
            # 使用自动分割模式
            # 这里用到的是 MedSam 的自动模式，可能需要根据具体实现调整
            image_embedding = self.predictor.get_image_embedding().cpu().numpy()
            
            # 这里简化处理，实际应根据 MedSam 的具体实现调整
            masks = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=multimask_output
            )[0]
            
        # 如果有多个掩码，选择得分最高的
        if multimask_output and len(masks.shape) == 3:
            mask_idx = np.argmax(scores)
            mask = masks[mask_idx]
        else:
            mask = masks[0] if len(masks.shape) == 3 else masks
            
        return mask.astype(bool) 