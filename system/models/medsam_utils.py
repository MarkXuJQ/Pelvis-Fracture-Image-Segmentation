import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os
import torch.nn.functional as F

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

    def segment_with_medsam(self, image, points=None, point_labels=None, box=None):
        """使用MedSAM进行分割"""
        try:
            print("=== MedSAM分割开始 ===")
            print(f"输入图像形状: {image.shape}, 类型: {image.dtype}")
            print(f"点: {points}")
            print(f"点标签: {point_labels}")
            print(f"框: {box}")
            
            if not hasattr(self, 'medsam_model') or self.medsam_model is None:
                print("错误: MedSAM模型未加载")
                return None
            
            if not hasattr(self, 'image_embedding') or self.image_embedding is None:
                print("错误: 图像嵌入未计算")
                return None
            
            # 确保图像是适当的格式
            if isinstance(box, list):
                box = np.array(box, dtype=np.float32)
            
            # 转换点和框坐标到标准格式
            H, W = image.shape[:2]
            print(f"图像尺寸: H={H}, W={W}")
            
            # 处理框提示
            if box is not None:
                # 确保框坐标是标准 numpy 数组
                print(f"原始框坐标: {box}, 类型: {type(box)}")
                if isinstance(box, list) and len(box) == 4:
                    box = np.array(box, dtype=np.float32)
                
                # 将框坐标调整为1024x1024比例
                box_1024 = box / np.array([W, H, W, H]) * 1024
                box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
                if len(box_torch.shape) == 1:
                    box_torch = box_torch.unsqueeze(0)  # 添加批次维度
                print(f"处理后的框: {box_torch}")
            else:
                box_torch = None
            
            # 处理点提示
            if points is not None and point_labels is not None and len(points) > 0:
                # 将点坐标调整为1024x1024比例
                points_1024 = np.array(points) / np.array([W, H]) * 1024
                points_torch = torch.as_tensor(points_1024, dtype=torch.float, device=self.device)
                if len(points_torch.shape) == 2:
                    points_torch = points_torch.unsqueeze(0)  # 添加批次维度
                
                point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                if len(point_labels_torch.shape) == 1:
                    point_labels_torch = point_labels_torch.unsqueeze(0)  # 添加批次维度
                print(f"处理后的点: {points_torch}")
                print(f"处理后的点标签: {point_labels_torch}")
            else:
                # 如果没有点提示，但有框提示，添加一个框内的默认前景点
                if box is not None:
                    # 创建一个框中心点作为前景点
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    points_1024 = np.array([[center_x, center_y]]) / np.array([W, H]) * 1024
                    points_torch = torch.as_tensor(points_1024, dtype=torch.float, device=self.device).unsqueeze(0)
                    point_labels_torch = torch.as_tensor([1], dtype=torch.int, device=self.device).unsqueeze(0)
                    print(f"创建默认前景点: {points_torch}")
                else:
                    points_torch = None
                    point_labels_torch = None
            
            # 使用提示编码器
            print("使用提示编码器...")
            with torch.no_grad():
                try:
                    sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
                        points=points_torch,
                        boxes=box_torch,
                        masks=None,
                    )
                    
                    print("提示编码完成，解码掩码...")
                    # 解码掩码
                    low_res_logits, _ = self.medsam_model.mask_decoder(
                        image_embeddings=self.image_embedding,
                        image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # 将低分辨率掩码调整到原始图像大小
                    print("调整掩码分辨率...")
                    low_res_pred = torch.sigmoid(low_res_logits)
                    low_res_pred = F.interpolate(
                        low_res_pred,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    
                    # 转换为numpy掩码
                    mask = low_res_pred.squeeze().cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    print(f"生成掩码完成，形状: {mask.shape}, 包含像素数: {np.sum(mask)}")
                    
                    return mask
                except Exception as e:
                    print(f"分割过程出错: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        except Exception as e:
            print(f"MedSAM分割总体错误: {e}")
            import traceback
            traceback.print_exc()
            return None 