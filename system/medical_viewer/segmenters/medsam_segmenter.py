import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform
from segment_anything import sam_model_registry

class MedSAMSegmenter:
    """专门用于MedSAM模型的分割处理器"""
    
    def __init__(self, checkpoint_path=None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果初始化时提供路径，自动加载
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path: str):
        """加载MedSAM模型权重
        参数:
            checkpoint_path (str): 必须提供的.pth权重文件路径
        """
        # 参数验证
        if not isinstance(checkpoint_path, str):
            raise TypeError(f"需要字符串路径，收到 {type(checkpoint_path)}")
        if not checkpoint_path.endswith('.pth'):
            raise ValueError("权重文件必须是.pth格式")
            
        print(f"正在从 {checkpoint_path} 加载MedSAM权重...")
        
        try:
            # 初始化模型结构
            self.model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            print("MedSAM模型加载完成！")
        except Exception as e:
            error_msg = f"""
            加载模型失败，请检查：
            1. 文件路径是否正确（当前路径：{os.path.abspath(checkpoint_path)}）
            2. PyTorch版本是否匹配（当前版本：{torch.__version__})
            3. 文件是否完整（建议MD5校验）
            原始错误信息：{str(e)}
            """
            raise RuntimeError(error_msg)
    
    def preprocess_image(self, image):
        """
        预处理输入图像到SAM模型所需格式
        
        参数:
            image: 输入图像 (numpy数组)
            
        返回:
            预处理后的图像和原始尺寸
        """
        # 记录原始尺寸
        original_size = image.shape[:2]
        
        # 确保图像是浮点型
        image = image.astype(np.float32)
        
        # 将值范围归一化到 [0,1]
        if image.max() > 1.0:
            image = image / 255.0 if image.max() > 255.0 else image / image.max()
        
        # 确保图像是3通道的
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
            
        # 调整大小到 1024x1024 (SAM的标准输入尺寸)
        img_1024 = transform.resize(
            image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        
        # 确保值范围在 [0,1]
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img_1024).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor, original_size
    
    def compute_image_embedding(self, image):
        """
        计算图像的嵌入表示
        
        参数:
            image: 输入图像 (numpy数组)
            
        返回:
            图像嵌入
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
            
        # 预处理图像
        img_tensor, original_size = self.preprocess_image(image)
        
        # 计算嵌入
        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_tensor)
            
        return image_embedding, original_size
    
    def segment(self, image, points=None, point_labels=None, box=None):
        """
        使用MedSAM模型进行分割
        
        参数:
            image: 输入图像 (numpy数组)
            points: 点提示的坐标，形状为 (N, 2)
            point_labels: 点标签，1表示前景，0表示背景
            box: 边界框提示，形状为 [x1, y1, x2, y2]
            
        返回:
            分割掩码，布尔类型的numpy数组
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        # 预处理图像并计算嵌入
        image_embedding, original_size = self.compute_image_embedding(image)
        
        H, W = original_size
        
        # 在1024x1024大小的图像坐标系下准备提示
        # 转换点坐标
        if points is not None and len(points) > 0:
            points_1024 = np.array(points) / np.array([W, H]) * 1024
            points_torch = torch.from_numpy(points_1024).float().to(self.device)
            if len(points_torch.shape) == 2:
                points_torch = points_torch.unsqueeze(0)  # 添加批次维度
                
            # 转换点标签
            if point_labels is not None:
                labels_torch = torch.from_numpy(np.array(point_labels)).int().to(self.device)
                if len(labels_torch.shape) == 1:
                    labels_torch = labels_torch.unsqueeze(0)
            else:
                # 默认所有点都是前景点
                labels_torch = torch.ones(points_torch.shape[:2], dtype=torch.int).to(self.device)
        else:
            points_torch = None
            labels_torch = None
        
        # 转换框坐标
        if box is not None:
            box_1024 = np.array(box) / np.array([W, H, W, H]) * 1024
            box_torch = torch.from_numpy(box_1024).float().to(self.device)
            if len(box_torch.shape) == 1:
                box_torch = box_torch.unsqueeze(0)
        else:
            box_torch = None
            
        print(f"分割提示 - 点: {points_torch.shape if points_torch is not None else None}, "
              f"标签: {labels_torch.shape if labels_torch is not None else None}, "
              f"框: {box_torch.shape if box_torch is not None else None}")
        
        # 使用提示编码器
        with torch.no_grad():
            # 获取提示编码
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(points_torch, labels_torch) if points_torch is not None else None,
                boxes=box_torch,
                masks=None,
            )
            
            # 解码掩码
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 将低分辨率掩码调整到原始图像大小
            low_res_pred = torch.sigmoid(low_res_logits)
            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            
            # 转换为numpy掩码
            mask = low_res_pred.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(bool)
            
        return mask 