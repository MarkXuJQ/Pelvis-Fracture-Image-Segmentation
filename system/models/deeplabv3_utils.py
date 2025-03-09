import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.segmentation as seg_models
import numpy as np
import cv2
import os

class DeepLabV3Segmenter:
    def __init__(self, num_classes=21, pretrained=False, device='cuda', backbone='resnet50'):
        """
        初始化DeepLabV3分割器
        
        参数:
            num_classes: 分割类别数量，默认为21（Pascal VOC数据集）
            pretrained: 是否使用预训练权重（已弃用，保留为兼容参数）
            device: 计算设备 ('cuda' 或 'cpu')
            backbone: 骨干网络类型 ('resnet50' 或 'resnet101')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.backbone = backbone
        print(f"使用设备: {self.device}, 骨干网络: {backbone}")
        
        # 根据指定的骨干网络加载相应的模型
        if backbone == 'resnet50':
            self.model = seg_models.deeplabv3_resnet50(weights=None)
        elif backbone == 'resnet101':
            self.model = seg_models.deeplabv3_resnet101(weights=None)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}, 只支持 'resnet50' 或 'resnet101'")
        
        # 修改最后的分类层以适应我们的类别数
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 将模型移至指定设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)
    
    def load_weights(self, checkpoint_path):
        """加载自定义权重"""
        try:
            # 检查文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"DeepLabV3 权重文件不存在: {checkpoint_path}")
            
            print(f"正在加载权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 检查加载的字典类型
            print(f"检查点类型: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"检查点字典键: {list(checkpoint.keys())}")
            
            # 尝试不同的加载方式
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 创建新的状态字典，修正键名不匹配问题
            new_state_dict = {}
            for k, v in state_dict.items():
                # 处理嵌套的分类器结构
                if k.startswith('classifier.classifier.'):
                    new_k = k.replace('classifier.classifier.', 'classifier.')
                    new_state_dict[new_k] = v
                elif k.startswith('module.'):
                    # 处理 DataParallel 封装的模型
                    new_k = k[7:]  # 移除 'module.' 前缀
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            # 创建模型的临时副本，以便在加载失败时恢复
            import copy
            model_copy = copy.deepcopy(self.model)
            
            try:
                # 尝试加载处理后的状态字典（使用严格模式）
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"模型加载完成。")
                if missing:
                    print(f"缺失的键: {len(missing)} 个")
                if unexpected:
                    print(f"意外的键: {len(unexpected)} 个")
                
            except Exception as e:
                # 如果加载失败，恢复模型
                self.model = model_copy
                print(f"加载状态字典失败，恢复原始模型: {str(e)}")
                raise e
            
        except Exception as e:
            print(f"加载权重时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 确保图像是RGB格式
        if len(image.shape) == 2:  # 灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:  # 单通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 调整大小到模型期望的输入尺寸
        original_size = image.shape[:2]
        image = cv2.resize(image, (513, 513))
        
        # 转换为PyTorch张量并归一化
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image = image.to(self.device)
        image = (image - self.mean) / self.std
        
        return image.unsqueeze(0), original_size  # 添加批次维度
    
    def segment(self, image, target_class=1):
        """
        使用DeepLabV3对图像进行分割
        
        参数:
            image: 输入图像，可以是灰度或RGB
            target_class: 需要分割的目标类别索引
            
        返回:
            binary_mask: 二值掩码，目标区域为1，背景为0
        """
        # 预处理图像
        input_tensor, original_size = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]  # (num_classes, H, W)
            
        # 提取目标类别的预测
        if target_class < output.shape[0]:
            target_pred = output[target_class].cpu().numpy()
        else:
            # 如果目标类别超出范围，则使用argmax分配最可能的类别
            target_pred = output.argmax(0).cpu().numpy()
            target_pred = (target_pred == target_class).astype(float)
        
        # 调整回原始尺寸
        target_pred = cv2.resize(target_pred, (original_size[1], original_size[0]))
        
        # 二值化
        binary_mask = (target_pred > 0.5).astype(np.uint8)
        
        return binary_mask 