from .medsam_utils import MedSamSegmenter
from .deeplabv3_utils import DeepLabV3Segmenter

class SegmentationAPI:
    def __init__(self, method='medsam', **kwargs):
        """
        初始化分割API
        
        参数:
            method: 分割方法 ('medsam' 或 'deeplabv3')
            **kwargs: 传递给相应分割器的参数
        """
        self.method = method
        self.segmenter = None
        
        if method == 'medsam':
            self.segmenter = MedSamSegmenter(
                model_type=kwargs.get('model_type', 'vit_b'),
                checkpoint_path=kwargs.get('checkpoint_path', 'weights/MedSAM/medsam_vit_b.pth'),
                device=kwargs.get('device', 'cuda')
            )
        elif method == 'deeplabv3':
            self.segmenter = DeepLabV3Segmenter(
                num_classes=kwargs.get('num_classes', 2),
                pretrained=kwargs.get('pretrained', True),
                device=kwargs.get('device', 'cuda')
            )
            
            if 'checkpoint_path' in kwargs:
                self.segmenter.load_weights(kwargs['checkpoint_path'])
        else:
            raise ValueError(f"不支持的分割方法: {method}")
    
    def segment(self, image, **kwargs):
        """
        对图像进行分割
        
        参数:
            image: 输入图像
            **kwargs: 传递给相应分割器的参数
        
        返回:
            mask: 分割掩码
        """
        if self.method == 'medsam':
            return self.segmenter.segment(
                image,
                points=kwargs.get('points', None),
                point_labels=kwargs.get('point_labels', None),
                boxes=kwargs.get('boxes', None)
            )
        elif self.method == 'deeplabv3':
            return self.segmenter.segment(
                image,
                target_class=kwargs.get('target_class', 1)
            ) 