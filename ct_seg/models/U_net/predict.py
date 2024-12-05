# ct_seg/models/U_net/predict.py
import torch
import SimpleITK as sitk
import numpy as np
from unet_model import UNet3D

def predict_single_scan(model_path, input_image_path, output_path=None, patch_size=(128, 128, 128), overlap=16):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU")
    
    # 加载模型
    try:
        model = UNet3D(in_channels=1, out_channels=1)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully to", device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # 加载和预处理图像
    try:
        image = sitk.ReadImage(input_image_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.clip(image_array, -1000, 1000)
        image_array = (image_array + 1000) / 2000
        
        # 获取图像尺寸
        depth, height, width = image_array.shape
        prediction_array = np.zeros_like(image_array, dtype=np.uint8)
        
        # 计算步长（考虑重叠）
        d_step = patch_size[0] - overlap
        h_step = patch_size[1] - overlap
        w_step = patch_size[2] - overlap
        
        # 分块处理
        for d in range(0, depth, d_step):
            for h in range(0, height, h_step):
                for w in range(0, width, w_step):
                    # 计算当前块的范围
                    d_end = min(d + patch_size[0], depth)
                    h_end = min(h + patch_size[1], height)
                    w_end = min(w + patch_size[2], width)
                    d_start = max(0, d_end - patch_size[0])
                    h_start = max(0, h_end - patch_size[1])
                    w_start = max(0, w_end - patch_size[2])
                    
                    # 提取块
                    patch = image_array[d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # 处理块
                    patch_tensor = torch.from_numpy(patch).float()
                    patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
                    patch_tensor = patch_tensor.to(device)
                    
                    # 清除 GPU 缓存
                    torch.cuda.empty_cache()
                    
                    # 预测
                    with torch.no_grad():
                        patch_pred = model(patch_tensor)
                        patch_pred = torch.sigmoid(patch_pred) > 0.5
                        patch_pred = patch_pred.cpu().numpy().squeeze().astype(np.uint8)
                    
                    # 将预测结果放回原始大小的数组
                    prediction_array[d_start:d_end, h_start:h_end, w_start:w_end] = patch_pred
                    
                    # 再次清除 GPU 缓存
                    torch.cuda.empty_cache()
        
        print("Prediction completed successfully")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
    # 保存结果
    if output_path:
        try:
            output_image = sitk.GetImageFromArray(prediction_array)
            output_image.CopyInformation(image)
            sitk.WriteImage(output_image, output_path)
            print(f"Saved segmentation to {output_path}")
        except Exception as e:
            print(f"Error saving output: {e}")
    
    return prediction_array

if __name__ == "__main__":
    # 设置路径
    model_path = r'ct_seg\notebooks\best_unet_model.pth'
    input_path = r'ct_seg\data\PENGWIN_CT_train_images\051.mha'
    output_path = r'ct_seg\models\results\segmentation.mha'
    
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    
    # 运行预测
    prediction = predict_single_scan(model_path, input_path, output_path)
