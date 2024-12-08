# ct_seg/models/U_net/predict.py
import torch
import SimpleITK as sitk
import numpy as np
from unet_model import UNet3D 
import matplotlib.pyplot as plt
import os
import traceback
from skimage.filters import threshold_otsu

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
        checkpoint = torch.load(model_path, map_location=device)
        
        # 打印检查点内容
        print("Checkpoint keys:", checkpoint.keys())
        
        # 如果检查点包含 'model_state_dict' 键
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        # 打印一些模型参数来验证
        for name, param in model.named_parameters():
            print(f"{name}: {param.mean().item():.4f} (mean)")
            break  # 只打印第一层作为示例
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # 加���和预处理图像
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
                        
                        print(f"Prediction stats before sigmoid:")
                        print(f"Mean: {patch_pred.mean().item():.4f}")
                        print(f"Max: {patch_pred.max().item():.4f}")
                        print(f"Min: {patch_pred.min().item():.4f}")
                        
                        # 1. 归一化预测值
                        if patch_pred.max().item() < 0:
                            patch_pred = (patch_pred - patch_pred.min()) / (patch_pred.max() - patch_pred.min())
                        
                        # 2. 应用sigmoid
                        patch_pred = torch.sigmoid(patch_pred)
                        
                        print(f"Prediction stats after sigmoid:")
                        print(f"Mean: {patch_pred.mean().item():.4f}")
                        print(f"Max: {patch_pred.max().item():.4f}")
                        print(f"Min: {patch_pred.min().item():.4f}")
                        
                        # 3. 转换到numpy进行处理
                        patch_pred_np = patch_pred.cpu().numpy().squeeze()
                        
                        # 4. 计算动态阈值
                        # 使用Otsu's方法或基于分布的阈值
                        try:
                            threshold = threshold_otsu(patch_pred_np)
                            print(f"Dynamic threshold: {threshold:.4f}")
                        except:
                            threshold = 0.5  # 如果Otsu方法失败，使用默认阈值
                            print(f"Using default threshold: {threshold}")
                        
                        # 5. 应用阈值
                        binary_pred = (patch_pred_np > threshold).astype(np.uint8)
                        
                        # 6. 打印二值化后的统计信息
                        print(f"Binary prediction stats:")
                        print(f"Non-zero pixels: {np.count_nonzero(binary_pred)}")
                        print(f"Total pixels: {binary_pred.size}")
                        print(f"Percentage: {100 * np.count_nonzero(binary_pred) / binary_pred.size:.2f}%")
                        
                        # 7. 如果预测比例不合理，进行调整
                        if np.count_nonzero(binary_pred) / binary_pred.size > 0.5:  # 如果白色像素超过50%
                            binary_pred.fill(0)  # 清零
                        elif np.count_nonzero(binary_pred) / binary_pred.size < 0.001:  # 如果白色像素太少
                            # 尝试使用更低的阈值
                            binary_pred = (patch_pred_np > threshold * 0.5).astype(np.uint8)
                        
                        prediction_array[d_start:d_end, h_start:h_end, w_start:w_end] = binary_pred
                        
                        # 再次清除 GPU 缓存
                        torch.cuda.empty_cache()
        
        print("Prediction completed successfully")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
    # 保存结果
    if output_path:
        try:
            # 如果文件存在，先删除
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Removed existing file: {output_path}")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            
            # 保存预测结果
            output_image = sitk.GetImageFromArray(prediction_array)
            output_image.CopyInformation(image)  # 复制原始图像的元数据
            
            # 打印一些调试信息
            print(f"Prediction array shape: {prediction_array.shape}")
            print(f"Prediction array dtype: {prediction_array.dtype}")
            print(f"Saving segmentation to: {output_path}")
            
            sitk.WriteImage(output_image, output_path)
            
            # 验证文件是否成功保存
            if os.path.exists(output_path):
                print(f"File successfully saved at {output_path}")
                print(f"File size: {os.path.getsize(output_path)} bytes")
            else:
                print("Warning: File was not saved successfully")
                
        except Exception as e:
            print(f"Error saving output: {e}")
            print(f"Full error details: {traceback.format_exc()}")
    
    # 在保存文件后添加验证代码
    try:
        # 尝试重新读取保存的文件
        test_read = sitk.ReadImage(output_path)
        test_array = sitk.GetArrayFromImage(test_read)
        print(f"Successfully verified file: {output_path}")
        print(f"Loaded array shape: {test_array.shape}")
        print(f"Loaded array dtype: {test_array.dtype}")
    except Exception as e:
        print(f"Error verifying saved file: {e}")
    
    # 在处理整个图像后，添加这些检查
    print(f"Final prediction array stats:")
    print(f"Unique values: {np.unique(prediction_array)}")
    print(f"Non-zero pixels: {np.count_nonzero(prediction_array)}")
    print(f"Total pixels: {prediction_array.size}")
    print(f"Percentage of non-zero pixels: {100 * np.count_nonzero(prediction_array) / prediction_array.size:.2f}%")

    # 如果需要，可以保存一个中间切片用于检查
    middle_slice = prediction_array[prediction_array.shape[0]//2]
    plt.imsave('debug_middle_slice.png', middle_slice)
    
    return prediction_array

if __name__ == "__main__":
    # 设置路径
    model_path = r'ct_seg\notebooks\best_model_loss.pth'
    input_path = r'ct_seg\data\PENGWIN_CT_train_images\051.mha'
    output_path = r'ct_seg\data\results\U_net\segmentation.mha'
    
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    
    # 运行预测
    prediction = predict_single_scan(model_path, input_path, output_path)
