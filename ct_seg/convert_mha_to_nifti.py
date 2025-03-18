import os
import SimpleITK as sitk
import glob
from pathlib import Path

def convert_mha_to_nifti(input_path, output_path, new_name):
    """
    将.mha文件转换为.nii.gz格式
    Convert .mha file to .nii.gz format
    """
    # 读取.mha文件
    img = sitk.ReadImage(input_path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入.nii.gz文件
    sitk.WriteImage(img, output_path)
    print(f"转换完成: {input_path} -> {output_path}")

def main():
    # 设置路径
    base_dir = Path("ct_seg/data")
    image_dir = base_dir / "PENGWIN_CT_train_images"
    label_dir = base_dir / "PENGWIN_CT_train_labels"
    
    # 创建输出目录
    output_image_dir = base_dir / "images"
    output_label_dir = base_dir / "labels"
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有.mha图像文件
    image_files = sorted(glob.glob(str(image_dir / "*.mha")))
    
    # 处理每个文件
    for i, image_file in enumerate(image_files, 1):
        # 获取文件名（不带路径和扩展名）
        filename = os.path.basename(image_file).split('.')[0]
        
        # 找到对应的标签文件
        label_file = label_dir / f"{filename}.mha"
        
        # 如果标签文件存在，则进行处理
        if os.path.exists(label_file):
            # 创建新的文件名
            new_filename = f"{i:03d}_0000.nii.gz"
            
            # 转换图像文件
            output_image_path = output_image_dir / new_filename
            convert_mha_to_nifti(image_file, str(output_image_path), new_filename)
            
            # 转换标签文件
            output_label_path = output_label_dir / new_filename
            convert_mha_to_nifti(str(label_file), str(output_label_path), new_filename)
            
            print(f"已处理文件 {i}/100: {filename} -> {new_filename}")

if __name__ == "__main__":
    main()
    print("所有文件转换完成！") 