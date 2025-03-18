#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将.mha文件转换为.nii.gz文件格式的脚本
适用于Colab环境，路径已硬编码
用法: python convert_mha_to_nifti.py
"""

import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import shutil

def convert_directory(input_dir, output_dir):
    """
    将一个目录中的所有.mha文件转换为.nii.gz格式
    
    参数:
        input_dir: 包含.mha文件的目录
        output_dir: 保存.nii.gz文件的目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.mha文件
    mha_files = glob.glob(os.path.join(input_dir, "*.mha"))
    
    print(f"在 {input_dir} 中找到 {len(mha_files)} 个.mha文件")
    
    # 转换每个文件
    for mha_path in tqdm(mha_files, desc="转换文件"):
        # 获取基本文件名
        base_name = os.path.basename(mha_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 设置输出路径
        nifti_path = os.path.join(output_dir, name_without_ext + ".nii.gz")
        
        try:
            # 读取.mha文件
            img = sitk.ReadImage(mha_path)
            
            # 写入.nii.gz文件
            sitk.WriteImage(img, nifti_path)
        except Exception as e:
            print(f"转换 {mha_path} 时出错: {e}")

def main():
    """主函数"""
    # 硬编码输入路径
    data_base_dir = '/content/drive/MyDrive/ct_segmentation/data'
    train_images_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_images')
    train_labels_dir = os.path.join(data_base_dir, 'PENGWIN_CT_train_labels')
    
    # 硬编码输出路径
    output_base_dir = '/content/drive/MyDrive/ct_segmentation/3DU-net'
    images_output_dir = os.path.join(output_base_dir, 'images')
    labels_output_dir = os.path.join(output_base_dir, 'labels')
    
    # 确认目录存在
    if not os.path.isdir(train_images_dir):
        print(f"错误: 输入图像目录 {train_images_dir} 不存在")
        return
        
    if not os.path.isdir(train_labels_dir):
        print(f"错误: 输入标签目录 {train_labels_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    print("\n======== 开始转换.mha到.nii.gz ========")
    print(f"输入图像目录: {train_images_dir}")
    print(f"输入标签目录: {train_labels_dir}")
    print(f"输出图像目录: {images_output_dir}")
    print(f"输出标签目录: {labels_output_dir}")
    
    # 转换训练图像到新的images目录
    print("\n转换图像文件...")
    convert_directory(train_images_dir, images_output_dir)
    
    # 转换训练标签到新的labels目录
    print("\n转换标签文件...")
    convert_directory(train_labels_dir, labels_output_dir)
    
    # 检查是否有验证集 - 也将它们合并到相同的输出目录
    val_images_dir = os.path.join(data_base_dir, "PENGWIN_CT_val_images")
    val_labels_dir = os.path.join(data_base_dir, "PENGWIN_CT_val_labels")
    
    if os.path.exists(val_images_dir) and os.path.exists(val_labels_dir):
        print("\n发现验证集，添加到相同目录...")
        # 转换验证图像
        convert_directory(val_images_dir, images_output_dir)
        
        # 转换验证标签
        convert_directory(val_labels_dir, labels_output_dir)
    
    print("\n转换完成！")
    print(f"转换后的NIfTI图像文件保存在：{images_output_dir}")
    print(f"转换后的NIfTI标签文件保存在：{labels_output_dir}")
    
    # 检查转换结果
    image_files = glob.glob(os.path.join(images_output_dir, "*.nii.gz"))
    label_files = glob.glob(os.path.join(labels_output_dir, "*.nii.gz"))
    
    print(f"\n统计信息:")
    print(f"- 转换了 {len(image_files)} 个图像文件")
    print(f"- 转换了 {len(label_files)} 个标签文件")

if __name__ == "__main__":
    main() 