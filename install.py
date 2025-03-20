import os
import sys
import subprocess

def main():
    print("正在设置 Pelvis Fracture Image Segmentation 环境...")
    print("Setting up Pelvis Fracture Image Segmentation environment...")
    
    # 检测 conda | Detect conda
    try:
        subprocess.run("conda --version", shell=True, check=True, stdout=subprocess.PIPE)
        print("检测到 conda，使用 conda 安装依赖...")
        print("Conda detected, using conda to install dependencies...")
        
        # 创建环境 | Create environment
        os.system("conda env create -f environment.yml")
        
        # 提示用户激活环境 | Prompt user to activate environment
        if sys.platform.startswith('win'):
            print("\n环境设置完成! 请使用以下命令激活环境:")
            print("Environment setup complete! Please use the following command to activate the environment:")
            print("    conda activate pelvis_seg")
        else:
            print("\n环境设置完成! 请使用以下命令激活环境:")
            print("Environment setup complete! Please use the following command to activate the environment:")
            print("    conda activate pelvis_seg")
            
    except subprocess.CalledProcessError:
        print("未检测到 conda。请先安装 Anaconda 或 Miniconda。")
        print("Conda not detected. Please install Anaconda or Miniconda first.")
        print("下载链接 | Download link: https://www.anaconda.com/download")

if __name__ == "__main__":
    main() 