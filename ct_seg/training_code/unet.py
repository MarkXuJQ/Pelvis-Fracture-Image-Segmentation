import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch
import torch.nn as nn
import numpy as np

print_config()
# 使用指定路径
root_dir = r"D:\pelvis\ct_seg\data\results\U_net"
os.makedirs(root_dir, exist_ok=True)  # 确保目录存在
print(f"模型和结果将保存到: {root_dir}")

# 更新num_classes为31，与实际标签值范围(0-30)匹配
num_classes = 31

# 使用一致的空间尺寸变量
patch_size = (64, 64, 64)  # 改为64³而不是96³

# 修改训练转换以使用正确的裁剪尺寸
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # 保留必要的预处理步骤
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 添加更激进的裁剪以减少内存需求
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 使用与模型匹配的尺寸
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,  # 使用一致的尺寸
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
data_dir = "D:\\pelvis\\ct_seg\\data"
split_json = os.path.join(data_dir, "dataset.json")

datasets = split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

# 添加进度显示，监控数据加载过程
print("开始加载训练数据集...")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
    progress=True,  # 添加进度显示
)
print("训练数据集加载完成!")

print("开始加载验证数据集...")
val_ds = CacheDataset(
    data=val_files, 
    transform=val_transforms, 
    cache_num=6, 
    cache_rate=1.0, 
    num_workers=4,
    progress=True,  # 添加进度显示
)
print("验证数据集加载完成!")

# 创建数据加载器时禁用多进程
print("创建无多进程的训练数据加载器...")
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, 
    num_workers=0,  # 关键修复: 禁用多进程
    pin_memory=True
)

print("创建无多进程的验证数据加载器...")
val_loader = DataLoader(
    val_ds, batch_size=1, 
    num_workers=0,  # 关键修复: 禁用多进程
    pin_memory=True
)

slice_map = {
    "001_0000.nii.gz": 143,
    "002_0000.nii.gz": 178,
    "003_0000.nii.gz": 195,
    "004_0000.nii.gz": 162,
    "005_0000.nii.gz": 186,
    "006_0000.nii.gz": 153,
    "007_0000.nii.gz": 201,
    "008_0000.nii.gz": 172,
    "009_0000.nii.gz": 128,
    "010_0000.nii.gz": 190,
    "011_0000.nii.gz": 165,
    "012_0000.nii.gz": 148,
    "081_0000.nii.gz": 75,
    "082_0000.nii.gz": 156,
    "083_0000.nii.gz": 183,
}
case_num = 0

# 获取图像中间切片作为默认值
def get_middle_slice(img_shape):
    return img_shape[3] // 2  # 获取z轴的中间切片

img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")

# 使用字典的get方法，如果键不存在则使用默认值
slice_index = slice_map.get(img_name, get_middle_slice(img_shape))
print(f"Using slice index: {slice_index} for image: {img_name}")

plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, slice_index].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_index].detach().cpu())
plt.show()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保模型定义使用相同的尺寸变量
model = UNETR(
    in_channels=1,
    out_channels=num_classes,  # 修改为31
    img_size=patch_size,
    feature_size=16,
    hidden_size=384,
    mlp_dim=1536,
    num_heads=6,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# 使用标准损失函数，不进行标签映射
loss_function = DiceCELoss(
    to_onehot_y=True, 
    softmax=True,
    include_background=False  # 不计算背景类的损失
)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 滑动窗口推理也应该使用相同的尺寸
roi_size = patch_size  # 确保推理窗口尺寸匹配
sw_batch_size = 4

# 确保评估函数定义在训练函数之前
def evaluate_by_region(model, dataloader):
    """根据正确的标签分配评估各解剖区域的分割性能"""
    model.eval()
    
    # 正确的区域定义
    regions = {
        "sacrum": list(range(1, 11)),     # 骶骨区域: 标签1-10
        "left_hip": list(range(11, 21)),  # 左髋骨区域: 标签11-20
        "right_hip": list(range(21, 31))  # 右髋骨区域: 标签21-30
    }
    
    # 存储区域指标
    region_metrics = {region: [] for region in regions}
    total_dice = 0.0
    count = 0
    
    # 添加调试信息
    print("\n======== 评估开始 ========")
    
    with torch.no_grad():
        for val_idx, val_data in enumerate(dataloader):
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            # 打印输入和标签的形状以及值范围
            print(f"样本 {val_idx}:")
            print(f"  图像形状: {val_inputs.shape}, 范围: [{val_inputs.min().item():.2f}, {val_inputs.max().item():.2f}]")
            unique_labels = torch.unique(val_labels).cpu().numpy().tolist()
            print(f"  标签形状: {val_labels.shape}, 包含值: {unique_labels}")
            
            # 滑动窗口推理
            val_outputs = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model, overlap=0.7
            )
            
            # 获取预测类别
            val_pred = torch.argmax(val_outputs, dim=1)
            unique_preds = torch.unique(val_pred).cpu().numpy().tolist()
            print(f"  预测形状: {val_pred.shape}, 包含值: {unique_preds}")
            
            # 评估每个解剖区域
            sample_dice = 0.0
            sample_region_count = 0
            
            for region_name, label_range in regions.items():
                # 创建该区域的二值掩码
                region_pred = torch.zeros_like(val_pred, dtype=torch.float32)
                region_true = torch.zeros_like(val_labels, dtype=torch.float32)
                
                # 检查这个区域是否在真实标签中存在
                region_exists = False
                
                # 收集区域内所有标签
                for label in label_range:
                    # 加入预测掩码
                    pred_mask = (val_pred == label)
                    region_pred[pred_mask] = 1
                    
                    # 加入真实标签掩码
                    true_mask = (val_labels == label)
                    region_true[true_mask] = 1
                    
                    # 检查该标签是否存在
                    if true_mask.sum() > 0:
                        region_exists = True
                
                # 仅当区域在真实标签中存在时计算Dice
                if region_exists:
                    # 计算此区域的Dice分数
                    intersection = torch.sum(region_pred * region_true).item()
                    pred_sum = torch.sum(region_pred).item()
                    true_sum = torch.sum(region_true).item()
                    
                    # 调试输出区域统计
                    print(f"  {region_name}: 交集={intersection}, 预测总和={pred_sum}, 真实总和={true_sum}")
                    
                    if pred_sum + true_sum > 0:
                        dice = 2.0 * intersection / (pred_sum + true_sum)
                        region_metrics[region_name].append(dice)
                        sample_dice += dice
                        sample_region_count += 1
                        print(f"    Dice = {dice:.4f}")
                    else:
                        print(f"    跳过 (预测和真实总和为0)")
                else:
                    print(f"  {region_name}: 在此样本中不存在")
            
            # 计算此样本的平均Dice
            if sample_region_count > 0:
                sample_avg_dice = sample_dice / sample_region_count
                total_dice += sample_avg_dice
                count += 1
                print(f"  样本平均Dice: {sample_avg_dice:.4f}")
            else:
                print(f"  样本中没有可评估的区域")
    
    # 计算并显示每个区域的平均分数
    print("\n======== 区域评估结果 ========")
    
    region_avgs = {}
    for region, scores in region_metrics.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            region_avgs[region] = avg_score
            print(f"{region}区域Dice: {avg_score:.4f} (基于{len(scores)}个样本)")
        else:
            region_avgs[region] = 0.0
            print(f"{region}区域: 无评估数据")
    
    # 计算总体平均
    overall_avg = total_dice / count if count > 0 else 0.0
    print(f"所有样本平均Dice: {overall_avg:.4f}")
    print("======== 评估结束 ========\n")
    
    return overall_avg, region_avgs

# 训练函数，确保不进行标签映射
def train(global_step, train_loader, dice_val_best, global_step_best, max_iter=25000):
    """执行完整训练循环，不做标签映射"""
    model.train()
    epoch_loss = 0
    step = 0
    epoch = 0
    
    # 显示目标训练信息
    print(f"\n开始训练 - 目标迭代次数: {max_iter}")
    print(f"当前最佳Dice: {dice_val_best:.4f}, 在步骤: {global_step_best}")
    
    # 继续训练直到达到最大迭代次数
    while global_step < max_iter:
        epoch += 1
        epoch_loss = 0
        step = 0
        
        # 添加进度条显示
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="批次", 
                           mininterval=10.0)  # 增加最小更新间隔为10秒
        
        for batch_data in progress_bar:
            step += 1
            # 直接使用原始图像和标签，不进行映射
            x, y = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            # 前向传播
            outputs = model(x)
            
            # 使用原始标签计算损失，不需要映射
            loss = loss_function(outputs, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 仅在步数为100的倍数时更新进度条显示详情
            if global_step % 100 == 0:
                progress_bar.set_postfix({
                    '迭代': f"{global_step}/{max_iter}",
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_loss / step:.4f}"
                })
            
            # 保存损失记录
            if global_step % 200 == 0:  # 增加到每200步记录一次
                epoch_loss_values.append(epoch_loss / step)
                
            # 每1000步评估一次
            if global_step % 1000 == 0:
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 使用正确的评估函数，不进行标签映射
                dice_val, region_metrics = evaluate_by_region(model, val_loader)
                metric_values.append(dice_val)
                
                # 简化输出，只记录改进
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print(f"\n新最佳模型! 步骤: {global_step}, Dice: {dice_val:.4f}")
                
                # 打印评估信息（显示区域指标）
                print(f"\n评估 @ 步骤 {global_step}")
                print(f"总体Dice: {dice_val:.4f}, 最佳: {dice_val_best:.4f}")
                print("区域指标:")
                for region, score in region_metrics.items():
                    print(f"- {region}: {score:.4f}")
                
                # 每5000步打印一次当前的学习曲线
                if global_step % 5000 == 0:
                    plt.figure("train", (12, 6))
                    plt.subplot(1, 2, 1)
                    plt.title("训练平均损失")
                    x = [(i + 1) * 200 for i in range(len(epoch_loss_values))]
                    y = epoch_loss_values
                    plt.xlabel("迭代次数")
                    plt.plot(x, y)
                    plt.subplot(1, 2, 2)
                    plt.title("验证平均Dice")
                    x = [(i + 1) * 1000 for i in range(len(metric_values))]
                    y = metric_values
                    plt.xlabel("迭代次数")
                    plt.plot(x, y)
                    plt.savefig(os.path.join(root_dir, f"learning_curve_{global_step}.png"))
                    plt.close()
                
                # 恢复训练模式
                model.train()
                
                # 定期保存检查点，以防训练中断
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dice_val_best": dice_val_best,
                        "global_step_best": global_step_best,
                    },
                    os.path.join(root_dir, f"checkpoint_{global_step}.pth"),
                )
            
            # 每10个epoch显示一次平均损失
            if global_step % (10 * len(train_loader)) == 0:
                print(f"Epoch {epoch} - 平均损失: {epoch_loss / step:.4f}")
            
            # 达到最大迭代次数则提前结束
            if global_step >= max_iter:
                break
                
        # 每个epoch结束时清理GPU缓存
        torch.cuda.empty_cache()
    
    print(f"训练完成! 共执行{global_step}步，最佳Dice: {dice_val_best:.4f} @ 步骤 {global_step_best}")
    return global_step, dice_val_best, global_step_best

# 显示训练设备信息，保留这部分有用信息
print("\n\n==================================================")
print(f"训练设备: {device}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"CUDA版本: {torch.version.cuda}")
print("==================================================\n")

# 原来监视器代码之后的测试函数
def minimal_train_test():
    """最小化训练循环，只测试第一个批次"""
    print("开始最小化训练测试...")
    model.train()
    
    try:
        # 获取一个批次进行测试
        for batch_data in train_loader:
            x, y = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            # 简化输出，只显示关键信息
            print("执行测试批次前向传播...")
            outputs = model(x)
            
            print("计算损失...")
            loss = loss_function(outputs, y)
            print(f"损失值: {loss.item():.4f}")
            
            print("执行反向传播和优化器步骤...")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print("最小化训练测试成功完成!")
            return True
        
        print("警告: 没有找到训练数据!")
        return False
            
    except Exception as e:
        print(f"最小化训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 添加这行代码以支持Windows多进程
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 显示训练设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"训练设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: 没有可用的GPU，训练将在CPU上进行(速度会很慢!)")
    print(f"{'='*50}\n")
    
    # 将主要执行代码移到这里
    max_iterations = 25000  # 减少迭代次数以便更快完成训练测试
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    # 注释掉所有监视器相关代码
    """
    import threading
    import time

    # 添加监视器线程检测死锁
    def monitor_thread():
        #监视主线程，如果长时间无进展则打印堆栈信息
        import sys
        import traceback
        
        start_time = time.time()
        while True:
            time.sleep(30)  # 每30秒检查一次
            elapsed = time.time() - start_time
            print(f"\n[监视器] 已等待 {elapsed:.1f} 秒无明显进展")
            
            # 打印所有线程的堆栈信息
            print("\n[监视器] 当前所有线程堆栈信息:")
            for th in threading.enumerate():
                print(f"\n线程 {th.name}:")
                traceback.print_stack(sys._current_frames()[th.ident])
            
            if elapsed > 300:  # 5分钟后提供建议
                print("\n[监视器] 程序可能已死锁。建议:")
                print("1. 按Ctrl+C中断程序")
                print("2. 尝试以下修复方案:")
                print("   - 设置num_workers=0")
                print("   - 移除CacheDataset")
                print("   - 创建更简单的训练循环")

    # 启动监视器线程
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    print("监视器线程已启动，将监控程序进展...")
    """
    
    print("\n===== 尝试最小化训练测试 =====")
    if minimal_train_test():
        print("基础功能测试成功，继续完整训练...")
        # 修复: 传入最大迭代次数参数
        train_results = train(0, train_loader, 0, 0, max_iterations)
        
        print(f"训练返回结果: {train_results}")
        
        # 正确解析三个返回值
        iterations, dice_val_best, global_step_best = train_results
        
        print(f"训练完成，总迭代次数: {iterations}")
        print(f"最佳指标: {dice_val_best:.4f}，在迭代步骤: {global_step_best}")
        
        # 强制保存最终模型，无论性能如何
        print("强制保存最终模型...")
        model_file = os.path.join(root_dir, "final_model.pth")
        torch.save(model.state_dict(), model_file)
        print(f"已保存最终模型至: {model_file}")
        
        # 检查最终模型文件
        if os.path.exists(model_file):
            print(f"确认: 最终模型文件已成功保存")
            
            # 加载最终模型用于评估
            model.load_state_dict(torch.load(model_file))
            print(f"已加载最终模型进行评估")
            
            # 确保使用正确的滑动窗口大小
            val_inputs = torch.unsqueeze(img, 1).cuda()
            val_labels = torch.unsqueeze(label, 1).cuda()
            
            # 使用与模型定义一致的patch_size
            val_outputs = sliding_window_inference(
                val_inputs, 
                roi_size=patch_size,  # 使用与模型定义相同的尺寸
                sw_batch_size=1,      # 减小批次大小从4到1
                predictor=model, 
                overlap=0.5           # 减小重叠度从0.8到0.5
            )
            
            # 模型评估和可视化
            plt.figure("train", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Iteration Average Loss")
            x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
            y = epoch_loss_values
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.subplot(1, 2, 2)
            plt.title("Val Mean Dice")
            x = [eval_num * (i + 1) for i in range(len(metric_values))]
            y = metric_values
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.show()

            case_num = 4
            model.eval()
            with torch.no_grad():
                img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
                img = val_ds[case_num]["image"]
                label = val_ds[case_num]["label"]
                val_inputs = torch.unsqueeze(img, 1).cuda()
                val_labels = torch.unsqueeze(label, 1).cuda()
                val_outputs = sliding_window_inference(val_inputs, patch_size, 1, model, overlap=0.5)
                plt.figure("check", (18, 6))
                plt.subplot(1, 3, 1)
                plt.title("image")
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title("label")
                plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
                plt.subplot(1, 3, 3)
                plt.title("output")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
                plt.show()
        else:
            print(f"错误: 最终模型文件未能保存: {model_file}")
            print("请检查磁盘空间和写入权限")
        
        # 单独检查是否有最佳模型(如果Dice分数大于0)
        if dice_val_best > 0:
            best_model_file = os.path.join(root_dir, "best_metric_model.pth")
            if os.path.exists(best_model_file):
                print(f"最佳模型也已保存: {best_model_file}")
    else:
        print("基础功能测试失败，跳过完整训练...")
        print("建议检查数据和模型兼容性问题")

def visualize_prediction(model, val_data, index=0):
    """可视化模型预测结果，使用原始标签值（0-30）"""
    model.eval()
    
    # 生成颜色映射, 0为背景(黑色), 其他按区域分组
    colors = plt.cm.get_cmap('tab20', 30)  # 使用tab20色图获取20种颜色
    
    with torch.no_grad():
        # 获取单个样本
        val_inputs = val_data["image"].to(device)
        val_labels = val_data["label"].to(device)
        
        # 滑动窗口推理
        val_outputs = sliding_window_inference(
            val_inputs, roi_size, sw_batch_size, model, overlap=0.7
        )
        
        # 获取预测类别
        val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
        # 准备显示数据
        image = val_inputs[0, 0].cpu().numpy()
        label = val_labels[0, 0].cpu().numpy()
        pred = val_pred[0]
        
        # 获取中心切片
        z_idx = image.shape[2] // 2
        
        # 显示结果
        plt.figure("预测结果", (18, 6))
        
        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.title("原始图像")
        plt.imshow(image[:, :, z_idx], cmap="gray")
        plt.colorbar(shrink=0.8)
        
        # 显示真实标签，使用自定义颜色映射
        plt.subplot(1, 3, 2)
        plt.title("真实标签")
        masked_label = np.ma.masked_where(label[:, :, z_idx] == 0, label[:, :, z_idx])
        plt.imshow(image[:, :, z_idx], cmap="gray")
        # 使用不同颜色显示不同区域
        sacrum_mask = np.ma.masked_where((label[:, :, z_idx] < 1) | (label[:, :, z_idx] > 10), label[:, :, z_idx])
        left_hip_mask = np.ma.masked_where((label[:, :, z_idx] < 11) | (label[:, :, z_idx] > 20), label[:, :, z_idx])
        right_hip_mask = np.ma.masked_where((label[:, :, z_idx] < 21) | (label[:, :, z_idx] > 30), label[:, :, z_idx])
        
        plt.imshow(sacrum_mask, cmap='Reds', alpha=0.7)
        plt.imshow(left_hip_mask, cmap='Greens', alpha=0.7)
        plt.imshow(right_hip_mask, cmap='Blues', alpha=0.7)
        plt.colorbar(shrink=0.8)
        
        # 显示预测结果
        plt.subplot(1, 3, 3)
        plt.title("模型预测")
        masked_pred = np.ma.masked_where(pred[:, :, z_idx] == 0, pred[:, :, z_idx])
        plt.imshow(image[:, :, z_idx], cmap="gray")
        
        # 使用不同颜色显示不同预测区域
        sacrum_pred = np.ma.masked_where((pred[:, :, z_idx] < 1) | (pred[:, :, z_idx] > 10), pred[:, :, z_idx])
        left_hip_pred = np.ma.masked_where((pred[:, :, z_idx] < 11) | (pred[:, :, z_idx] > 20), pred[:, :, z_idx])
        right_hip_pred = np.ma.masked_where((pred[:, :, z_idx] < 21) | (pred[:, :, z_idx] > 30), pred[:, :, z_idx])
        
        plt.imshow(sacrum_pred, cmap='Reds', alpha=0.7)
        plt.imshow(left_hip_pred, cmap='Greens', alpha=0.7)
        plt.imshow(right_hip_pred, cmap='Blues', alpha=0.7)
        plt.colorbar(shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"prediction_vis_{index}.png"), dpi=300)
        plt.show()