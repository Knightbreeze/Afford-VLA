import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

# Configuration
GT_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/GT"
EGOCENTRIC_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/egocentric"
OUTPUT_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/GT_heatmap"


def overlay_gt_mask_as_heatmap(image, mask, save_path, smooth_sigma=2.0):
    """
    将GT掩码以热力图方式叠加到原图上
    
    Args:
        image: 原始图像 (PIL.Image 或 numpy array)
        mask: GT掩码 (PIL.Image 或 numpy array)
        save_path: 保存路径
        smooth_sigma: 高斯平滑参数
        threshold: 显示阈值
    """
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask).astype(np.float32) / 255.0
    else:
        mask_np = mask.astype(np.float32)
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0
    
    # 高斯平滑
    if smooth_sigma > 0:
        mask_smooth = gaussian_filter(mask_np, sigma=smooth_sigma)
    else:
        mask_smooth = mask_np
    
    # 归一化到0-1
    if mask_smooth.max() > mask_smooth.min():
        heatmap_probs_smooth = (mask_smooth - mask_smooth.min()) / (mask_smooth.max() - mask_smooth.min())
    else:
        heatmap_probs_smooth = mask_smooth
    
    # 创建透明度mask
    alpha_mask = np.clip(heatmap_probs_smooth, 0, 1)
    
    # 创建单张图像（只保存叠加效果）
    fig = plt.figure(figsize=(image_np.shape[1]/100, image_np.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # 显示原图和热力图叠加
    ax.imshow(image_np)
    ax.imshow(heatmap_probs_smooth, cmap='jet', alpha=alpha_mask * 0.9, vmin=0, vmax=1)
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved GT heatmap overlay to {save_path}")


def process_single_pair(affordance_label, object_name, filename):
    """
    处理单对图片和掩码
    
    Args:
        affordance_label: 行为标签
        object_name: 物体名称
        filename: 文件名（不含扩展名）
    """
    # 构建路径
    ego_path = Path(EGOCENTRIC_DIR) / affordance_label / object_name / f"{filename}.jpg"
    gt_path = Path(GT_DIR) / affordance_label / object_name / f"{filename}.png"
    output_path = Path(OUTPUT_DIR) / affordance_label / object_name / f"{filename}.png"
    
    # 检查文件是否存在
    if not ego_path.exists():
        print(f"Warning: Egocentric image not found: {ego_path}")
        return
    
    if not gt_path.exists():
        print(f"Warning: GT mask not found: {gt_path}")
        return
    
    # 加载图片和掩码
    image = Image.open(ego_path)
    mask = Image.open(gt_path)
    
    # 生成叠加热力图
    overlay_gt_mask_as_heatmap(
        image=image,
        mask=mask,
        save_path=output_path,
        smooth_sigma=0.0,
    )
    
    print(f"Processed: {affordance_label}/{object_name}/{filename}")


def process_all_gt_masks():
    """处理GT目录下的所有掩码"""
    gt_dir = Path(GT_DIR)
    
    # 遍历所有affordance标签
    for affordance_folder in sorted(gt_dir.iterdir()):
        if not affordance_folder.is_dir():
            continue
        
        affordance_label = affordance_folder.name
        
        # 遍历所有物体
        for object_folder in sorted(affordance_folder.iterdir()):
            if not object_folder.is_dir():
                continue
            
            object_name = object_folder.name
            
            print(f"\n{'='*60}")
            print(f"Processing: {affordance_label}/{object_name}")
            print(f"{'='*60}")
            
            # 获取所有GT掩码文件
            gt_files = sorted(list(object_folder.glob("*.png")))
            
            if len(gt_files) == 0:
                print(f"No GT masks found in {object_folder}")
                continue
            
            print(f"Found {len(gt_files)} GT masks")
            
            # 处理每个掩码
            for idx, gt_file in enumerate(gt_files, 1):
                filename = gt_file.stem  # 文件名（不含扩展名）
                
                try:
                    process_single_pair(affordance_label, object_name, filename)
                    
                    # 每处理10张图片显示进度
                    if idx % 10 == 0:
                        print(f"  Progress: {idx}/{len(gt_files)}")
                
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    continue
            
            print(f"Completed: {affordance_label}/{object_name} - {len(gt_files)} images processed")
    
    print(f"\n{'='*60}")
    print("All GT masks processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 处理所有GT掩码
    process_all_gt_masks()
    
    # 或者处理单个样本
    # process_single_pair("hit", "axe", "axe_000108")
