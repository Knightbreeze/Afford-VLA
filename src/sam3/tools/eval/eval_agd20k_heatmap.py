import torch
import json
import gc
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configuration
JSONL_PATH = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/agd20k_unseen_testset.jsonl"
EGOCENTRIC_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/egocentric"
OUTPUT_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/result"
HEATMAP_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/heatmap"
HEATMAP_RESULT_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/heatmap_result"


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_prompt_from_jsonl(affordance_label, object_name):
    """从jsonl文件中查找对应的sam3_prompt"""
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            if record['affordance_lable'] == affordance_label and record['object_name'] == object_name:
                return record['sam3_prompt']
    raise ValueError(f"No prompt found for {affordance_label}/{object_name}")


def save_heatmap_overlay(image, heatmap_logits, save_path, apply_sigmoid=True, smooth_sigma=6, threshold=0.2):
    """
    保存热力图叠加在原图上的可视化结果
    
    Args:
        image: 原始图像 (PIL.Image)
        heatmap_logits: 热力图logits
        save_path: 保存路径
        apply_sigmoid: 是否应用sigmoid
        smooth_sigma: 高斯平滑的sigma值
        threshold: 阈值，低于此值的部分将显示为透明
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if isinstance(heatmap_logits, torch.Tensor):
        heatmap_logits = heatmap_logits.cpu().numpy()
    
    # 应用sigmoid
    if apply_sigmoid:
        heatmap_probs = 1 / (1 + np.exp(-heatmap_logits))
    else:
        heatmap_probs = heatmap_logits
    
    # 高斯平滑
    if smooth_sigma > 0:
        heatmap_probs = gaussian_filter(heatmap_probs, sigma=smooth_sigma)
    
    # 创建带阈值的热力图
    high_heat_min = 0.7
    high_heat_max = 0.95
    heatmap_display = np.zeros_like(heatmap_probs)
    mask_above_threshold = heatmap_probs >= threshold
    
    if mask_above_threshold.any():
        values_above = heatmap_probs[mask_above_threshold]
        normalized = (values_above - threshold) / (heatmap_probs.max() - threshold + 1e-8)
        heatmap_display[mask_above_threshold] = high_heat_min + normalized * (high_heat_max - high_heat_min)
    
    # 再次平滑
    if smooth_sigma > 0:
        heatmap_probs_smooth = gaussian_filter(heatmap_display, sigma=smooth_sigma * 1.5)
    else:
        heatmap_probs_smooth = heatmap_display
    
    # 创建透明度mask
    alpha_mask_raw = np.where(heatmap_probs_smooth >= threshold * 0.5,
                               np.clip((heatmap_probs_smooth - threshold * 0.5) / (threshold * 0.5), 0, 1),
                               0)
    if smooth_sigma > 0:
        alpha_mask = gaussian_filter(alpha_mask_raw, sigma=smooth_sigma * 0.5)
    else:
        alpha_mask = alpha_mask_raw
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 热力图
    im = axes[1].imshow(heatmap_probs_smooth, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Heatmap (threshold={threshold}, sigma={smooth_sigma})', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加
    axes[2].imshow(image_np)
    im2 = axes[2].imshow(heatmap_probs_smooth, cmap='jet', alpha=alpha_mask * 0.8, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap overlay to {save_path}")


def save_single_heatmap_overlay(image, heatmap_logits, save_path, apply_sigmoid=True, smooth_sigma=1.0, threshold=0.2):
    """
    保存单张热力图叠加在原图上的结果（不包含子图）
    
    Args:
        image: 原始图像 (PIL.Image)
        heatmap_logits: 热力图logits
        save_path: 保存路径
        apply_sigmoid: 是否应用sigmoid
        smooth_sigma: 高斯平滑的sigma值
        threshold: 阈值
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if isinstance(heatmap_logits, torch.Tensor):
        heatmap_logits = heatmap_logits.cpu().numpy()
    
    # 应用sigmoid
    if apply_sigmoid:
        heatmap_probs = 1 / (1 + np.exp(-heatmap_logits))
    else:
        heatmap_probs = heatmap_logits
    
    # 高斯平滑
    if smooth_sigma > 0:
        heatmap_probs = gaussian_filter(heatmap_probs, sigma=smooth_sigma)
    
    # 创建带阈值的热力图
    high_heat_min = 0.7
    high_heat_max = 0.9
    heatmap_display = np.zeros_like(heatmap_probs)
    mask_above_threshold = heatmap_probs >= threshold
    
    if mask_above_threshold.any():
        values_above = heatmap_probs[mask_above_threshold]
        normalized = (values_above - threshold) / (heatmap_probs.max() - threshold + 1e-8)
        heatmap_display[mask_above_threshold] = high_heat_min + normalized * (high_heat_max - high_heat_min)
    
    # 再次平滑
    if smooth_sigma > 0:
        heatmap_probs_smooth = gaussian_filter(heatmap_display, sigma=smooth_sigma * 1.5)
    else:
        heatmap_probs_smooth = heatmap_display
    
    # 创建透明度mask
    alpha_mask_raw = np.where(heatmap_probs_smooth >= threshold * 0.5,
                               np.clip((heatmap_probs_smooth - threshold * 0.5) / (threshold * 0.5), 0, 1),
                               0)
    if smooth_sigma > 0:
        alpha_mask = gaussian_filter(alpha_mask_raw, sigma=smooth_sigma * 0.5)
    else:
        alpha_mask = alpha_mask_raw
    
    # 创建单张图像（只有叠加效果）
    fig = plt.figure(figsize=(image_np.shape[1]/100, image_np.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # 显示原图和热力图叠加
    ax.imshow(image_np)
    ax.imshow(heatmap_probs_smooth, cmap='jet', alpha=alpha_mask * 0.8, vmin=0, vmax=1)
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved single heatmap overlay to {save_path}")


def save_heatmap_as_grayscale(heatmap_logits, save_path, apply_sigmoid=True, smooth_sigma=15.0):
    """
    将热力图保存为0-255灰度PNG图片（保持原图大小）
    
    Args:
        heatmap_logits: 热力图logits
        save_path: 保存路径
        apply_sigmoid: 是否应用sigmoid
        smooth_sigma: 高斯平滑参数
    """
    if isinstance(heatmap_logits, torch.Tensor):
        heatmap_logits = heatmap_logits.cpu().numpy()
    
    # 应用sigmoid得到概率值
    if apply_sigmoid:
        heatmap_probs = 1 / (1 + np.exp(-heatmap_logits))
    else:
        heatmap_probs = heatmap_logits
    
    # 高斯模糊
    if smooth_sigma > 0:
        heatmap_probs = gaussian_filter(heatmap_probs, sigma=smooth_sigma)
    
    # 归一化到0-1
    heatmap_min = heatmap_probs.min()
    heatmap_max = heatmap_probs.max()
    if heatmap_max > heatmap_min:
        heatmap_normalized = (heatmap_probs - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = heatmap_probs
    
    # 转换为0-255灰度值
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
    
    # 转换为PIL图像（保持原始大小）
    heatmap_image = Image.fromarray(heatmap_uint8, mode='L')
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_image.save(save_path)
    print(f"Saved heatmap to {save_path}")


def single_image_process(processor, affordance_label, object_name, img_name):
    """
    处理单张图片
    
    Args:
        processor: Sam3Processor实例
        affordance_label: 行为标签，如 "hit"
        object_name: 物体名称，如 "axe"
        img_name: 图片文件名，如 "axe_000001.jpg"
    """
    # 获取prompt
    prompt = load_prompt_from_jsonl(affordance_label, object_name)
    if not prompt:
        print(f"Warning: Empty prompt for {affordance_label}/{object_name}, skipping...")
        return
    
    # 构建图片路径
    image_dir = Path(EGOCENTRIC_DIR) / affordance_label / object_name
    image_path = image_dir / img_name
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # 加载图片
    image = Image.open(image_path)
    
    # 推理
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    # 获取热力图
    full_heatmap = output['full_heatmap_logits']
    
    # 保存灰度图
    # output_path = Path(OUTPUT_DIR) / affordance_label / object_name / img_name.replace('.jpg', '.png')
    # save_heatmap_as_grayscale(
    #     heatmap_logits=full_heatmap,
    #     save_path=output_path,
    #     apply_sigmoid=False,
    # )
    
    # 保存热力图叠加可视化
    heatmap_path = Path(HEATMAP_DIR) / affordance_label / object_name / img_name.replace('.jpg', '.png')
    save_heatmap_overlay(
        image=image,
        heatmap_logits=full_heatmap,
        save_path=heatmap_path,
        apply_sigmoid=False,
        smooth_sigma=2.0,
        threshold=0.2
    )
    
    # 保存单张热力图叠加结果
    heatmap_result_path = Path(HEATMAP_RESULT_DIR) / affordance_label / object_name / img_name.replace('.jpg', '.png')
    save_single_heatmap_overlay(
        image=image,
        heatmap_logits=full_heatmap,
        save_path=heatmap_result_path,
        apply_sigmoid=False,
        smooth_sigma=2.0,
        threshold=0.2
    )
    
    print(f"Processed: {affordance_label}/{object_name}/{img_name}")
    
    # 清理内存
    del output, full_heatmap, inference_state, image
    clear_gpu_memory()


def batch_process(processor, affordance_label, object_name):
    """
    批量处理整个文件夹下的所有图片
    
    Args:
        processor: Sam3Processor实例
        affordance_label: 行为标签，如 "hit"
        object_name: 物体名称，如 "axe"
    """
    # 获取prompt
    prompt = load_prompt_from_jsonl(affordance_label, object_name)
    if not prompt:
        print(f"Warning: Empty prompt for {affordance_label}/{object_name}, skipping...")
        return
    
    # 构建图片文件夹路径
    image_dir = Path(EGOCENTRIC_DIR) / affordance_label / object_name
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # 获取所有jpg图片
    image_files = sorted(list(image_dir.glob("*.jpg")))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nProcessing {affordance_label}/{object_name} - {len(image_files)} images")
    print(f"Prompt: '{prompt}'")
    
    # 批量处理
    for idx, img_file in enumerate(image_files, 1):
        try:
            # 加载图片
            image = Image.open(img_file)
            
            # 推理
            inference_state = processor.set_image(image)
            
            with torch.no_grad():
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            # 获取热力图
            full_heatmap = output['full_heatmap_logits']
            
            # # 保存灰度图
            # output_path = Path(OUTPUT_DIR) / affordance_label / object_name / img_file.name.replace('.jpg', '.png')
            # save_heatmap_as_grayscale(
            #     heatmap_logits=full_heatmap,
            #     save_path=output_path,
            #     apply_sigmoid=False,
            # )
            
            # 保存热力图叠加可视化
            heatmap_path = Path(HEATMAP_DIR) / affordance_label / object_name / img_file.name.replace('.jpg', '.png')
            save_heatmap_overlay(
                image=image,
                heatmap_logits=full_heatmap,
                save_path=heatmap_path,
                apply_sigmoid=False,
                threshold=0.5
            )
            
            # 保存单张热力图叠加结果
            heatmap_result_path = Path(HEATMAP_RESULT_DIR) / affordance_label / object_name / img_file.name.replace('.jpg', '.png')
            save_single_heatmap_overlay(
                image=image,
                heatmap_logits=full_heatmap,
                save_path=heatmap_result_path,
                apply_sigmoid=False,
                smooth_sigma=2.0,
                threshold=0.5
            )
            
            # 清理内存
            del output, full_heatmap, inference_state, image
            
            # 每处理5张图片清理一次GPU缓存
            if idx % 3 == 0:
                clear_gpu_memory()
                print(f"  Progress: {idx}/{len(image_files)} - GPU memory cleared")
        
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
            clear_gpu_memory()
            continue
    
    # 最后清理一次
    clear_gpu_memory()
    print(f"Completed: {affordance_label}/{object_name} - {len(image_files)} images processed")


def process_all_from_jsonl(processor):
    """从jsonl文件读取所有记录并批量处理"""
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            affordance_label = record['affordance_lable']
            object_name = record['object_name']
            
            try:
                batch_process(processor, affordance_label, object_name)
            except Exception as e:
                print(f"Error processing {affordance_label}/{object_name}: {e}")
                continue


if __name__ == "__main__":
    # 加载模型
    model = build_sam3_image_model(mode="sam3")
    processor = Sam3Processor(model)
    
    affordance_label = "wash"
    object_name = "knife"
    # img_name = f"{object_name}_000108.jpg"

    # 示例1: 处理单张图片
    # single_image_process(processor, affordance_label, object_name, img_name)
    
    # 示例2: 批量处理某个对象的所有图片
    # batch_process(processor, affordance_label, object_name)
    
    # 示例3: 处理jsonl中的所有对象
    process_all_from_jsonl(processor)
