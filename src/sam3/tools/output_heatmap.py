import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def enhance_contrast(heatmap, gamma=1.5):
    """通过 gamma 校正增强对比度"""
    return np.power(heatmap, gamma)

def visualize_full_heatmap(image, heatmap_logits, save_path=None, apply_sigmoid=True, smooth_sigma=1):
    """
    可视化全图热力图
    
    Args:
        image: 原始图像
        heatmap_logits: 热力图数据
        save_path: 保存路径
        apply_sigmoid: 是否应用sigmoid
        smooth_sigma: 高斯平滑的sigma值，越大越平滑（0表示不平滑）
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(heatmap_logits, torch.Tensor):
        heatmap_logits = heatmap_logits.cpu().numpy()
    
    if apply_sigmoid:
        heatmap_probs = 1 / (1 + np.exp(-heatmap_logits))
    else:
        heatmap_probs = heatmap_logits

    print(f"Logits range: [{heatmap_logits.min():.3f}, {heatmap_logits.max():.3f}]")

    # 高斯平滑处理，创建渐变效果
    if smooth_sigma > 0:
        heatmap_probs_smooth = gaussian_filter(heatmap_probs, sigma=smooth_sigma)
        # heatmap_probs_smooth = enhance_contrast(heatmap_probs_smooth, gamma=1.5)
        print(f"After smoothing (sigma={smooth_sigma}): [{heatmap_probs_smooth.min():.4f}, {heatmap_probs_smooth.max():.4f}]")
    else:
        heatmap_probs_smooth = heatmap_probs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 热力图
    im = axes[1].imshow(heatmap_probs_smooth, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Full Heatmap', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加
    axes[2].imshow(image)
    im2 = axes[2].imshow(heatmap_probs_smooth, cmap='jet', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.001)

Image_Path = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000000/frame_0002.png"
Prompt = "the skillet handle"

image = Image.open(Image_Path)
inference_state = processor.set_image(image)

output = processor.set_text_prompt(state=inference_state, prompt=Prompt)
full_heatmap = output['full_heatmap_logits']

# 可视化热力图
visualize_full_heatmap(
    image=image,
    heatmap_logits=full_heatmap,
    save_path='/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/output_heatmap/output_heatmap.png',
    apply_sigmoid=False
)

print("\n✓ Visualization complete!")