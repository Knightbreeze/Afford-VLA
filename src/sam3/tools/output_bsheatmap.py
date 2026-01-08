import torch
#################################### For Image ####################################
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.model.sam3_image_processor_aff_memory import Sam3AffProcessor

# Load the model
# model = build_sam3_image_model()
# processor = Sam3Processor(model)

def enhance_contrast(heatmap, gamma=1.5):
    """通过 gamma 校正增强对比度"""
    return np.power(heatmap, gamma)


def visualize_full_heatmap(image, heatmap_logits, save_path=None, apply_sigmoid=True, smooth_sigma=1, threshold=0.2):
    """
    可视化全图热力图
    
    Args:
        image: 原始图像
        heatmap_logits: 热力图数据
        save_path: 保存路径
        apply_sigmoid: 是否应用sigmoid
        smooth_sigma: 高斯平滑的sigma值，越大越平滑（0表示不平滑）
        threshold: 阈值，低于此值的部分将显示为透明（0-1之间，默认0.3）
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

    # 第一步：先对原始热力图进行平滑，减少噪声
    if smooth_sigma > 0:
        heatmap_probs = gaussian_filter(heatmap_probs, sigma=smooth_sigma)
        print(f"After initial smoothing (sigma={smooth_sigma}): [{heatmap_probs.min():.4f}, {heatmap_probs.max():.4f}]")

    # 创建带阈值的热力图：超过阈值的部分映射到高热度区域
    # 设置高热度范围的最小值和最大值
    high_heat_min = 0.7
    high_heat_max = 0.9

    # 创建处理后的热力图
    heatmap_display = np.zeros_like(heatmap_probs)
    mask_above_threshold = heatmap_probs >= threshold
    
    # 低于阈值的部分保持为0（透明）
    # 高于阈值的部分重新归一化到高热度范围
    if mask_above_threshold.any():
        values_above = heatmap_probs[mask_above_threshold]
        # 将超过阈值的值归一化到 [high_heat_min, high_heat_max] 范围
        normalized = (values_above - threshold) / (heatmap_probs.max() - threshold)
        heatmap_display[mask_above_threshold] = high_heat_min + normalized * (high_heat_max - high_heat_min)
    
    # 第二步：对处理后的热力图再次平滑，使边缘更柔和
    if smooth_sigma > 0:
        heatmap_probs_smooth = gaussian_filter(heatmap_display, sigma=smooth_sigma * 1.5)
        print(f"After final smoothing (sigma={smooth_sigma * 1.5}): [{heatmap_probs_smooth.min():.4f}, {heatmap_probs_smooth.max():.4f}]")
    else:
        heatmap_probs_smooth = heatmap_display
    
    # 创建平滑的透明度mask：使用渐变而不是硬边界
    # 对alpha mask也进行平滑处理，创建柔和的边缘
    alpha_mask_raw = np.where(heatmap_probs_smooth >= threshold * 0.5, 
                               np.clip((heatmap_probs_smooth - threshold * 0.5) / (threshold * 0.5), 0, 1), 
                               0)
    if smooth_sigma > 0:
        alpha_mask = gaussian_filter(alpha_mask_raw, sigma=smooth_sigma * 0.5)
    else:
        alpha_mask = alpha_mask_raw

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 热力图 - 使用平滑后的高热度显示
    im = axes[1].imshow(heatmap_probs_smooth, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Full Heatmap (threshold={threshold}, high heat, sigma={smooth_sigma})', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加 - 使用自定义alpha通道
    axes[2].imshow(image)
    # 使用alpha_mask来控制透明度
    im2 = axes[2].imshow(heatmap_probs_smooth, cmap='jet', alpha=alpha_mask * 0.8, vmin=0, vmax=1)
    axes[2].set_title('Overlay (with threshold)', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def single_image_example(processor):
    test_image_path = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000006/2.png"

    Prompt = "the surface of the skillet"

    image = Image.open(test_image_path)
    inference_state = processor.set_image(image)

    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=Prompt)

    full_heatmap = output['full_heatmap_logits']
    
    visualize_full_heatmap(
        image=image,
        heatmap_logits=full_heatmap,
        save_path='/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/output_heatmap/2.png',
        apply_sigmoid=False
    )
    # Get the masks, bounding boxes, and scores
    # masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    # plot_results(image, inference_state)

# ============ 用于训练的批处理示例 ============
def training_batch_example():
    """
    模拟训练时的批处理用法
    展示如何在训练循环中使用 Sam3AffProcessor
    """
    import torch
    from torchvision.transforms import ToPILImage
    
    print("\n=== Training Batch Example ===")
    
    # 模拟从 DataLoader 获取的批次数据
    # 假设图像已经是 tensor 格式 [B, C, H, W]
    batch_size = 4
    batch_image_tensors = torch.randn(batch_size, 3, 480, 640)  # 模拟数据
    
    # 从 JSON 加载的 prompts
    batch_prompts = [
        "grasp the handle",
        "pick up the cup", 
        "press the button",
        "open the drawer"
    ]
    
    # 转换 tensor 为 PIL images
    to_pil = ToPILImage()
    pil_images = []
    for i in range(batch_size):
        img_tensor = batch_image_tensors[i]
        # 归一化到 [0, 1] (如果需要)
        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1.0) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)
        pil_img = to_pil(img_tensor.cpu())
        pil_images.append(pil_img)
    
    # 批量推理 (不计算梯度)
    with torch.no_grad():
        # 设置图像批次
        inference_state = processor.set_image_batch(pil_images)
        
        # 设置文本提示批次
        with torch.autocast("cuda", dtype=torch.bfloat16):
            aff_output = processor.set_text_prompt_batch(
                prompts=batch_prompts,
                state=inference_state
            )
    
    # 提取用于对齐的特征
    aff_hidden = aff_output["encoder_hidden_states"]  # [B, N, D]
    prompt_features = aff_output["prompt_features"]    # [B, L, D]
    
    print(f"Affordance hidden shape: {aff_hidden.shape}")
    print(f"Prompt features shape: {prompt_features.shape}")
    print(f"Feature dtype: {aff_hidden.dtype}")
    print(f"Feature device: {aff_hidden.device}")
    
    # 这些特征可以直接用于对齐损失计算
    # align_loss = align_projector(vision_hidden, aff_hidden, align_mask)
    
    return aff_output


# ============ 测试不同 batch size ============
def test_different_batch_sizes(processor):
    
    print("\n=== Testing Different Batch Sizes ===")
    
    # 准备测试图片
    test_image_path = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000000/frame_0002.png"
    base_image = Image.open(test_image_path)
    
    batch_size = 2
    
    images = [base_image] * batch_size
    prompts = ["the right yellow part"] * batch_size
    
    with torch.no_grad():
        inference_states = processor.set_image_batch(images)
        output = processor.set_text_prompt_batch(prompts=prompts, states=inference_states)

    image = images[1]
    full_heatmap = output['full_heatmap_logits'][1]
    
    visualize_full_heatmap(
        image=image,
        heatmap_logits=full_heatmap,
        save_path='/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/output_heatmap/output_heatmap2.png',
        apply_sigmoid=False
    )


if __name__ == "__main__":
    # model = build_sam3_image_model(mode="affordance")
    # processor = Sam3AffProcessor(model)

    model = build_sam3_image_model(mode="sam3")
    processor = Sam3Processor(model)


    # Run single image example
    single_image_example(processor)

    # test_different_batch_sizes(processor)