import matplotlib.pyplot as plt
import numpy as np

#################################### For Image ####################################
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# Load the model
# model = build_sam3_image_model()
# processor = Sam3Processor(model)


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
        print(
            f"After smoothing (sigma={smooth_sigma}): [{heatmap_probs_smooth.min():.4f}, {heatmap_probs_smooth.max():.4f}]"
        )
    else:
        heatmap_probs_smooth = heatmap_probs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # 热力图
    im = axes[1].imshow(heatmap_probs_smooth, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Full Heatmap", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加
    axes[2].imshow(image)
    im2 = axes[2].imshow(heatmap_probs_smooth, cmap="jet", alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def single_image_example(processor):
    test_image_path = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000000/2.png"

    Prompt = "the interior surface of the skillet"

    image = Image.open(test_image_path)
    inference_state = processor.set_image(image)

    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=Prompt)

    full_heatmap = output["full_heatmap_logits"]

    visualize_full_heatmap(
        image=image,
        heatmap_logits=full_heatmap,
        save_path="/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/output_heatmap/2.png",
        apply_sigmoid=False,
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
    batch_prompts = ["grasp the handle", "pick up the cup", "press the button", "open the drawer"]

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
            aff_output = processor.set_text_prompt_batch(prompts=batch_prompts, state=inference_state)

    # 提取用于对齐的特征
    aff_hidden = aff_output["encoder_hidden_states"]  # [B, N, D]
    prompt_features = aff_output["prompt_features"]  # [B, L, D]

    print(f"Affordance hidden shape: {aff_hidden.shape}")
    print(f"Prompt features shape: {prompt_features.shape}")
    print(f"Feature dtype: {aff_hidden.dtype}")
    print(f"Feature device: {aff_hidden.device}")

    # 这些特征可以直接用于对齐损失计算
    # align_loss = align_projector(vision_hidden, aff_hidden, align_mask)

    return aff_output


# ============ 测试不同 batch size ============
def test_different_batch_sizes(processor):
    """测试不同批次大小的性能"""

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
    full_heatmap = output["full_heatmap_logits"][1]

    visualize_full_heatmap(
        image=image,
        heatmap_logits=full_heatmap,
        save_path="/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/output_heatmap/output_heatmap2.png",
        apply_sigmoid=False,
    )


if __name__ == "__main__":
    # model = build_sam3_image_model(mode="affordance")
    # processor = Sam3AffProcessor(model)

    model = build_sam3_image_model(mode="sam3")
    processor = Sam3Processor(model)

    # Run single image example
    single_image_example(processor)

    # test_different_batch_sizes(processor)
