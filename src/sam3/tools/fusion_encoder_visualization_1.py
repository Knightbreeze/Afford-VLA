"""
注意力特征可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import os
from scipy.ndimage import gaussian_filter

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==================== 用户配置区域 ====================
# 1. 输入图片路径
IMAGE_PATH = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000006/2.png"

# 2. 文本提示词
TEXT_PROMPT = "the interior surface of the skillet"

# 3. 模型路径
CHECKPOINT_PATH = "/home/nightbreeze/research/ckpt/sam3/sam3.pt"

# 4. 输出目录
OUTPUT_DIR = "/home/nightbreeze/research/Code/AffVLA/affvla/src/sam3/output/fusion_encoder_heatmap/2"

# 5. 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6. 可视化选项
VISUALIZE_ALL_LAYERS = True  # 是否可视化所有层
NUM_LAYERS_TO_SHOW = 6       # 显示的层数

# 7. 热力图处理选项
INVERT_HEATMAP = True        # 是否反转热力图（True=低值变高值，蓝变红）
GAUSSIAN_SIGMA = 1.0         # 高斯平滑强度（0=不平滑，1-3=适中，>5=强平滑）
FEATURE_AGGREGATION = "l2"   # 特征聚合方式 "l2", "mean", "max", "std", "entropy"

# 8. 阈值和对比度增强选项
USE_THRESHOLD = False       # 是否使用阈值过滤
THRESHOLD_VALUE = 0        # 阈值（0-1之间），低于此值的部分透明
HIGH_HEAT_MIN = 0.1         # 超过阈值的部分映射到的最小热度值
HIGH_HEAT_MAX = 1.0          # 超过阈值的部分映射到的最大热度值
CONTRAST_GAMMA = 1.5         # 对比度增强系数（>1增强对比度）
# ====================================================


class FusionLayerVisualizer: 
    def __init__(self, model):
        self.model = model
        self.fusion_encoder = model.transformer.encoder
        
        # 存储每层的输出
        self.layer_outputs = []
        self.hooks = []
        
        print(f"✓ Fusion Encoder: {self.fusion_encoder.__class__.__name__}")
        print(f"✓ 总层数: {self.fusion_encoder.num_layers}")
    
    # 为所有层注册 hook
    def register_all_layer_hooks(self):
        for layer_idx, layer in enumerate(self.fusion_encoder.layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # 捕获输出并立即移到 CPU 以节省显存
                    if isinstance(output, tuple):
                        captured = output[0].detach().cpu()
                    else:
                        captured = output.detach().cpu()
                    
                    self.layer_outputs.append({
                        'layer_idx': idx,
                        'features': captured
                    })
                    print(f"  ✓ Captured Layer {idx}: {captured.shape}")
                
                return hook_fn
            
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
        
        print(f"✓ 注册了 {len(self.hooks)} 个 hooks")
    
    # 移除所有 hooks
    def remove_all_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    # 运行推理并捕获所有层特征
    def run_inference(self, image, prompt):
        
        self.layer_outputs = []

        self.register_all_layer_hooks()
        
        print(f"\n运行推理: '{prompt}'")
        print("=" * 80)

        processor = Sam3Processor(self.model)
        
        with torch.no_grad():
            state = processor.set_image(image)
            _ = processor.set_text_prompt(state=state, prompt=prompt)
        
        self.remove_all_hooks()
        
        return self.layer_outputs


def feature_to_heatmap(features, target_size, invert=True, smooth_sigma=0, aggregation="l2", 
                      use_threshold=False, threshold=0.3, high_heat_min=0.65, high_heat_max=1.0, gamma=1.0):
    """
    将特征转换为 2D 热力图
    
    Args:
        features: [1, seq_len, dim] 或 [seq_len, dim]
        target_size: (H, W) 目标图像尺寸
        invert: 是否反转热力图(True=低值变高值)
        smooth_sigma: 高斯平滑的 sigma 值(0=不平滑)
        use_threshold: 是否使用阈值过滤
        threshold: 阈值，低于此值的部分将透明
        high_heat_min: 超过阈值部分映射到的最小热度
        high_heat_max: 超过阈值部分映射到的最大热度
        gamma: 对比度增强系数
    
    Returns:
        heatmap: numpy array of shape (H, W)
        alpha_mask: numpy array of shape (H, W) for transparency
    """
    # 1. 处理维度
    if features.dim() == 3:
        features = features.squeeze(0)  # [seq_len, dim]
    
    # 2. 根据不同方式计算特征强度
    if aggregation == "l2":
        # L2 范数（原始方法）
        heatmap = torch.norm(features, dim=-1)
    
    elif aggregation == "mean":
        # 特征均值
        heatmap = features.mean(dim=-1)
    
    elif aggregation == "max":
        # 最大激活值
        heatmap = features.max(dim=-1)[0]
    
    elif aggregation == "std":
        # 标准差（特征变化程度）
        heatmap = features.std(dim=-1)
    
    elif aggregation == "entropy":
        # 信息熵（需要先归一化为概率分布）
        features_prob = torch.softmax(features, dim=-1)
        heatmap = -(features_prob * torch.log(features_prob + 1e-8)).sum(dim=-1)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    
    # 3. 重塑为 2D 空间
    seq_len = heatmap.shape[0]
    side_len = int(np.sqrt(seq_len))
    
    if side_len * side_len == seq_len:
        h, w = side_len, side_len
    else:
        # 处理非正方形情况
        h = int(np.sqrt(seq_len))
        w = seq_len // h
        while h * w != seq_len and h > 1:
            h -= 1
            w = seq_len // h
    
    heatmap = heatmap[:h*w].reshape(h, w).float()
    
    # 4. 归一化到 [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 5. 反转（如果需要）
    if invert:
        heatmap = 1.0 - heatmap
    
    # 6. 转为 numpy
    heatmap_np = heatmap.numpy()
    
    # 7. 第一次高斯平滑（在阈值处理前）
    if smooth_sigma > 0:
        heatmap_np = gaussian_filter(heatmap_np, sigma=smooth_sigma)
        # 重新归一化
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 8. 对比度增强
    if gamma != 1.0:
        heatmap_np = np.power(heatmap_np, gamma)
    
    # 9. 阈值处理和高热度映射
    alpha_mask = np.ones_like(heatmap_np)
    if use_threshold:
        # 创建处理后的热力图
        heatmap_display = np.zeros_like(heatmap_np)
        mask_above_threshold = heatmap_np >= threshold
        
        if mask_above_threshold.any():
            values_above = heatmap_np[mask_above_threshold]
            # 将超过阈值的值归一化到高热度范围
            normalized = (values_above - threshold) / (heatmap_np.max() - threshold + 1e-8)
            heatmap_display[mask_above_threshold] = high_heat_min + normalized * (high_heat_max - high_heat_min)
        
        # 再次平滑处理后的热力图
        if smooth_sigma > 0:
            heatmap_display = gaussian_filter(heatmap_display, sigma=smooth_sigma * 1.5)
        
        heatmap_np = heatmap_display
        
        # 创建平滑的alpha mask
        alpha_mask_raw = np.where(heatmap_np >= threshold * 0.5, 
                                 np.clip((heatmap_np - threshold * 0.5) / (threshold * 0.5 + 1e-8), 0, 1), 
                                 0)
        if smooth_sigma > 0:
            alpha_mask = gaussian_filter(alpha_mask_raw, sigma=smooth_sigma * 0.5)
        else:
            alpha_mask = alpha_mask_raw
    
    # 10. 插值到目标尺寸
    heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0).float()
    alpha_tensor = torch.from_numpy(alpha_mask).unsqueeze(0).unsqueeze(0).float()
    
    heatmap_tensor = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    alpha_tensor = F.interpolate(alpha_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    return heatmap_tensor.squeeze().numpy(), alpha_tensor.squeeze().numpy()


def visualize_all_layers(image_np, layer_outputs, prompt, save_dir, invert=True, smooth_sigma=0, aggregation="l2"):

    num_layers = len(layer_outputs)
    target_size = (image_np.shape[0], image_np.shape[1])
    
    # 限制显示的层数
    layers_to_show = min(NUM_LAYERS_TO_SHOW, num_layers)
    
    # 创建 2x4 的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    # 第一张：原始图像
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    available_slots = 7  # 总共8个子图，减去原图，剩7个
    if num_layers <= available_slots:
        # 如果层数不多，全部显示
        layer_indices = list(range(num_layers))
    else:
        # 如果层数很多，均匀采样（确保包含第一层和最后一层）
        layer_indices = np.linspace(0, num_layers - 1, layers_to_show, dtype=int).tolist()
    
    # 显示每层的特征
    for subplot_idx, layer_idx in enumerate(layer_indices, start=1):
        if subplot_idx > 6:  # 前6个位置显示层演化
            break

        layer_data = layer_outputs[layer_idx]
        features = layer_data['features']
        
        # 转换为热力图
        heatmap, alpha_mask = feature_to_heatmap(
            features, target_size, invert=invert, smooth_sigma=smooth_sigma, aggregation=aggregation,
            use_threshold=USE_THRESHOLD, threshold=THRESHOLD_VALUE, 
            high_heat_min=HIGH_HEAT_MIN, high_heat_max=HIGH_HEAT_MAX, gamma=CONTRAST_GAMMA
        )
        
        # 绘制
        im = axes[subplot_idx].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[subplot_idx].set_title(
            f'Layer {layer_data["layer_idx"]} Features', 
            fontsize=12
        )
        axes[subplot_idx].axis('off')
        plt.colorbar(im, ax=axes[subplot_idx], fraction=0.046, pad=0.04)
    
    # 最后一张：叠加在原图上
    if num_layers > 0:
        last_features = layer_outputs[-1]['features']
        last_heatmap, last_alpha = feature_to_heatmap(
            last_features, target_size, invert=invert, smooth_sigma=smooth_sigma, aggregation=aggregation,
            use_threshold=USE_THRESHOLD, threshold=THRESHOLD_VALUE,
            high_heat_min=HIGH_HEAT_MIN, high_heat_max=HIGH_HEAT_MAX, gamma=CONTRAST_GAMMA
        )
        
        axes[7].imshow(image_np)
        im = axes[7].imshow(last_heatmap, cmap='jet', alpha=last_alpha * 0.8, vmin=0, vmax=1)
        
        title_parts = [f'Final Layer Overlay', f'"{prompt}"']
        
        axes[7].set_title(
            '\n'.join(title_parts), 
            fontsize=12, 
            fontweight='bold'
        )
        axes[7].axis('off')
        plt.colorbar(im, ax=axes[7], fraction=0.046, pad=0.04)
    
    # 隐藏未使用的子图
    for idx in range(subplot_idx + 1, 7):
        axes[idx].axis('off')
    
    title = f'Fusion Encoder Layer Evolution\nPrompt: "{prompt}"'

    plt.suptitle(
        title,
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(save_dir, 'layer_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_final_layer_detailed(image_np, layer_outputs, prompt, save_dir, invert=True, smooth_sigma=0, aggregation="l2"):
    
    if len(layer_outputs) == 0:
        print("  ⚠ 没有捕获到层输出")
        return
    
    last_layer = layer_outputs[-1]
    features = last_layer['features']
    target_size = (image_np.shape[0], image_np.shape[1])
    
    # 生成热力图和alpha mask
    heatmap, alpha_mask = feature_to_heatmap(
        features, target_size, invert=invert, smooth_sigma=smooth_sigma, aggregation=aggregation,
        use_threshold=USE_THRESHOLD, threshold=THRESHOLD_VALUE,
        high_heat_min=HIGH_HEAT_MIN, high_heat_max=HIGH_HEAT_MAX, gamma=CONTRAST_GAMMA
    )
    
    # 创建三联图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 子图1: 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=16)
    axes[0].axis('off')
    
    # 子图2: 热力图
    im1 = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    
    threshold_info = f', threshold={THRESHOLD_VALUE}' if USE_THRESHOLD else ''
    title_parts = [f'Fusion Encoder Features', f'(Layer {last_layer["layer_idx"]}, L2 Norm{threshold_info})']
    
    axes[1].set_title(
        '\n'.join(title_parts),
        fontsize=16
    )
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 子图3: 叠加图（使用alpha mask）
    axes[2].imshow(image_np)
    im2 = axes[2].imshow(heatmap, cmap='jet', alpha=alpha_mask * 0.7, vmin=0, vmax=1)
    axes[2].set_title(f'Overlay\n"{prompt}"', fontsize=16)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(save_dir, 'final_layer_detailed.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"\n❌ 错误: 图片不存在 {IMAGE_PATH}")
        return

    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    model = model.to(DEVICE).eval()
    
    visualizer = FusionLayerVisualizer(model)
    
    # 运行推理并捕获特征
    layer_outputs = visualizer.run_inference(image_pil, TEXT_PROMPT)
    
    if VISUALIZE_ALL_LAYERS:
        # 可视化所有层的演化
        visualize_all_layers(
            image_np, 
            layer_outputs, 
            TEXT_PROMPT, 
            OUTPUT_DIR,
            invert=INVERT_HEATMAP,
            smooth_sigma=GAUSSIAN_SIGMA,
            aggregation=FEATURE_AGGREGATION
        )
    
    # 可视化最后一层
    visualize_final_layer_detailed(
        image_np, 
        layer_outputs, 
        TEXT_PROMPT, 
        OUTPUT_DIR,
        invert=INVERT_HEATMAP,
        smooth_sigma=GAUSSIAN_SIGMA,
        aggregation=FEATURE_AGGREGATION
    )
    

    print(f"\n生成的文件:")
    if VISUALIZE_ALL_LAYERS:
        print(f"  • {OUTPUT_DIR}/layer_evolution.png - 所有层的特征演化")
    print(f"  • {OUTPUT_DIR}/final_layer_detailed.png - 最终层详细热力图")


if __name__ == "__main__":
    main()