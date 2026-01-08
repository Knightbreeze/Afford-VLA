from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image_processor import Sam3Processor

# 导入 SAM 3 相关模块
from sam3.model_builder import build_sam3_image_model


class CachedSAM3Inference:
    """
    SAM 3 高效推理包装器：
    1. encode_image(): 只运行一次重型 Vision Encoder，缓存特征。
    2. predict(): 接收文本列表，复用缓存特征，快速解码。
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = model.device
        self.cached_backbone_out = None
        self.original_size = None  # (width, height)
        self.img_tensor = None

    def encode_image(self, image_path_or_pil):
        """
        第一阶段：图像编码（最耗时步骤，约占 90% 时间）。
        只需运行一次。
        """
        # 1. 加载图像
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        self.original_size = image.size  # (W, H)

        # 尝试自动检测模型期望的输入尺寸
        target_size = 1024  # 默认 SAM 标准
        try:
            # 检查常见的属性名
            if hasattr(self.model.backbone, "img_size"):
                target_size = self.model.backbone.img_size
            elif hasattr(self.model, "image_encoder") and hasattr(self.model.image_encoder, "img_size"):
                target_size = self.model.image_encoder.img_size
            print(f"ℹ️  Model expects input size: {target_size}x{target_size}")
        except Exception:
            pass

        # 优先尝试使用 processor 提供的 set_image
        # 我们移除 try-except 以便看到 processor 内部的真实错误（如果有）
        try:
            inference_state = self.processor.set_image(image)
            if isinstance(inference_state, dict):
                for key in ["image_tensor", "img_tensor", "images", "input_tensor"]:
                    if key in inference_state and isinstance(inference_state[key], torch.Tensor):
                        self.img_tensor = inference_state[key].unsqueeze(0).to(self.device).contiguous()
                        print("✅ Used processor.set_image() for preprocessing.")
                        break
        except Exception as e:
            print(f"⚠️  processor.set_image() failed or returned unknown format: {e}")
            inference_state = None

        # 如果 processor 失败，使用手动预处理（标准 SAM 逻辑：Resize Longest + Pad）
        if self.img_tensor is None:
            print(f"⚠️  Falling back to manual preprocessing (Target: {target_size}x{target_size})...")

            img_np = np.array(image)
            old_h, old_w = img_np.shape[:2]
            scale = target_size * 1.0 / max(old_h, old_w)
            new_h, new_w = int(old_h * scale), int(old_w * scale)

            # Resize longest side
            img_resized = cv2.resize(img_np, (new_w, new_h))

            # Pad to target_size (bottom-right padding)
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

            # Normalize & Convert
            img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1).contiguous() / 255.0

            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(3, 1, 1)

            img_tensor = img_tensor.to(self.device)
            img_tensor = (img_tensor - mean) / std
            self.img_tensor = img_tensor.unsqueeze(0).contiguous()

        # 3. 运行 Vision Encoder (Backbone)
        self.model.eval()
        # 强制同步模型设备
        self.model.to(self.device)

        with torch.no_grad():
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                backbone_out = {"img_batch_all_stages": self.img_tensor}

                # 再次确保 tensor 在 device 上
                if self.img_tensor.device != self.device:
                    self.img_tensor = self.img_tensor.to(self.device)

                print(f"🚀 Running backbone forward... Input shape: {self.img_tensor.shape}")
                image_feats = self.model.backbone.forward_image(self.img_tensor)
                backbone_out.update(image_feats)

                self.cached_backbone_out = backbone_out
                print("✅ Image encoded successfully.")
            except AssertionError as e:
                print("\n❌ AssertionError detected in model forward pass!")
                print(
                    "This usually means the input image size does not match the model's pre-computed positional embeddings."
                )
                print(f"Current Input Shape: {self.img_tensor.shape}")

                # 尝试检查模型内部的 freqs_cis 形状以帮助调试
                try:
                    # 深度优先搜索 freqs_cis
                    for name, module in self.model.named_modules():
                        if hasattr(module, "freqs_cis") and isinstance(module.freqs_cis, torch.Tensor):
                            print(f"Found internal 'freqs_cis' in {name}: shape={module.freqs_cis.shape}")
                            break
                except:
                    pass
                raise e  # filepath: /home/nightbreeze/research/Code/AffVLA/sam3/multi_prompt_inference.py

    # ...existing code...
    def encode_image(self, image_path_or_pil):
        """
        第一阶段：图像编码（最耗时步骤，约占 90% 时间）。
        只需运行一次。
        """
        # 1. 加载图像
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        self.original_size = image.size  # (W, H)

        # 尝试自动检测模型期望的输入尺寸
        target_size = 1024  # 默认 SAM 标准
        try:
            # 检查常见的属性名
            if hasattr(self.model.backbone, "img_size"):
                target_size = self.model.backbone.img_size
            elif hasattr(self.model, "image_encoder") and hasattr(self.model.image_encoder, "img_size"):
                target_size = self.model.image_encoder.img_size
            print(f"ℹ️  Model expects input size: {target_size}x{target_size}")
        except Exception:
            pass

        # 优先尝试使用 processor 提供的 set_image
        # 我们移除 try-except 以便看到 processor 内部的真实错误（如果有）
        try:
            inference_state = self.processor.set_image(image)
            if isinstance(inference_state, dict):
                for key in ["image_tensor", "img_tensor", "images", "input_tensor"]:
                    if key in inference_state and isinstance(inference_state[key], torch.Tensor):
                        self.img_tensor = inference_state[key].unsqueeze(0).to(self.device).contiguous()
                        print("✅ Used processor.set_image() for preprocessing.")
                        break
        except Exception as e:
            print(f"⚠️  processor.set_image() failed or returned unknown format: {e}")
            inference_state = None

        # 如果 processor 失败，使用手动预处理（标准 SAM 逻辑：Resize Longest + Pad）
        if self.img_tensor is None:
            print(f"⚠️  Falling back to manual preprocessing (Target: {target_size}x{target_size})...")

            img_np = np.array(image)
            old_h, old_w = img_np.shape[:2]
            scale = target_size * 1.0 / max(old_h, old_w)
            new_h, new_w = int(old_h * scale), int(old_w * scale)

            # Resize longest side
            img_resized = cv2.resize(img_np, (new_w, new_h))

            # Pad to target_size (bottom-right padding)
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

            # Normalize & Convert
            img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1).contiguous() / 255.0

            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(3, 1, 1)

            img_tensor = img_tensor.to(self.device)
            img_tensor = (img_tensor - mean) / std
            self.img_tensor = img_tensor.unsqueeze(0).contiguous()

        # 3. 运行 Vision Encoder (Backbone)
        self.model.eval()
        # 强制同步模型设备
        self.model.to(self.device)

        with torch.no_grad():
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                backbone_out = {"img_batch_all_stages": self.img_tensor}

                # 再次确保 tensor 在 device 上
                if self.img_tensor.device != self.device:
                    self.img_tensor = self.img_tensor.to(self.device)

                print(f"🚀 Running backbone forward... Input shape: {self.img_tensor.shape}")
                image_feats = self.model.backbone.forward_image(self.img_tensor)
                backbone_out.update(image_feats)

                self.cached_backbone_out = backbone_out
                print("✅ Image encoded successfully.")
            except AssertionError as e:
                print("\n❌ AssertionError detected in model forward pass!")
                print(
                    "This usually means the input image size does not match the model's pre-computed positional embeddings."
                )
                print(f"Current Input Shape: {self.img_tensor.shape}")

                # 尝试检查模型内部的 freqs_cis 形状以帮助调试
                try:
                    # 深度优先搜索 freqs_cis
                    for name, module in self.model.named_modules():
                        if hasattr(module, "freqs_cis") and isinstance(module.freqs_cis, torch.Tensor):
                            print(f"Found internal 'freqs_cis' in {name}: shape={module.freqs_cis.shape}")
                            break
                except:
                    pass
                raise e
            except Exception as e:
                print(f"Error in backbone forward: {e}")
                raise


def predict(self, prompt_list, conf_threshold=0.2):
    def predict(self, prompt_list, conf_threshold=0.2):
        """
        第二阶段：多 Prompt 解码（非常快）。
        可以一次性传入多个 prompt。
        """
        if self.cached_backbone_out is None:
            raise ValueError("Please run encode_image() first.")

        all_results = []

        # 1. 批量处理文本编码 (Text Encoding)
        # 将所有 prompt 一次性编码
        # SAM3 backbone.forward_text 接收 list of strings
        with torch.no_grad():
            text_outputs = self.model.backbone.forward_text(prompt_list, device=self.device)

            # 2. 针对每个 Prompt 进行解码
            # 注意：虽然 Text Encoder 可以 Batch，但 Decoder 通常需要逐个 Prompt 处理
            # 或者我们需要构造复杂的 Batched Input。为了代码清晰，这里逐个循环，但复用 Image Feature

            for i, prompt_text in enumerate(prompt_list):
                # 构造当前 Prompt 的 backbone_out
                # 我们需要从 text_outputs 中切片出第 i 个文本的特征
                # text_outputs['language_features'] shape: [B, L, C]

                # 浅拷贝缓存，避免修改
                current_backbone_out = self.cached_backbone_out.copy()
                # 更新文本特征（这里直接引用全部，通过 text_ids 选择）
                current_backbone_out.update(text_outputs)

                # 3. 构造 Input 对象 (Mock)
                find_input = SimpleNamespace(
                    img_ids=torch.tensor([0], device=self.device),
                    text_ids=torch.tensor([i], device=self.device),  # 指向当前 prompt
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                )

                # 4. 构造空的几何提示 (Geometric Prompt)
                geometric_prompt = Prompt(box_embeddings=None, box_mask=None, box_labels=None)

                # 5. 运行 Grounding (Fusion + Decoder)
                # 这步非常快，因为它只运行 Transformer Decoder
                out = self.model.forward_grounding(
                    backbone_out=current_backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )

                # 6. 提取结果
                pred_masks = out["pred_masks"][0]  # [N, H, W]
                pred_scores = out["pred_logits"][0].sigmoid().squeeze(-1)  # [N]
                pred_boxes = out["pred_boxes"][0]  # [N, 4] (cx, cy, w, h) normalized

                # 7. 过滤低置信度结果
                keep_indices = pred_scores > conf_threshold
                if keep_indices.sum() > 0:
                    valid_masks = pred_masks[keep_indices]
                    valid_scores = pred_scores[keep_indices]
                    valid_boxes = pred_boxes[keep_indices]

                    # Resize masks 到原图大小
                    valid_masks = torch.nn.functional.interpolate(
                        valid_masks.unsqueeze(1),
                        size=(self.original_size[1], self.original_size[0]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    valid_masks = (valid_masks > 0.0).cpu().numpy()

                    all_results.append(
                        {
                            "prompt": prompt_text,
                            "scores": valid_scores.cpu().numpy(),
                            "masks": valid_masks,
                            "boxes": valid_boxes.cpu().numpy(),  # normalized format
                        }
                    )
                else:
                    print(f"  - No object found for prompt: '{prompt_text}'")

        return all_results


def visualize_multi_results(image_path, results, output_path="output_multi_prompt.jpg"):
    """可视化多个类别的结果并保存"""
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        prompt = res["prompt"]

        for box, score, mask in zip(res["boxes"], res["scores"], res["masks"]):
            # 绘制 Mask (半透明)
            # 创建彩色 mask 层
            mask_img = np.zeros((mask.shape[0], mask.shape[1], 4))
            mask_img[mask, :] = list(plt.cm.colors.to_rgba(color))
            mask_img[mask, 3] = 0.4  # Alpha
            ax.imshow(mask_img)

            # 绘制 Box (cxcywh -> xyxy 转换并反归一化)
            w, h = image.size
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            rect = plt.Rectangle((x1, y1), bw * w, bh * h, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

            # 绘制标签
            ax.text(x1, y1, f"{prompt}: {score:.2f}", fontsize=10, bbox=dict(facecolor=color, alpha=0.5), color="white")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    # plt.show() # 如果在无头服务器上运行，请注释掉此行


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 设置设备和加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 model on {device}...")

    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model)

    # 2. 初始化缓存推理器
    cache_infer = CachedSAM3Inference(model, processor)

    # 3. 输入配置
    IMAGE_PATH = "sam3/input/test1.png"  # 替换为你的图片路径

    # 定义多个 Prompt (这就是你想要的“多类别”输入)
    # 例如：分解后的指令 ["spoon handle", "cup rim", "spoon bowl"]
    PROMPTS = ["spoon handle", "cup", "spoon"]

    print(f"\nProcessing image: {IMAGE_PATH}")
    print(f"Prompts: {PROMPTS}")

    # 4. 运行推理流程
    # Step A: 编码图像 (只运行一次，耗时大头)
    cache_infer.encode_image(IMAGE_PATH)

    # Step B: 预测所有 Prompt (运行多次，但非常快)
    # 你可以在这里传入整个列表
    results = cache_infer.predict(PROMPTS, conf_threshold=0.25)

    # 5. 输出和可视化
    print(f"\nFound objects for {len(results)}/{len(PROMPTS)} prompts.")
    for res in results:
        print(f"  Prompt '{res['prompt']}': found {len(res['scores'])} instances. Max score: {res['scores'].max():.2f}")

    if len(results) > 0:
        visualize_multi_results(IMAGE_PATH, results)
