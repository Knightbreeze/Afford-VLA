from PIL import Image
import torch

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def compare_single_vs_batch():
    """详细对比单图和批处理的中间结果"""

    model = build_sam3_image_model(mode="sam3")
    model.eval()  # 确保 eval 模式
    processor = Sam3Processor(model)

    test_image_path = "/home/nightbreeze/research/Data/robotwin/lerobotV2.1_data/place_bread_skillet-aloha-agilex_clean_50/images/rgb_global/episode_000000/frame_0002.png"
    image = Image.open(test_image_path)
    prompt = "the right yellow part"

    print("=" * 80)
    print("SINGLE IMAGE MODE")
    print("=" * 80)

    with torch.no_grad():
        state_single = processor.set_image(image)

        # 检查 backbone 输出
        print("\n[1] Backbone output:")
        for key, val in state_single["backbone_out"].items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, mean={val.mean():.6f}, std={val.std():.6f}")

        # 检查 geometric_prompt
        # print(f"\n[2] Geometric prompt:")
        # gp = state_single["geometric_prompt"]
        # print(f"  box_embeddings: {gp.box_embeddings.shape}")
        # print(f"  box_mask: {gp.box_mask.shape}")

        output_single = processor.set_text_prompt(state=state_single, prompt=prompt)

    print("\n[3] Final output:")
    print(f"  Detected boxes: {len(output_single['boxes'])}")
    print(f"  Scores: {output_single['scores'][:5] if len(output_single['scores']) > 0 else 'None'}")
    print(
        f"  Heatmap logits range: [{output_single['full_heatmap_logits'].min():.6f}, {output_single['full_heatmap_logits'].max():.6f}]"
    )
    print(f"  Heatmap mean: {output_single['full_heatmap_logits'].mean():.6f}")

    print("\n" + "=" * 80)
    print("BATCH MODE (same image twice)")
    print("=" * 80)

    with torch.no_grad():
        state_batch = processor.set_image_batch([image, image])

        # 检查 backbone 输出
        print("\n[1] Backbone output:")
        for key, val in state_batch["backbone_out"].items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, mean={val.mean():.6f}, std={val.std():.6f}")

        # # 检查 geometric_prompt
        # print(f"\n[2] Geometric prompt:")
        # gp = state_batch["geometric_prompt"]
        # print(f"  box_embeddings: {gp.box_embeddings.shape}")
        # print(f"  box_mask: {gp.box_mask.shape}")

        output_batch = processor.set_text_prompt_batch(prompts=[prompt, prompt], states=state_batch)

    print("\n[3] Final output (first image):")
    print(f"  Detected boxes: {len(output_batch['boxes'][0])}")
    print(f"  Scores: {output_batch['scores'][0][:5] if len(output_batch['scores'][0]) > 0 else 'None'}")
    print(
        f"  Heatmap logits range: [{output_batch['full_heatmap_logits'][0].min():.6f}, {output_batch['full_heatmap_logits'][0].max():.6f}]"
    )
    print(f"  Heatmap mean: {output_batch['full_heatmap_logits'][0].mean():.6f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # 比较热力图
    heatmap_single = output_single["full_heatmap_logits"].cpu()
    heatmap_batch = output_batch["full_heatmap_logits"][0].cpu()

    diff = torch.abs(heatmap_single - heatmap_batch)
    print("\nHeatmap difference:")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff: {diff.max():.6f}")
    print(f"  Relative diff: {(diff / (torch.abs(heatmap_single) + 1e-8)).mean():.6f}")

    # 比较 backbone 输出
    print("\nBackbone vision_features difference:")
    if "vision_features" in state_single["backbone_out"]:
        vf_single = state_single["backbone_out"]["vision_features"]
        vf_batch = state_batch["backbone_out"]["vision_features"][0:1]  # 取第一个样本
        vf_diff = torch.abs(vf_single - vf_batch)
        print(f"  Mean abs diff: {vf_diff.mean():.6f}")
        print(f"  Max abs diff: {vf_diff.max():.6f}")


if __name__ == "__main__":
    compare_single_vs_batch()
