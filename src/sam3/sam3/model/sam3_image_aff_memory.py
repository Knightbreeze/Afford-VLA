import torch
from sam3.model.sam3_image import Sam3Image
from sam3.model.geometry_encoders import Prompt


class Sam3AffordanceModel(Sam3Image):
    def forward_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
    ):
        # backbone_out -> image encoder
        # prompt encodr
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        
        # fusion encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        
        # 组织输出 - 直接返回 encoder 的输出
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prompt_features": encoder_out["prompt_after_enc"],
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
            # 额外的空间信息
            "spatial_shapes": encoder_out["spatial_shapes"],
            "vis_feat_sizes": encoder_out["vis_feat_sizes"],
            "level_start_index": encoder_out["level_start_index"],
            "pos_embed": encoder_out["pos_embed"],
            "padding_mask": encoder_out["padding_mask"],
        }
        
        # 计算 patch 起始索引（用于提取纯图像特征）
        num_text_tokens = prompt.shape[0] - geometric_prompt.box_embeddings.shape[0]
        out["patch_start_idx"] = num_text_tokens
        
        return out
    
    def extract_patch_features(self, encoder_output: dict) -> torch.Tensor:
        """
        从 encoder 输出中提取图像 patch 特征
        
        Args:
            encoder_output: forward_grounding 返回的字典
            
        Returns:
            patch_features: [B, num_patches, C] 形状的特征
        """
        hidden_states = encoder_output["encoder_hidden_states"]  # [seq_len, B, C]
        patch_start_idx = encoder_output["patch_start_idx"]
        
        # 转换为 batch-first: [B, seq_len, C]
        hidden_states = hidden_states.transpose(0, 1)
        
        # 提取 patch 特征 (去除文本和几何 prompt)
        patch_features = hidden_states[:, patch_start_idx:, :]
        
        return patch_features
    


# from sam3.model_builder import build_sam3_image_model

# aff_model = build_sam3_image_model(mode="affordance")
