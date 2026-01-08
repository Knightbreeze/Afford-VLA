"""Implementation of additional/mlp projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from typing import List, Dict, Tuple, Union, Optional
from einops import rearrange

class CrossAttentionAlignment(nn.Module):
    """
    Cross-Attention based alignment module.
    """
    def __init__(
        self,
        vision_dim: int,
        aff_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.aff_dim = aff_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Projection layer
        self.aff_to_vision_proj = nn.Linear(aff_dim, vision_dim)
        
        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer Normalization
        self.norm_q = nn.LayerNorm(vision_dim)
        self.norm_kv = nn.LayerNorm(vision_dim)
        self.norm_out = nn.LayerNorm(vision_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 4, vision_dim),
            nn.Dropout(dropout),
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights similar to MLPAlignProjector"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
    def forward(
        self,
        vision_hidden: torch.Tensor,  # [B, N_vis, D_vis]
        aff_hidden: torch.Tensor,     # [S_aff, B, D_aff] or [B, S_aff, D_aff]
        vision_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            vision_hidden: [B, N_vis, D_vis]
            aff_hidden: [S_aff, B, D_aff] or [B, S_aff, D_aff]
            vision_mask: [B, N_vis] - True 表示有效 token
            return_attention: 是否返回注意力权重
        """
        # 1. 处理 aff_hidden 维度: [S, B, D] -> [B, S, D]
        if aff_hidden.ndim == 3 and aff_hidden.shape[1] == vision_hidden.shape[0]:
            # SAM3 format: [S_aff, B, D_aff] -> [B, S_aff, D_aff]
            aff_hidden = aff_hidden.permute(1, 0, 2)
        
        # 2. 投影 affordance 到 vision 维度
        aff_projected = self.aff_to_vision_proj(aff_hidden)  # [B, N_aff, D_vis]
        
        # 3. Layer Norm
        vision_q = self.norm_q(vision_hidden)
        aff_kv = self.norm_kv(aff_projected)
        
        # 4. 准备 mask
        if vision_mask is not None:
            key_padding_mask = ~vision_mask
        else:
            key_padding_mask = None
        
        # 5. Cross-Attention
        residual = vision_hidden
        attn_output, attn_weights = self.cross_attn(
            query=vision_q,
            key=aff_kv,
            value=aff_kv,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        
        # 6. 残差连接
        vision_hidden = residual + attn_output
        
        # 7. FFN
        residual = vision_hidden
        vision_hidden = self.norm_out(vision_hidden)
        vision_hidden = residual + self.ffn(vision_hidden)
        
        if return_attention:
            return vision_hidden, attn_weights
        return vision_hidden


class CrossAttentionAlignProjector(nn.Module):
    """
    Calculate the alignment between VLA and Affordance embeddings using Cross-Attention.
    Follows the same loss computation and mask handling as MLPAlignProjector.
    """
    def __init__(
        self,
        vla_dim: int,
        aff_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vla_dim = vla_dim
        self.aff_dim = aff_dim
        
        self.cross_attn_module = CrossAttentionAlignment(
            vision_dim=vla_dim,
            aff_dim=aff_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def compute_align_loss_cosine(self, vision_hidden, aff_hidden, align_mask):
        """
        Compute cosine similarity loss following MLPAlignProjector's approach.
        
        Args:
            vision_hidden: [B, N, D] - aligned vision features
            aff_hidden: [B, N, D] - original vision features (target)
            align_mask: [B, N] - True for valid tokens
        
        Returns:
            align_loss: scalar
        """
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        
        align_loss = 0
        bsz = vision_hidden.shape[0]
        
        for _vision, _aff, _mask in zip(vision_hidden, aff_hidden, align_mask):
            # Normalize features
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _aff = torch.nn.functional.normalize(_aff, dim=-1)
            
            # Compute cosine similarity loss with mask
            # Only compute loss on valid tokens (where _mask is True)
            align_loss += 1 - mean_flat((_vision * _aff)[_mask].sum(dim=-1))
        
        align_loss /= bsz  # Average over batch size
        return align_loss

    def forward(
        self,
        vision_hidden: torch.Tensor,  # vla_emb: [B, N_vis, D_vis]
        aff_hidden: torch.Tensor,     # tager_emb: [S_aff, B, D_aff]
        align_mask: torch.Tensor,     # [B, N_vis] - True for valid tokens
    ):
        """s
        Compute alignment loss using Cross-Attention.
        
        Args:
            vision_hidden: [B, N_vis, D_vis] - VLA features (original)
            aff_hidden: [S_aff, B, D_aff] - Affordance features
            align_mask: [B, N_vis] - True for valid tokens
        
        Returns:
            align_loss: scalar
        """

        # 1. Pass through Cross-Attention to get aligned features
        aligned_vision = self.cross_attn_module(
            vision_hidden, 
            aff_hidden, 
            align_mask, 
            return_attention=False
        ) # aligned_vision: [B, N_vis, D_vis]

        # 2. Compute cosine similarity loss (following MLPAlignProjector)
        align_loss = self.compute_align_loss_cosine(
            aligned_vision,  # Transformed features
            aff_hidden,      # Original features (target)
            align_mask       # Mask for valid tokens
        ).mean()  # Mean for sequence length

        return align_loss


class MLPAlignProjector(nn.Module):
    """
    calculate the alignment between Affordance and VLA embeddings.
    """
    def __init__(
            self,
            vla_dim: int, 
            aff_dim: int,
            use_vlm_norm: bool = False,
        ) -> None:
        super().__init__()

        self.vla_dim = vla_dim # 2048
        self.aff_dim = aff_dim # 256

        self.fc1 = nn.Linear(self.vla_dim, 4 * self.aff_dim, bias=True)
        self.fc2 = nn.Linear(4 * self.aff_dim, self.aff_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(vla_dim) if use_vlm_norm else None

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor = None) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, aff_hidden, align_mask):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _aff, _mask in zip(vision_hidden, aff_hidden, align_mask):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _aff = torch.nn.functional.normalize(_aff, dim=-1)
            # align_loss += 1 - torch.mean(vision_hidden * vggt_hidden).sum(dim=-1).mean()
            align_loss += 1 - mean_flat((_vision * _aff)[_mask].sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_emb, target_emb, align_mask):
        # project vla dimension and calculate align loss
        LLM_emb = self.align_dimension(LLM_emb)
        align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb, align_mask).mean()  # mean for sequence length
        return align_loss
