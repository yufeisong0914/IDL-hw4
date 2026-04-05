import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer


class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention (no causal mask).
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn       = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # No causal mask for encoder — can attend to all positions
        x, mha_attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=None)
        x = self.ffn(x)
        # Average over heads: (B, H, L, L) -> (B, L, L)
        if mha_attn_weights is not None and mha_attn_weights.dim() == 4:
            mha_attn_weights = mha_attn_weights.mean(dim=1)
        return x, mha_attn_weights
