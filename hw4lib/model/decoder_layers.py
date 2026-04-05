import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer


## -------------------------------------------------------------------------------------------------
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer: masked self-attention + feed-forward.
    Used in decoder-only Transformer (HW4P1).
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn       = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mha_attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.ffn(x)
        # Average over heads: (B, H, L, L) -> (B, L, L)
        if mha_attn_weights is not None and mha_attn_weights.dim() == 4:
            mha_attn_weights = mha_attn_weights.mean(dim=1)
        return x, mha_attn_weights


## -------------------------------------------------------------------------------------------------
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer: masked self-attention + cross-attention + feed-forward.
    Used in encoder-decoder Transformer (HW4P2).
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn        = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                dec_key_padding_mask: Optional[torch.Tensor] = None,
                enc_key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, self_attn_weights  = self.self_attn(x, key_padding_mask=dec_key_padding_mask, attn_mask=attn_mask)
        x, cross_attn_weights = self.cross_attn(x, enc_output, key_padding_mask=enc_key_padding_mask)
        x = self.ffn(x)
        # Average over heads: (B, H, L, S) -> (B, L, S)
        if self_attn_weights is not None and self_attn_weights.dim() == 4:
            self_attn_weights = self_attn_weights.mean(dim=1)
        if cross_attn_weights is not None and cross_attn_weights.dim() == 4:
            cross_attn_weights = cross_attn_weights.mean(dim=1)
        return x, self_attn_weights, cross_attn_weights
