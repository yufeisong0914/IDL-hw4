import torch.nn as nn
import torch
from typing import Tuple, Optional


class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    Causally-masked self-attention with residual connection.
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha     = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        x, mha_attn_weights = self.mha(x, x, x,
                                        key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask,
                                        need_weights=True,
                                        average_attn_weights=False)
        x = residual + self.dropout(x)
        return x, mha_attn_weights


## -------------------------------------------------------------------------------------------------
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    Cross-attention between decoder (query) and encoder (key/value).
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha     = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        # query from decoder (x), key/value from encoder (y)
        x, mha_attn_weights = self.mha(x, y, y,
                                        key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask,
                                        need_weights=True,
                                        average_attn_weights=False)
        x = residual + self.dropout(x)
        return x, mha_attn_weights


## -------------------------------------------------------------------------------------------------
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    Position-wise feed-forward network with residual connection.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = residual + self.dropout(self.ffn(x))
        return x
