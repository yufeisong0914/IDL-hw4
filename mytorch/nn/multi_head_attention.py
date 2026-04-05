from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Initialize scaled dot product attention
        self.attention = ScaledDotProductAttention()

        # Initialize Q, K, V, and output projection layers (embed_dim -> embed_dim)
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where True indicates positions to ignore
        :param attn_mask: (L, S) where True indicates positions to ignore
        :return: (N, L, E)
        """
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # Project inputs: (N, L/S, E)
        q = self.q_proj.forward(query)   # (N, L, E)
        k = self.k_proj.forward(key)     # (N, S, E)
        v = self.v_proj.forward(value)   # (N, S, E)

        # Reshape for multiple heads: (N, H, L/S, E/H)
        q = self._split_heads(q)   # (N, H, L, E/H)
        k = self._split_heads(k)   # (N, H, S, E/H)
        v = self._split_heads(v)   # (N, H, S, E/H)

        # Combine padding and causal masks -> (N, H, L, S)
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply scaled dot-product attention
        attn_outputs = self.attention.forward(q, k, v, mask)  # (N, H, L, E/H)

        # Merge heads: (N, L, E)
        attn_output = self._concat_heads(attn_outputs)

        # Final projection: (N, L, E)
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        Backward pass for multi-head attention.
        """
        # Backpropagate through output projection
        d_attn_output = self.out_proj.backward(d_output)  # (N, L, E)

        # Undo head concatenation: (N, H, L, E/H)
        d_attn_outputs = self._split_heads(d_attn_output)

        # Backpropagate through scaled dot-product attention
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Merge head gradients: (N, L/S, E)
        d_q = self._concat_heads(d_q)
        d_k = self._concat_heads(d_k)
        d_v = self._concat_heads(d_v)

        # Backpropagate through input projections
        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask (N, S) and attn_mask (L, S) into (N, H, L, S).
        Returns None if both masks are None.
        """
        combined_mask = None

        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S) -> broadcast to (N, H, L, S)
            key_mask = key_padding_mask[:, np.newaxis, np.newaxis, :]  # (N, 1, 1, S)
            combined_mask = np.broadcast_to(key_mask, (self.N, self.num_heads, self.L, self.S)).copy()

        if attn_mask is not None:
            # (L, S) -> (1, 1, L, S)
            attention_mask = attn_mask[np.newaxis, np.newaxis, :, :]   # (1, 1, L, S)
            if combined_mask is None:
                combined_mask = np.broadcast_to(attention_mask, (self.N, self.num_heads, self.L, self.S)).copy()
            else:
                combined_mask = combined_mask | np.broadcast_to(attention_mask, (self.N, self.num_heads, self.L, self.S))

        return combined_mask

    def _split_heads(self, x):
        """
        Reshape (N, L, E) -> (N, H, L, E/H).
        """
        N, L, E = x.shape
        head_dim = E // self.num_heads
        # (N, L, H, E/H) -> (N, H, L, E/H)
        x = x.reshape(N, L, self.num_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _concat_heads(self, x):
        """
        Reshape (N, H, L, E/H) -> (N, L, E).
        """
        N, H, L, head_dim = x.shape
        # (N, H, L, E/H) -> (N, L, H, E/H) -> (N, L, E)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(N, L, H * head_dim)
        return x
