import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize softmax along the last dimension (keys/source dimension)
        self.eps = 1e10  # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E)
        :param K: Key matrix of shape (N, ..., H, S, E)
        :param V: Value matrix of shape (N, ..., H, S, Ev)
        :param mask: Boolean mask of shape (N, ..., H, L, S), True = ignore position
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Store for backward pass
        self.Q = Q
        self.K = K
        self.V = V

        d_k = Q.shape[-1]

        # Scaled dot product: Q @ K^T / sqrt(d_k)  -> (N, ..., H, L, S)
        scaled_dot_product = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

        # Apply mask: add -eps to masked positions (True means mask)
        if mask is not None:
            scaled_dot_product = scaled_dot_product + mask * (-self.eps)

        # Softmax over key/source dimension (last dim): (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Weighted sum of values: (N, ..., H, L, Ev)
        output = np.matmul(self.attention_scores, V)

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output, shape (N, ..., H, L, Ev)
        :return: Gradients wrt Q, K, V
        """
        d_k = self.Q.shape[-1]

        # 1. Gradient wrt V: A^T @ d_output  -> (N, ..., H, S, Ev)
        d_V = np.matmul(self.attention_scores.swapaxes(-2, -1), d_output)

        # 2. Gradient wrt attention scores: d_output @ V^T  -> (N, ..., H, L, S)
        d_attention_scores = np.matmul(d_output, self.V.swapaxes(-2, -1))

        # 3. Gradient through softmax -> (N, ..., H, L, S)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)

        # 4. Scale by 1/sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)

        # 5. Gradient wrt Q: d_scaled_dot_product @ K  -> (N, ..., H, L, E)
        d_Q = np.matmul(d_scaled_dot_product, self.K)

        # 6. Gradient wrt K: d_scaled_dot_product^T @ Q  -> (N, ..., H, S, E)
        d_K = np.matmul(d_scaled_dot_product.swapaxes(-2, -1), self.Q)

        return d_Q, d_K, d_V
