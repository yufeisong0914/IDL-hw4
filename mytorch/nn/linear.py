import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)

        Handles arbitrary batch dimensions like PyTorch
        """
        # Store input for backward pass
        self.A = A
        self.input_shape = A.shape

        # Flatten to 2D: (batch_size, in_features)
        batch_size = int(np.prod(A.shape[:-1]))
        A_2d = A.reshape(batch_size, -1)

        # Affine transformation: Z = A * W^T + b
        Z_2d = A_2d @ self.W.T + self.b

        # Unflatten back to original batch dimensions
        Z = Z_2d.reshape(*A.shape[:-1], -1)
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Flatten gradients and inputs to 2D
        batch_size = int(np.prod(dLdZ.shape[:-1]))
        dLdZ_2d = dLdZ.reshape(batch_size, -1)
        A_2d = self.A.reshape(batch_size, -1)

        # Compute gradients
        self.dLdA = (dLdZ_2d @ self.W).reshape(self.input_shape)
        self.dLdW = dLdZ_2d.T @ A_2d
        self.dLdb = dLdZ_2d.sum(axis=0)

        # Return gradient of loss wrt input
        return self.dLdA
