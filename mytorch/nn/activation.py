import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        # Normalize dim to positive index
        dim = self.dim if self.dim >= 0 else len(Z.shape) + self.dim

        # Move the target dim to last position
        Z_moved = np.moveaxis(Z, dim, -1)
        orig_shape = Z_moved.shape

        # Flatten to 2D: (batch, C)
        Z_2d = Z_moved.reshape(-1, Z_moved.shape[-1])

        # Numerically stable softmax
        Z_max = Z_2d.max(axis=-1, keepdims=True)
        exp_Z = np.exp(Z_2d - Z_max)
        A_2d = exp_Z / exp_Z.sum(axis=-1, keepdims=True)

        # Reshape back and move dim back to original position
        A_moved = A_2d.reshape(orig_shape)
        self.A = np.moveaxis(A_moved, -1, dim)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Normalize dim to positive index
        dim = self.dim if self.dim >= 0 else len(self.A.shape) + self.dim

        # Move the target dim to last position
        A_moved = np.moveaxis(self.A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)
        orig_shape = A_moved.shape

        # Flatten to 2D
        A_2d = A_moved.reshape(-1, A_moved.shape[-1])
        dLdA_2d = dLdA_moved.reshape(-1, dLdA_moved.shape[-1])

        n, C = A_2d.shape
        dLdZ_2d = np.zeros_like(A_2d)

        for i in range(n):
            a = A_2d[i]           # (C,)
            J = np.diag(a) - np.outer(a, a)  # (C, C)
            dLdZ_2d[i] = dLdA_2d[i] @ J

        # Reshape and move dim back
        dLdZ_moved = dLdZ_2d.reshape(orig_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        return dLdZ
