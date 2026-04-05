import torch

def PadMask(padded_input, input_lengths):
    """
    Create a mask for padding positions.
    Args:
        padded_input: The input tensor, shape (N, T, ...).
        input_lengths: Actual lengths before padding, shape (N,).
    Returns:
        Boolean mask tensor with shape (N, T). True = padding position.
    """
    N = padded_input.shape[0]
    T = padded_input.shape[1]
    # indices: (N, T); mask[i,t] = True if t >= lengths[i]
    indices = torch.arange(T, device=padded_input.device).unsqueeze(0).expand(N, -1)
    mask = indices >= input_lengths.unsqueeze(1).to(padded_input.device)
    return mask


def CausalMask(padded_input):
    """
    Create a causal mask for self-attention.
    Args:
        padded_input: Input tensor, shape (N, T, ...).
    Returns:
        Boolean mask tensor with shape (T, T). True = should NOT attend (future positions).
    """
    T = padded_input.shape[1]
    # Upper triangular excluding diagonal: position j > i should be masked
    mask = torch.ones(T, T, dtype=torch.bool, device=padded_input.device).triu(diagonal=1)
    return mask
