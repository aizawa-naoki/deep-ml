import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    t_a, t_b = torch.as_tensor(a), torch.as_tensor(b)

    if t_a.ndim != 2 or t_b.ndim != 2 or t_a.shape[1] != t_b.shape[0]:
        return -1

    return t_a @ t_b
