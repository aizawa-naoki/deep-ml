import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    t_a, t_b = torch.as_tensor(a), torch.as_tensor(b)

    if not (t_a.ndim == t_b.ndim == 2) or t_a.size(-1) != t_b.size(-2):
        return torch.tensor(-1)

    # d
    return t_a.to(t_b).mm(t_b)
