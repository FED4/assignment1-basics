import torch
from torch import nn

class Softmax(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        max = torch.max(x, dim=dim, keepdim=True).values
        exp = torch.exp(x-max)
        return exp / torch.sum(exp, dim=dim, keepdim=True)