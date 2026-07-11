import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.dmodel = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        squared_x = torch.square(x)
        x_squared_sum = torch.sum(squared_x, dim=-1, keepdim=True)
        scaled = x_squared_sum / self.dmodel
        rms_norm = x * self.gain / torch.sqrt(scaled + self.eps)
        return rms_norm.to(in_type)