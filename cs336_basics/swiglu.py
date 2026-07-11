import torch
from torch import nn
from einops import einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int, device=None, dtype=None  ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.silu(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        v = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        uv = u * v 
        return einsum(self.W2, uv, "d_model d_ff, ... d_ff -> ... d_model")
