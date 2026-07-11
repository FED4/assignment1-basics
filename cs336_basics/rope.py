import torch
from torch import nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        k = torch.arange(0, d_k // 2, device=device, dtype=dtype)
        frequencies = 1 / theta ** (2 * k / d_k)
        #frequencies = torch.Tensor([1/ theta ** (2 * k / d_k) for k in range(0, d_k // 2)], device=device, dtype=torch.float32)
        # outer product
        i_list = torch.arange(max_seq_len, device=device, dtype=dtype)
        angles = einsum(frequencies, i_list, "channels, max_seq_len -> max_seq_len channels")
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos, sin = self.cos_cache[token_positions], self.sin_cache[token_positions]
        #print(x.shape,cos.shape, sin.shape)
        while(x.ndim > cos.ndim):
            cos = cos.unsqueeze(-3) #保证最后两维一直是max_seq_len, channels
            sin = sin.unsqueeze(-3)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[... , 0::2] = out_even
        out[..., 1::2] = out_odd
        return out