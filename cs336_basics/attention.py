import torch
from torch import nn
from cs336_basics.softmax import Softmax
import einops
from einops import einsum, rearrange
import math
from cs336_basics.rope import RoPE

class ScaledDotProductAttention(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()  
        self.device = device
        self.dtype = dtype

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = einsum(Q, K, "... n d_k, ... m d_k-> ... n m")
        scores = scores / math.sqrt(Q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) # False填-inf需要取反
        softmax = Softmax(device=self.device, dtype=self.dtype)
        softmax = softmax(scores, dim=-1)
        return einsum(softmax, V, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, theta: float, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = einsum(x, self.W_Q, "... d_model, d_out d_model -> ... d_out")
        K = einsum(x, self.W_K, "... d_model, d_out d_model -> ... d_out")
        V = einsum(x, self.W_V, "... d_model, d_out d_model -> ... d_out")
        Q_split = rearrange(Q, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        K_split = rearrange(K, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        V_split = rearrange(V, "... n (h d_v) -> ... h n d_v", h=self.num_heads)
        scaled_dot_product_attention = ScaledDotProductAttention(device=self.device, dtype=self.dtype)
        sequence_length = x.shape[-2]
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=self.device, dtype=torch.bool))
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device)
        rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=sequence_length, device=self.device, dtype=self.dtype)
        Q_rope = rope(Q_split, token_positions)
        K_rope = rope(K_split, token_positions)
        out = scaled_dot_product_attention(Q_rope, K_rope, V_split, mask)
        out = rearrange(out, "... h n d_v -> ... n (h d_v)")
        return einsum(out, self.W_O, "... d_model, d_out d_model -> ... d_out")
        
        