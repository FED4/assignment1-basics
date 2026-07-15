import torch
from torch import nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.swiglu import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, device=device, dtype=dtype)
        self.ff= SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        y = x + self.attention(self.rmsnorm1(x), self.theta, token_positions)
        out = y + self.ff(self.rmsnorm2(y))
        return out

class TransformerLM(nn.Module):
    def __init__(self, voacb_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, theta:float, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(voacb_size, d_model,  device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, voacb_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        rmsnorm = self.rmsnorm(x)
        logits = self.lm_head(rmsnorm)
        return logits