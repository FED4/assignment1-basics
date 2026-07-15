import torch
from torch import nn

class CrossEntropy(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        stable_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        log_p = stable_logits - torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
        cross_entropy = -torch.mean(torch.gather(log_p, dim=-1, index=targets.unsqueeze(-1)))
        return cross_entropy