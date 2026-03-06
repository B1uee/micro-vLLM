import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 风格融合激活。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
