import math
import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2 / math.pi) * (
                    x + 0.044715 * torch.pow(x, 3)
                )
            )
        )


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)