"""FedPara: Low-rank Parameterization for Federated Learning (Nam et al., ICLR 2022)."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from flwr.serverapp.strategy import FedAvg

from .base import BaseTrainer

# =====================================
# 1. Low-Rank Fully Connected Layer 
# =====================================

class LowRankNN(nn.Module):
    """Low-rank weight synthesis for fully connected layer (Y * X^T)."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.X = nn.Parameter(torch.empty(size=(input_dim, rank)), requires_grad=True)
        self.Y = nn.Parameter(torch.empty(size=(output_dim, rank)), requires_grad=True)
        init.kaiming_normal_(self.X, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.Y, mode="fan_out", nonlinearity="relu")

    def forward(self) -> torch.Tensor:
        return torch.einsum("yr,xr->yx", self.Y, self.X).contiguous()


class Linear(nn.Module):
    """Low-rank fully connected layer module: w = w1() * w2() + w1()."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ratio: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        rank = self._calc_from_ratio(ratio, input_dim, output_dim)
        self.w1 = LowRankNN(input_dim, output_dim, rank)
        self.w2 = LowRankNN(input_dim, output_dim, rank)
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

    @staticmethod
    def _calc_from_ratio(ratio: float, input_dim: int, output_dim: int) -> int:
        r1 = int(np.ceil(np.sqrt(output_dim)))
        r2 = int(np.ceil(np.sqrt(input_dim)))
        r = np.min((r1, r2))
        r3 = math.floor((output_dim * input_dim) / (2 * (output_dim + input_dim)))
        return max(1, math.ceil((1 - ratio) * r + ratio * r3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.w1() * self.w2() + self.w1()).contiguous()
        return F.linear(x, w, self.bias)

# ============================================================================
# 2. Low-Rank Convolution Layer (Tucker-2 Decomposition via Tensor Contraction)
# ============================================================================

class LowRank(nn.Module):
    """Low-rank weight synthesis for Conv2d layer via Tucker-2 tensor contraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_rank: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.T = nn.Parameter(
            torch.empty(size=(low_rank, low_rank, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.X = nn.Parameter(torch.empty(size=(low_rank, out_channels)), requires_grad=True)
        self.Y = nn.Parameter(torch.empty(size=(low_rank, in_channels)), requires_grad=True)
        init.kaiming_normal_(self.T, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.X, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.Y, mode="fan_out", nonlinearity="relu")

    def forward(self) -> torch.Tensor:
        return torch.einsum("xyzw,xo,yi->oizw", self.T, self.X, self.Y).contiguous()


class Conv2d(nn.Module):
    """Convolutional layer with Hadamard product of two low-rank tensors (W1 * W2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        low_rank = self._calc_from_ratio(in_channels, out_channels, kernel_size, ratio)
        self.W1 = LowRank(in_channels, out_channels, low_rank, kernel_size)
        self.W2 = LowRank(in_channels, out_channels, low_rank, kernel_size)

    @staticmethod
    def _calc_from_ratio(in_channels: int, out_channels: int, kernel_size: int, ratio: float) -> int:
        r1 = int(np.ceil(np.sqrt(out_channels)))
        r2 = int(np.ceil(np.sqrt(in_channels)))
        r = np.min((r1, r2))
        num_target_params = out_channels * in_channels * (kernel_size**2)
        a, b, c = kernel_size**2, out_channels + in_channels, -num_target_params / 2
        discriminant = b**2 - 4 * a * c
        r3 = math.floor((-b + math.sqrt(discriminant)) / (2 * a))
        return max(1, math.ceil((1 - ratio) * r + ratio * r3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = (self.W1() * self.W2()).contiguous()
        return F.conv2d(x, weight=W, bias=self.bias, stride=self.stride, padding=self.padding)

# ============================================================================
# 3. Model Converter (Applies to ANY PyTorch Model: CNN, VGG, ResNet, etc.)
# ============================================================================

def convert_to_fedpara(model: nn.Module, ratio: float = 0.1) -> nn.Module:
    """Recursively replaces standard Conv2d and Linear layers with FedPara low-rank layers."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and child.groups == 1:
            setattr(
                model,
                name,
                Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size[0],
                    stride=child.stride[0],
                    padding=child.padding[0],
                    bias=(child.bias is not None),
                    ratio=ratio,
                ),
            )
        elif isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                Linear(
                    input_dim=child.in_features,
                    output_dim=child.out_features,
                    ratio=ratio,
                    bias=(child.bias is not None),
                ),
            )
        else:
            convert_to_fedpara(child, ratio=ratio)

    return model

# ============================================================================
# 4. Client Trainer and Server Strategy
# ============================================================================

class FedParaTrainer(BaseTrainer):
    """Client-side trainer for FedPara matching Flower's baseline."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)


class FedPara(FedAvg):
    """Server-side strategy for FedPara.
    In Flower's fedpara baseline, the server aggregates low-rank parameters using FedAvg."""
    pass