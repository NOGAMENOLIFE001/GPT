"""
Layer Normalization

Manual implementation for full control and understanding.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes over the last dimension.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [..., normalized_shape]
        Returns:
            normalized: [..., normalized_shape]
        """
        # Compute mean and variance over last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used in LLaMA, PaLM, etc. More efficient than LayerNorm.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm forward: x * weight / sqrt(mean(x^2) + eps)"""
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


def create_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "layernorm":
        return LayerNorm(dim, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")