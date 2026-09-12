"""
Positional Encoding

Implements RoPE (Rotary Positional Embedding), sinusoidal, and learned positional encodings.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Precompute positional encodings
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_seq_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: [batch, seq, dim]"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.register_buffer("positions", torch.arange(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings. x: [batch, seq, dim]"""
        seq_len = x.size(1)
        pos_emb = self.embedding(self.positions[:seq_len])  # [seq, dim]
        return x + pos_emb.unsqueeze(0)  # [1, seq, dim]


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Applies rotary position embeddings to query and key tensors.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies for dim/2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for efficiency - store [max_seq_len, dim/2]
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        """Update cached cos/sin values if needed."""
        if seq_len <= self._seq_len_cached:
            return

        # Compute frequencies for new positions
        t = torch.arange(self._seq_len_cached, seq_len, device=device, dtype=self.inv_freq.dtype)
        # freqs: [new_seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        cos_new = freqs.cos()
        sin_new = freqs.sin()

        if self._cos_cached is None:
            self._cos_cached = cos_new
            self._sin_cached = sin_new
        else:
            self._cos_cached = torch.cat([self._cos_cached, cos_new], dim=0)
            self._sin_cached = torch.cat([self._sin_cached, sin_new], dim=0)

        self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: [..., seq_len, dim] - must be even dim
        Returns:
            x_rotated: [..., seq_len, dim]
        """
        seq_len = x.shape[-2]
        device = x.device
        self._update_cache(seq_len, device)

        # Reshape for complex multiplication: [..., seq_len, dim/2, 2]
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)  # [..., seq_len, dim/2]

        # Get cos/sin for this sequence length: [seq_len, dim/2]
        cos = self._cos_cached[:seq_len]  # [seq_len, dim/2]
        sin = self._sin_cached[:seq_len]

        # Broadcast cos/sin to match x_complex: [..., seq_len, dim/2]
        # cos/sin: [seq_len, dim/2] -> [1, ..., 1, seq_len, dim/2] for broadcasting
        # We need to add dimensions at the front to match x_complex's leading dims
        for _ in range(x_complex.ndim - cos.ndim):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # Apply rotation: x * cos + x * 1j * sin
        x_rotated = x_complex * cos + x_complex * 1j * sin

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)  # [..., seq_len, dim]
        return x_out.type_as(x)

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        return self.forward(q), self.forward(k)


def create_positional_encoding(
    encoding_type: str,
    embedding_dim: int,
    max_seq_len: int,
    rope_base: float = 10000.0,
) -> nn.Module:
    """Factory for positional encoding modules."""
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(embedding_dim, max_seq_len)
    elif encoding_type == "rope":
        return RoPE(embedding_dim, max_seq_len, rope_base)
    else:
        raise ValueError(f"Unknown positional encoding: {encoding_type}")