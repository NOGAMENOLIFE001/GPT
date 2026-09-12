"""
Multi-Head Attention with Causal Masking

Implements scaled dot-product attention with multiple heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Supports RoPE positional encoding applied to Q and K.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, embedding_dim]
            attn_mask: [batch, 1, seq_len, seq_len] or [seq_len, seq_len] causal mask
            rope: Optional RoPE module to apply to Q and K

        Returns:
            output: [batch, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K if provided
        if rope is not None:
            # RoPE expects [..., seq_len, head_dim] - we have [batch, num_heads, seq, head_dim]
            # Apply to each head independently
            q = q.transpose(1, 2)  # [batch, seq, num_heads, head_dim]
            k = k.transpose(1, 2)
            q, k = rope.apply_to_qk(q, k)
            q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
            k = k.transpose(1, 2)

        # Scaled dot-product attention
        # attn = softmax(Q @ K^T / sqrt(d_k)) @ V
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]

        # Apply causal mask
        if attn_mask is not None:
            # Handle different mask shapes
            if attn_mask.dim() == 2:
                # [seq, seq] -> [1, 1, seq, seq]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch, seq, seq] -> [batch, 1, seq, seq]
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        # Output projection
        output = self.out_proj(attn_output)
        return output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask: [seq_len, seq_len] with 1 for allowed positions, 0 for masked."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def create_attention_mask(
    seq_len: int,
    device: torch.device,
    mask_type: str = "causal",
) -> torch.Tensor:
    """Create attention mask for different types."""
    if mask_type == "causal":
        return create_causal_mask(seq_len, device)
    elif mask_type == "none":
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")