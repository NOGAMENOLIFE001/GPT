"""
Transformer Block

Combines attention, feed-forward, normalization, and residual connections.
"""

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import create_feedforward
from model.layernorm import create_norm


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block (used in GPT-2, LLaMA, etc.).

    Structure:
    x -> LN -> Attention -> Dropout -> Residual
        -> LN -> FFN -> Dropout -> Residual
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Pre-attention norm
        self.norm1 = create_norm(norm_type, embedding_dim, norm_eps)

        # Attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        # Post-attention dropout
        self.dropout1 = nn.Dropout(dropout)

        # Pre-FFN norm
        self.norm2 = create_norm(norm_type, embedding_dim, norm_eps)

        # Feed-forward
        self.ffn = create_feedforward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
        )

        # Post-FFN dropout
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: [batch, seq, dim]
            attn_mask: Optional attention mask
            rope: Optional RoPE module

        Returns:
            output: [batch, seq, dim]
        """
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attn_mask=attn_mask, rope=rope)
        x = self.dropout1(x)
        x = x + residual

        # FFN block with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual

        return x


class TransformerBlockPostLN(nn.Module):
    """
    Post-LN Transformer Block (original Vaswani et al. architecture).

    Structure:
    x -> Attention -> Dropout -> Residual -> LN
        -> FFN -> Dropout -> Residual -> LN
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_swiglu: bool = False,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = create_norm(norm_type, embedding_dim, norm_eps)

        self.ffn = create_feedforward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = create_norm(norm_type, embedding_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        """Forward pass with post-LN."""
        # Attention
        residual = x
        x = self.attention(x, attn_mask=attn_mask, rope=rope)
        x = self.dropout1(x)
        x = x + residual
        x = self.norm1(x)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.norm2(x)

        return x