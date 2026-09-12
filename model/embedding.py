"""
Token Embedding Layer

Maps token IDs to dense vectors.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding with optional weight tying."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: [batch, seq] -> [batch, seq, dim]"""
        return self.embedding(input_ids)

    def get_weight(self) -> torch.Tensor:
        """Get embedding weight matrix for weight tying."""
        return self.embedding.weight


class OutputProjection(nn.Module):
    """Output projection (language modeling head)."""

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        if weight is not None:
            # Weight tying: use provided weight (transposed)
            self.weight = weight
            self.bias = None
        else:
            # Separate output projection
            self.projection = nn.Linear(embedding_dim, vocab_size, bias=False)
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [batch, seq, dim] -> [batch, seq, vocab]"""
        if hasattr(self, "projection"):
            return self.projection(x)
        else:
            # Weight tied: x @ weight.T
            return torch.matmul(x, self.weight.t())


def create_embeddings(
    vocab_size: int,
    embedding_dim: int,
    tie_weights: bool = True,
    padding_idx: int | None = None,
) -> tuple[TokenEmbedding, OutputProjection]:
    """Create token embedding and output projection, optionally tied."""
    token_emb = TokenEmbedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    if tie_weights:
        output_proj = OutputProjection(embedding_dim, vocab_size, weight=token_emb.get_weight())
    else:
        output_proj = OutputProjection(embedding_dim, vocab_size)

    return token_emb, output_proj