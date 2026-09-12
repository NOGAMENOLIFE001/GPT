"""
Feed-Forward Network (FFN)

Standard transformer FFN with GELU activation.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Structure: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: [batch, seq, dim] -> [batch, seq, dim]"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (used in PaLM, LLaMA, etc.).

    Structure: (x @ W1) * SiLU(x @ W2) @ W3
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass."""
        gate = self.w1(x)
        up = self.w2(x)
        x = torch.nn.functional.silu(gate) * up
        x = self.w3(x)
        x = self.dropout(x)
        return x


def create_feedforward(
    embedding_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    activation: str = "gelu",
    use_swiglu: bool = False,
) -> nn.Module:
    """Factory for FFN modules."""
    if use_swiglu:
        return SwiGLU(embedding_dim, hidden_dim, dropout)
    else:
        return FeedForward(embedding_dim, hidden_dim, dropout, activation)