"""
Complete Language Model

Combines embeddings, positional encoding, transformer blocks, and output head.
"""

import torch
import torch.nn as nn

from core.config import load_model_config
from model.embedding import TokenEmbedding, OutputProjection, create_embeddings
from model.positional import create_positional_encoding, RoPE
from model.transformer import TransformerBlock
from model.layernorm import create_norm


class LanguageModel(nn.Module):
    """
    Causal Language Model (Decoder-only Transformer).

    Architecture:
    Input IDs -> Token Embedding -> [Positional Encoding] 
        -> Transformer Blocks (x num_layers) [with RoPE in attention if enabled]
        -> Final LayerNorm
        -> Output Projection -> Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        context_length: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        positional_encoding: str = "rope",
        rope_base: float = 10000.0,
        activation: str = "gelu",
        tie_weights: bool = True,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        padding_idx: int | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.padding_idx = padding_idx

        # Token embeddings
        self.token_embedding, self.output_projection = create_embeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            tie_weights=tie_weights,
            padding_idx=padding_idx,
        )

        # Positional encoding for input (only for sinusoidal/learned)
        self.positional_encoding_type = positional_encoding
        self.use_rope = positional_encoding == "rope"

        if self.use_rope:
            # No input positional encoding when using RoPE
            # Instead, create RoPE for attention heads (per-head dimension)
            head_dim = embedding_dim // num_heads
            self.rope = RoPE(
                dim=head_dim,
                max_seq_len=context_length,
                base=rope_base,
            )
            self.pos_encoding = None
        else:
            # Use traditional positional encoding (sinusoidal or learned)
            self.pos_encoding = create_positional_encoding(
                encoding_type=positional_encoding,
                embedding_dim=embedding_dim,
                max_seq_len=context_length,
                rope_base=rope_base,
            )
            self.rope = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final norm
        self.final_norm = create_norm(norm_type, embedding_dim, norm_eps)

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            attn_mask: Optional attention mask

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq, dim]
        x = self.dropout(x)

        # Input positional encoding (only for non-RoPE)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Causal mask if not provided
        if attn_mask is None:
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, rope=self.rope)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: dict | None = None, variant: str = "tiny") -> "LanguageModel":
        """Create model from config."""
        if config is None:
            config = load_model_config(variant)

        return cls(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            hidden_dim=config["hidden_dim"],
            context_length=config["context_length"],
            dropout=config["dropout"],
            attention_dropout=config.get("attention_dropout", 0.1),
            positional_encoding=config.get("positional_encoding", "rope"),
            rope_base=config.get("rope_base", 10000.0),
            activation=config.get("activation", "gelu"),
            tie_weights=config.get("tie_weights", True),
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters by type."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by component
    embedding_params = 0
    attention_params = 0
    ffn_params = 0
    norm_params = 0
    output_params = 0

    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            embedding_params += param.numel()
        elif "attention" in name.lower() or "attn" in name.lower():
            attention_params += param.numel()
        elif "ffn" in name.lower() or "feedforward" in name.lower():
            ffn_params += param.numel()
        elif "norm" in name.lower() or "ln" in name.lower():
            norm_params += param.numel()
        elif "output" in name.lower() or "projection" in name.lower() or "head" in name.lower():
            output_params += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "embedding": embedding_params,
        "attention": attention_params,
        "ffn": ffn_params,
        "norm": norm_params,
        "output": output_params,
    }


if __name__ == "__main__":
    # Quick test
    config = load_model_config("tiny")
    model = LanguageModel.from_config(config)
    print(f"Model created:")
    print(f"  Params: {model.get_num_params():,}")
    print(f"  Config: {config}")

    # Test forward
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    print(f"  Input: {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])
    print("  Forward pass OK")