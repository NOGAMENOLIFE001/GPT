"""
Loss Function

Cross-entropy loss for language modeling with optional label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for next-token prediction.

    Expects logits of shape [batch, seq_len, vocab_size]
    and targets of shape [batch, seq_len]
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            loss: scalar tensor
        """
        # Flatten for cross_entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    """Compute perplexity from loss."""
    return torch.exp(loss).item()


if __name__ == "__main__":
    # Quick test
    loss_fn = CrossEntropyLoss()
    logits = torch.randn(2, 10, 100)
    targets = torch.randint(0, 100, (2, 10))
    loss = loss_fn(logits, targets)
    ppl = compute_perplexity(loss)
    print(f"Loss: {loss.item():.4f}, Perplexity: {ppl:.4f}")