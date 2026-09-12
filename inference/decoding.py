"""
Decoding Strategies for Text Generation

Implements greedy, top-k, top-p (nucleus), and temperature sampling.
"""

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Sample next token from logits using various strategies.

    Args:
        logits: [batch, vocab_size] or [vocab_size]
        temperature: Temperature for sampling (1.0 = no change, <1.0 = sharper, >1.0 = flatter)
        top_k: Keep only top-k tokens (None = disable)
        top_p: Nucleus sampling - keep smallest set with cumulative prob >= top_p (None = disable)

    Returns:
        next_token: [batch] or scalar tensor
    """
    # Handle both batched and single
    is_batched = logits.dim() == 2
    if not is_batched:
        logits = logits.unsqueeze(0)  # [1, vocab]

    batch_size, vocab_size = logits.shape

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab_size)
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
        # Create mask for non-top-k
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_indices, topk_values)
        logits = mask

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Compute cumulative probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # Find cutoff
        cutoff_mask = cumulative_probs > top_p
        # Shift to keep first token above threshold
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False
        # Apply mask
        sorted_logits[cutoff_mask] = float("-inf")
        # Restore original order
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]

    if not is_batched:
        next_token = next_token.squeeze(0)

    return next_token


def greedy_decode(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Greedy decoding: pick token with highest probability."""
    return logits.argmax(dim=-1)


def top_k_decode(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Top-k sampling."""
    return sample_next_token(logits, top_k=k)


def top_p_decode(
    logits: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    return sample_next_token(logits, top_p=p)


def temperature_decode(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature sampling."""
    return sample_next_token(logits, temperature=temperature)


def sample_with_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: list[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Apply repetition penalty to logits."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    # Apply penalty to already generated tokens
    for token in set(generated_tokens):
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty

    return logits


class DecodingConfig:
    """Configuration for decoding strategy."""

    def __init__(
        self,
        strategy: str = "top_p",  # greedy, top_k, top_p, temperature
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 100,
        stop_tokens: list[int] | None = None,
    ):
        self.strategy = strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.stop_tokens = stop_tokens or []

    def decode_next(self, logits: torch.Tensor, generated: list[int]) -> torch.Tensor:
        """Apply decoding strategy to get next token."""
        # Apply repetition penalty
        if self.repetition_penalty != 1.0:
            logits = sample_with_repetition_penalty(logits, generated, self.repetition_penalty)

        if self.strategy == "greedy":
            return greedy_decode(logits)
        elif self.strategy == "top_k":
            return top_k_decode(logits, self.top_k)
        elif self.strategy == "top_p":
            return top_p_decode(logits, self.top_p)
        elif self.strategy == "temperature":
            return temperature_decode(logits, self.temperature)
        else:
            # Default to top_p
            return top_p_decode(logits, self.top_p)


if __name__ == "__main__":
    # Quick test
    logits = torch.randn(100)
    print("Greedy:", greedy_decode(logits).item())
    print("Top-k:", top_k_decode(logits, 10).item())
    print("Top-p:", top_p_decode(logits, 0.9).item())
    print("Temp:", temperature_decode(logits, 0.8).item())
    print("Combined:", sample_next_token(logits, temperature=0.8, top_k=50, top_p=0.9).item())