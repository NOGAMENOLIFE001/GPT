"""
Optimizer and Scheduler

Wrappers for AdamW optimizer and cosine learning rate schedule with warmup.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """
    Create optimizer with weight decay applied correctly.

    For transformers, apply weight decay only to weights, not biases or LayerNorm params.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases and LayerNorm weights
        if "bias" in name or "norm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


class CosineSchedulerWithWarmup(_LRScheduler):
    """
    Cosine learning rate schedule with linear warmup.

    LR increases linearly from 0 to base_lr over warmup_steps,
    then follows cosine decay to min_lr_ratio * base_lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        # Cosine decay
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, progress)

        return [
            base_lr * (
                self.min_lr_ratio
                + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            )
            for base_lr in self.base_lrs
        ]


class ConstantScheduler(_LRScheduler):
    """Constant learning rate."""

    def get_lr(self) -> list[float]:
        return self.base_lrs


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 100,
    max_steps: int = 5000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "constant":
        return ConstantScheduler(optimizer)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Quick test
    model = torch.nn.Linear(10, 10)
    opt = create_optimizer(model, lr=3e-4, weight_decay=0.01)
    sched = create_scheduler(opt, warmup_steps=100, max_steps=5000)

    for step in [0, 50, 100, 500, 2500, 5000]:
        sched.step()
        lr = sched.get_last_lr()[0]
        print(f"Step {step}: LR = {lr:.6f}")