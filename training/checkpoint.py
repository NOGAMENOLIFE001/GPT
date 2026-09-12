"""
Checkpoint Management

Save and load model checkpoints with optimizer, scheduler, and training state.
"""

import os
import torch
from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    config: dict,
    **kwargs,
) -> None:
    """
    Save training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        epoch: Current epoch
        train_loss: Latest training loss
        val_loss: Latest validation loss (or None)
        config: Model/training config
        **kwargs: Additional data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        **kwargs,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with training state
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return checkpoints[-1]


def cleanup_old_checkpoints(
    checkpoint_dir: str | Path,
    keep_last_n: int = 3,
) -> None:
    """Remove old checkpoints, keeping only the most recent N."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if len(checkpoints) <= keep_last_n:
        return

    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    for ckpt in checkpoints[:-keep_last_n]:
        ckpt.unlink()


if __name__ == "__main__":
    # Quick test
    import tempfile
    model = torch.nn.Linear(10, 10)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, opt, sched, step=100, epoch=1, train_loss=1.5, val_loss=1.6, config={})

        # Load back
        model2 = torch.nn.Linear(10, 10)
        opt2 = torch.optim.AdamW(model2.parameters())
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)

        ckpt = load_checkpoint(path, model2, opt2, sched2)
        print(f"Loaded step: {ckpt['step']}")
        print(f"Loaded loss: {ckpt['train_loss']}")