"""
Training Loop

Main training orchestration with logging, evaluation, and checkpointing.
"""

import time
import torch
from pathlib import Path
from typing import Any

from training.loss import CrossEntropyLoss, compute_perplexity
from training.optimizer import create_optimizer, create_scheduler
from training.checkpoint import save_checkpoint, cleanup_old_checkpoints
from training.dataset import create_datasets_from_config, create_dataloaders
from model.model import LanguageModel, count_parameters
from core.config import load_training_config, load_model_config
from core.seed import set_seed


class Trainer:
    """
    Main training loop for language model.
    """

    def __init__(
        self,
        model: LanguageModel,
        train_loader: Any,
        val_loader: Any,
        config: dict,
        device: str = "cpu",
        log_dir: str = "logs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        self.max_steps = config["max_steps"]
        self.eval_interval = config["eval_interval"]
        self.eval_iters = config["eval_iters"]
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.save_dir = Path(config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_grad_norm = config["max_grad_norm"]
        self.keep_last_n = config.get("keep_last_n", 3)

        # Loss
        self.loss_fn = CrossEntropyLoss()

        # Optimizer and scheduler
        self.optimizer = create_optimizer(
            model,
            optimizer_type=config["optimizer"],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config["betas"],
            eps=config["eps"],
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config["scheduler"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            min_lr_ratio=config["min_lr_ratio"],
        )

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.log"

        # Early stopping config
        self.early_stopping = config.get("early_stopping", False)
        self.patience = config.get("patience", 500)
        self.min_delta = config.get("min_delta", 0.0)
        self.early_stop_counter = 0

        # State
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # Log initial info
        self._log(f"Model parameters: {count_parameters(model)}")
        self._log(f"Device: {device}")
        self._log(f"Max steps: {self.max_steps}")
        self._log(f"Batch size: {config['batch_size']}")
        self._log(f"Context length: {config.get('max_seq_len', 'N/A')}")
        if self.early_stopping:
            self._log(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")

    def _log(self, message: str) -> None:
        """Log message to file and console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)

        # Forward
        logits = self.model(input_ids)
        loss = self.loss_fn(logits, target_ids)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), grad_norm.item()

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_iters:
                break

            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss, compute_perplexity(torch.tensor(avg_loss))

    def save_checkpoint(self, is_best: bool = False) -> str:
        """Save current checkpoint."""
        filename = f"step_{self.step}.pt"
        if is_best:
            filename = "best.pt"

        path = self.save_dir / filename
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            self.epoch,
            self.train_losses[-1] if self.train_losses else 0.0,
            self.val_losses[-1][0] if self.val_losses else None,
            self.config,
        )
        cleanup_old_checkpoints(self.save_dir, self.keep_last_n)
        return str(path)

    def train(self) -> dict[str, Any]:
        """Main training loop."""
        self._log("Starting training...")

        start_time = time.time()
        total_tokens = 0

        while self.step < self.max_steps:
            self.epoch += 1

            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break

                step_start = time.time()
                loss, grad_norm = self.train_step(batch)
                step_time = time.time() - step_start

                self.train_losses.append(loss)
                self.step += 1

                # Token counting
                batch_size, seq_len = batch["input_ids"].shape
                total_tokens += batch_size * seq_len

                # Logging
                if self.step % self.log_interval == 0:
                    tokens_per_sec = (batch_size * seq_len) / step_time
                    lr = self.scheduler.get_last_lr()[0]
                    self._log(
                        f"Step {self.step}/{self.max_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"GradNorm: {grad_norm:.4f} | "
                        f"LR: {lr:.6f} | "
                        f"Tokens/sec: {tokens_per_sec:.0f} | "
                        f"Step time: {step_time*1000:.1f}ms"
                    )

                # Evaluation
                if self.step % self.eval_interval == 0:
                    val_loss, val_ppl = self.evaluate()
                    self.val_losses.append((val_loss, val_ppl))
                    self._log(f"  VAL: Loss={val_loss:.4f}, PPL={val_ppl:.2f}")

                    # Save best
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        self._log(f"  New best model saved (val_loss={val_loss:.4f})")

                    # Early stopping check
                    if self.early_stopping:
                        if val_loss < self.best_val_loss - self.min_delta:
                            self.early_stop_counter = 0
                        else:
                            self.early_stop_counter += self.eval_interval
                            self._log(f"  Early stopping counter: {self.early_stop_counter}/{self.patience}")
                            if self.early_stop_counter >= self.patience:
                                self._log(f"Early stopping triggered at step {self.step}")
                                return {
                                    "total_steps": self.step,
                                    "total_time": time.time() - start_time,
                                    "train_losses": self.train_losses,
                                    "val_losses": self.val_losses,
                                    "best_val_loss": self.best_val_loss,
                                    "tokens_per_sec": total_tokens / (time.time() - start_time),
                                    "early_stopped": True,
                                    "early_stop_step": self.step,
                                }

                # Checkpoint
                if self.step % self.checkpoint_interval == 0:
                    self.save_checkpoint()

        # Final evaluation
        val_loss, val_ppl = self.evaluate()
        self.val_losses.append((val_loss, val_ppl))
        self._log(f"FINAL VAL: Loss={val_loss:.4f}, PPL={val_ppl:.2f}")

        # Final checkpoint
        self.save_checkpoint()

        total_time = time.time() - start_time
        self._log(f"Training completed in {total_time:.1f}s ({total_tokens/total_time:.0f} tokens/sec)")

        return {
            "total_steps": self.step,
            "total_time": total_time,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "tokens_per_sec": total_tokens / total_time,
            "early_stopped": False,
            "early_stop_step": None,
        }


def create_trainer_from_config(
    experiment_id: str = "EXP-001",
    model_variant: str = "tiny",
    dataset_name: str = "d1",
) -> Trainer:
    """Create trainer from experiment config."""
    # Load configs
    model_config = load_model_config(model_variant)
    training_config = load_training_config()

    # Override save_dir and log_dir with experiment-specific paths
    training_config = training_config.copy()
    training_config["save_dir"] = f"checkpoints/{experiment_id}"
    training_config["log_dir"] = f"logs/{experiment_id}"

    # Create datasets and tokenizer
    train_ds, val_ds, tokenizer = create_datasets_from_config(dataset_name, model_variant)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
    )

    # Create model
    model = LanguageModel.from_config(model_config)

    # Set seed
    set_seed(training_config["seed"])

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=training_config["device"],
        log_dir=training_config["log_dir"],
    )

    return trainer


if __name__ == "__main__":
    # Quick test
    trainer = create_trainer_from_config("EXP-TEST", "tiny", "d1")
    print("Trainer created successfully")
    print(f"Model params: {count_parameters(trainer.model)}")