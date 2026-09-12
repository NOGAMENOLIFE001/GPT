"""
Training Entry Point

Usage:
    python -m scripts.train EXP-001
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import create_trainer_from_config
from core.config import load_model_config


def main():
    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument("experiment_id", help="Experiment ID (e.g., EXP-001)")
    parser.add_argument("--model", default="tiny", help="Model variant (tiny, small, medium, large)")
    parser.add_argument("--dataset", default="d1", help="Dataset name")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print(f"Starting training for {args.experiment_id}")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Device: {args.device}")

    # Create trainer
    trainer = create_trainer_from_config(
        experiment_id=args.experiment_id,
        model_variant=args.model,
        dataset_name=args.dataset,
    )

    # Resume if requested
    if args.resume:
        from training.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler, device=args.device)
        trainer.step = checkpoint.get("step", 0)
        trainer.epoch = checkpoint.get("epoch", 0)
        trainer.best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from step {trainer.step}")

    # Train
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Total steps: {results['total_steps']}")
    print(f"Total time: {results['total_time']:.1f}s")
    print(f"Tokens/sec: {results['tokens_per_sec']:.0f}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    if results['val_losses']:
        final_loss, final_ppl = results['val_losses'][-1]
        print(f"Final val loss: {final_loss:.4f}")
        print(f"Final perplexity: {final_ppl:.2f}")

    return results


if __name__ == "__main__":
    main()