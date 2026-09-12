import sys
sys.path.insert(0, '.')

from training.trainer import create_trainer_from_config

# Create trainer for EXP-002
trainer = create_trainer_from_config('EXP-002', 'tiny', 'd1')

print('=' * 60)
print('EXP-002: Data Scaling & Early Stopping')
print('=' * 60)
print(f'Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}')
print(f'Max steps: {trainer.max_steps}')
print(f'Batch size: {trainer.config["batch_size"]}')
print(f'Context length: {trainer.model.context_length}')
print(f'Train sequences: {len(trainer.train_loader.dataset)}')
print(f'Val sequences: {len(trainer.val_loader.dataset)}')
print(f'Early stopping: {trainer.early_stopping} (patience={trainer.patience})')
print('=' * 60)

results = trainer.train()

print('=' * 60)
print('EXP-002 COMPLETED')
print('=' * 60)
print(f'Total steps: {results["total_steps"]}')
print(f'Total time: {results["total_time"]:.1f}s')
print(f'Tokens/sec: {results["tokens_per_sec"]:.0f}')
print(f'Best val loss: {results["best_val_loss"]:.4f}')
if results['val_losses']:
    final_loss, final_ppl = results['val_losses'][-1]
    print(f'Final val loss: {final_loss:.4f}')
    print(f'Final perplexity: {final_ppl:.2f}')
print(f'Early stopped: {results.get("early_stopped", False)}')
if results.get("early_stopped"):
    print(f'Early stop step: {results.get("early_stop_step")}')