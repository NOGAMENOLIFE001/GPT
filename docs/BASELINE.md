# AIOS Initial Baseline

## Hardware Constraints
- **CPU:** AMD Ryzen 5 5500U (12 threads)
- **RAM:** 5.8 GB (no swap expansion feasible)
- **GPU:** None (CPU-only PyTorch)
- **Storage:** 47 GB free on D:

## Memory Budget
| Component | Allocation |
|-----------|-----------|
| OS + Python + PyTorch | ~2.5 GB |
| Available for ML | ~3.3 GB |
| Safe operating margin | ~2.0 GB |

## Baseline Model Specification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Size class** | `tiny` | Fits in 2 GB RAM |
| **vocab_size** | 1024 | BPE tokenizer; balances coverage and memory |
| **embedding_dim** | 128 | Small enough for CPU training |
| **num_layers** | 2 | Minimal depth for next-token prediction |
| **num_heads** | 2 | Must divide embedding_dim evenly |
| **hidden_dim** | 256 | 2x embedding_dim (standard ratio) |
| **context_length** | 64 | Short context due to CPU constraints |
| **dropout** | 0.1 | Standard regularization |
| **positional_encoding** | `rope` | Modern, efficient |
| **activation** | `gelu` | Standard for transformers |
| **tie_weights** | true | Saves ~25% parameters on output head |
| **Estimated params** | ~0.4M | Well within memory budget |

## Baseline Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **optimizer** | AdamW | Standard for transformers |
| **learning_rate** | 3e-4 | Standard range for tiny models |
| **weight_decay** | 0.01 | Mild regularization |
| **warmup_steps** | 100 | 2% of total steps |
| **scheduler** | cosine | Smooth decay |
| **batch_size** | 4 | Limited by CPU memory |
| **max_seq_length** | 64 | Matches model context |
| **max_steps** | 5000 | Feasible training duration |
| **eval_interval** | 250 | Frequent evaluation |
| **gradient_accumulation** | 1 | No accumulation needed |
| **max_grad_norm** | 1.0 | Gradient clipping |
| **seed** | 42 | Standard for reproducibility |
| **device** | cpu | No GPU available |
| **dtype** | float32 | CPU-only; no AMP |

## Baseline Dataset (D1)

| Parameter | Value |
|-----------|-------|
| **Dataset** | D1 — clean English text |
| **Source** | Curated sample texts |
| **Max tokens** | 100,000 (after tokenization) |
| **Train split** | 90% |
| **Val split** | 10% |
| **Min doc length** | 10 tokens |
| **Max doc length** | 64 tokens (truncate) |

## Expected Outcomes

| Metric | Target | Acceptable |
|--------|--------|-----------|
| **Train loss** | < 2.0 | < 3.0 |
| **Val loss** | < 2.5 | < 3.5 |
| **Perplexity** | < 12.0 | < 30.0 |
| **Training time** | < 2 hours | < 4 hours |
| **Peak RAM** | < 1.5 GB | < 2.0 GB |
| **Model size** | ~1.6 MB | ~2 MB |

## Baseline Definition

The baseline serves as the reference point for all subsequent experiments. Any architectural change will be compared against this configuration (EXP-001).

**Baseline ID:** BASELINE-v0.0.1
