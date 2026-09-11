# AIOS Architecture v0.1

## Project Name
**AIOS** — Artificial Intelligence Operating System

## Philosophy
Build a complete AI system from scratch, one experiment at a time. No external AI APIs. Every capability must be proven by experiment.

## Hardware Constraints
| Component | Specification | Impact |
|-----------|--------------|--------|
| CPU | AMD Ryzen 5 5500U (6P+2E, 12 threads) | Model must be CPU-optimized |
| RAM | 5.8 GB | Extremely tight memory budget |
| GPU | None | CPU-only training; no CUDA, no AMP |
| Storage | 47 GB free (D:) | Small checkpoints, streaming data |
| Python | 3.14.3 | |
| Framework | PyTorch 2.13.0 (CPU-only) | |
| NumPy | 2.5.2 | |

## System Architecture (v0.1 — Foundation)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER                                           │
└──────────┬────────────────────────────┬────────────────────────────┘
           │                            │
           ▼                            ▼
┌────────────────────┐       ┌─────────────────────┐
│  INPUT PROCESSOR    │       │  CONTEXT MANAGER    │
│  (tokenizer)        │       │  (window, trunc.)   │
└──────────┬─────────┘       └──────────┬──────────┘
           │                            │
           └───────────────┬────────────┘
                           ▼
                    ┌──────────────┐
                    │   AGENT CORE │
                    │   (planned)  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    MODEL     │
                    │  (transformer)│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   INFERENCE  │
                    │  (decoding)  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ACTION /    │
                    │  RESPONSE    │
                    └──────┬───────┘
                           │
                    ┌──────┴──────┐
                    │ FEEDBACK    │
                    │  → MEMORY   │
                    │  → LEARNING │
                    └─────────────┘
```

## Component Architecture

### 1. Core (`core/`)
- `config.py` — YAML config loader with validation
- `logging.py` — structured logging to file and console
- `seed.py` — deterministic seed management
- `utils.py` — file I/O, serialization helpers

### 2. Tokenizer (`tokenizer/`)
- `base.py` — Tokenizer interface (abstract base)
- `char.py` — Character-level tokenizer
- `byte.py` — Byte-level tokenizer
- `bpe.py` — Byte-Pair Encoding tokenizer
- `vocab.py` — Vocabulary management

### 3. Model (`model/`)
- `embedding.py` — Token embedding layer
- `positional.py` — Positional encoding (learned, sinusoidal, RoPE)
- `attention.py` — Multi-head self-attention with masking
- `feedforward.py` — Feed-forward network
- `layernorm.py` — Layer normalization (manual implementation)
- `transformer.py` — Transformer block
- `model.py` — Full language model

### 4. Training (`training/`)
- `dataset.py` — Dataset abstraction and data loading
- `dataloader.py` — Batched dataloader
- `optimizer.py` — Optimizer wrapper (AdamW)
- `scheduler.py` — Learning rate scheduler (cosine + warmup)
- `trainer.py` — Main training loop
- `loss.py` — Cross-entropy loss
- `checkpoint.py` — Checkpoint save/load

### 5. Inference (`inference/`)
- `engine.py` — Generation engine
- `decoding.py` — Greedy, top-k, top-p, temperature
- `streaming.py` — Streaming generation

### 6. Evaluation (`evaluation/`)
- `metrics.py` — Loss, perplexity, accuracy
- `benchmarks.py` — Benchmark definitions
- `self_eval.py` — Self-evaluation engine

### 7. Agent (`agent/`)
*(Phase 2 — not implemented in v0.1)*
- `loop.py` — Agent loop (Observe → Plan → Act → Evaluate)
- `planner.py` — Task planning
- `reasoner.py` — Chain-of-thought reasoning

### 8. Memory (`memory/`)
*(Phase 2 — not implemented in v0.1)*
- `short_term.py` — Context window memory
- `long_term.py` — Fact storage
- `episodic.py` — Episode storage
- `semantic.py` — Knowledge storage

### 9. Tools (`tools/`)
*(Phase 2 — not implemented in v0.1)*
- `base.py` — Tool interface
- `calculator.py` — Calculator tool
- `filesystem.py` — Filesystem tool
- `python_exec.py` — Python code executor

### 10. Datasets (`datasets/`)
- Data pipeline: Clean → Filter → Deduplicate → Tokenize → Shard → Split

### 11. Experiments (`experiments/`)
- `EXP-XXX/config.yaml` — Experiment config
- `EXP-XXX/results.json` — Metrics
- `EXP-XXX/report.md` — Analysis report

## Memory Budget Analysis (5.8 GB RAM)

Critical constraint: with 5.8 GB RAM, we must be extremely careful.

| Component | Estimated Usage |
|-----------|----------------|
| Python interpreter + PyTorch | ~1.5 GB |
| OS overhead | ~1.0 GB |
| Model (1M params, fp32) | ~4 MB |
| Model (10M params, fp32) | ~40 MB |
| Gradients | ~4 MB (1M) / ~40 MB (10M) |
| Optimizer states (AdamW) | ~8 MB / ~80 MB |
| Batch (seq=64, batch=4) | ~1 KB |
| Dataset cache | ~100 MB max |
| **Total available for ML** | ~3.3 GB |
| **Safe operating margin** | ~2.0 GB |

Conclusion: Models up to ~10M params are feasible. No batch > 32. Seq length max 256.

## Versioning Strategy
- v0.x.x = research/experimental phase
- v1.0.0 = stable tiny model with training/inference
- v2.0.0 = full agent system
- v3.0.0 = multimodal

## Experiment Standards
- One independent variable at a time
- Fixed seed for reproducibility
- Record hardware, software versions, training duration
- Document in `experiments/EXP-XXX/`
- Update `docs/RESEARCH_LOG.md` after each experiment
