# EXP-001 Report — First Training Pipeline & Tiny Language Model

**Date:** 2026-09-11  
**Phase:** 1 — Foundation  
**Status:** COMPLETE  
**Baseline:** BASELINE-v0.0.1

---

## Hypothesis

A minimal transformer language model (~400K params) can be trained from scratch on CPU-only hardware and will demonstrate convergent training loss, producing a functional text generation pipeline.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Tiny: 1024 vocab, 128 dim, 2 layers, 2 heads, 256 FFN, context 64 |
| **Parameters** | 395,264 |
| **Tokenizer** | BPE, 1024 vocab, byte-level |
| **Dataset** | D1: 1 text file, ~11K chars, ~3.4K tokens |
| **Train/Val Split** | 90/10 (token-level) |
| **Optimizer** | AdamW, LR=3e-4, WD=0.01, β=(0.9, 0.95) |
| **Scheduler** | Cosine + 100 step warmup, min_lr_ratio=0.1 |
| **Batch Size** | 4 |
| **Max Steps** | 5000 |
| **Eval Interval** | 250 steps |
| **Device** | CPU (AMD Ryzen 5 5500U) |
| **Dtype** | float32 |
| **Seed** | 42 |

---

## Training Results

### Loss Curves

| Step | Train Loss | Val Loss | Perplexity | LR |
|------|-----------|----------|------------|-----|
| 10 | 6.94 | - | - | 3.3e-5 |
| 250 | 2.07 | 6.77 | 874 | 1.6e-4 |
| 500 | 1.68 | 6.59 | 724 | 2.8e-4 |
| 750 | 1.34 | 6.51 | 670 | 2.8e-4 |
| 1000 | 1.17 | **5.42** | 225 | 2.8e-4 |
| 1250 | 1.03 | 5.58 | 265 | 2.7e-4 |
| 1500 | 0.87 | 6.58 | 724 | 2.6e-4 |
| 2000 | 0.75 | 7.21 | 1350 | 2.1e-4 |
| 2500 | 0.67 | 7.74 | 2287 | 1.7e-4 |
| 3000 | 0.53 | 8.05 | 3149 | 1.3e-4 |
| 3500 | 0.38 | 8.29 | 3998 | 8.8e-5 |
| 4000 | 0.26 | 8.46 | 4738 | 5.7e-5 |
| 4500 | 0.23 | 8.55 | 5185 | 3.7e-5 |
| 5000 | 0.22 | 8.63 | 5605 | 3.0e-5 |

### Key Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Final Train Loss | 0.22 | < 2.0 | ✅ EXCEEDED |
| Best Val Loss | 5.42 (step ~1000) | < 2.5 | ❌ FAILED |
| Final Val Loss | 8.63 | < 2.5 | ❌ FAILED |
| Best Perplexity | 225 | < 12 | ❌ FAILED |
| Training Time | 267s (4.5 min) | < 2 hours | ✅ PASSED |
| Peak RAM | ~1.2 GB | < 2 GB | ✅ PASSED |
| Tokens/sec | 4,791 | - | - |
| Checkpoint Size | 4.8 MB | < 5 MB | ✅ PASSED |

---

## Analysis

### What Worked ✅

1. **Full pipeline functional**: Data → Tokenizer → Model → Training → Checkpoint → Inference → Generation
2. **Training converges**: Train loss monotonically decreases from 6.94 → 0.22
3. **Tokenizer works**: BPE learns merges, encodes/decodes correctly
4. **RoPE attention works**: No shape errors, proper broadcasting
5. **Checkpointing works**: Save/load with optimizer/scheduler state
6. **CPU training viable**: ~4.8K tokens/sec on Ryzen 5500U
7. **Memory within budget**: Peak ~1.2 GB RAM

### What Failed ❌

1. **Severe overfitting**: Validation loss increases from 6.77 → 8.63 while train loss drops to 0.22
2. **Best val loss (5.42) far above target (2.5)**: Model memorizes tiny dataset
3. **Perplexity extremely high**: 5605 at end, best 225 (target < 12)
4. **Generated text incoherent**: Hallucinates, mixes concepts from training data

### Root Causes

| Cause | Impact | Evidence |
|-------|--------|----------|
| **Dataset too small** | ~3.4K tokens for 395K params = 116 params/token | Val loss diverges early |
| **Model too large for data** | 395K params need ~40M tokens for proper training | Train loss → 0.22 (memorization) |
| **Training too long** | 5000 steps on tiny data = massive overfitting | Best val at step ~1000 |
| **No regularization** | Only dropout=0.1, WD=0.01 insufficient | Gap between train/val loss huge |

---

## Generated Samples (Step 5000)

```
Prompt: "Machine learning"
Output: "Machine learning is the study of algorithms that improve through experience.
Deep learning uses neural networks with multiple l..."

Prompt: "The quick brown"
Output: "The quick brown to over beginning there was nothervesine-le step.
U training. Detection but code generation uses dra"
```

---

## Decision

**MODIFY** — Continue with architecture but fix data/model scale mismatch.

### Next Steps

1. **EXP-002**: Scale dataset (target 100K+ tokens) — add more text files
2. **EXP-003**: Model scaling sweep — test tiny (1M), small (10M) with adequate data
3. **EXP-004**: Regularization experiments — dropout, weight decay, label smoothing
4. **EXP-005**: Early stopping based on val loss

### Lessons Learned

1. **Data scales with parameters**: Need ~100 tokens/param minimum for language modeling
2. **Early stopping critical**: Best model at step 1000, not 5000
3. **Val loss is the truth**: Train loss → 0 means memorization, not learning
4. **CPU training is viable**: 4.8K tok/s sufficient for small experiments

---

## Artifacts

| Artifact | Path |
|----------|------|
| Config | `experiments/EXP-001/config.yaml` |
| Best Checkpoint | `checkpoints/EXP-001/step_1000.pt` (deleted by keep_last_n=3) |
| Final Checkpoint | `checkpoints/EXP-001/step_5000.pt` |
| Training Log | `logs/EXP-001/training.log` |
| Tokenizer | `datasets/d1_clean_text/tokenizer.json` |

---

## Next Experiment

**EXP-002**: Data Scaling — Increase D1 dataset to ~100K tokens, retrain with early stopping at best val loss.