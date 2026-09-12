# AIOS Research Log

> **Philosophy:** Every experiment is an iteration. Failures are data, not setbacks.

## Log Format

Each entry follows this template:

```
## EXP-XXX — <Experiment Title>

**Date:** YYYY-MM-DD
**Phase:** <Phase N — Name>
**Status:** PLANNED | RUNNING | COMPLETE | FAILED | ABANDONED
**Hypothesis:** <One sentence>
**Metrics:** <Key measurements>
**Decision:** KEEP | MODIFY | REVERT
**Next:** <EXP-XXX or next step>
```

## Active Experiments

(None yet)

## Completed Experiments

### EXP-001 — First Training Pipeline & Tiny Language Model

**Date:** 2026-09-11  
**Phase:** 1 — Foundation  
**Status:** COMPLETE  
**Hypothesis:** Tiny transformer can learn next-token prediction on CPU  
**Metrics:** Train loss 6.94→0.22, Val loss 6.77→8.63, PPL 225→5605, 4.8K tok/s  
**Decision:** MODIFY  
**Next:** EXP-002 (Data Scaling)

### EXP-002 — Data Scaling & Early Stopping

**Date:** 2026-09-11  
**Phase:** 1 — Foundation  
**Status:** COMPLETE  
**Hypothesis:** 11x more data (~38K tokens) with early stopping eliminates overfitting  
**Metrics:** Best val loss 4.83 (vs 5.42), PPL 125 (vs 225), no val divergence, 4.6K tok/s  
**Decision:** MODIFY  
**Next:** EXP-003 or EXP-004

## Experiment Index

| EXP ID | Title | Date | Status | Decision |
|--------|-------|------|--------|----------|
| EXP-001 | First Training Pipeline & Tiny LM | 2026-09-11 | COMPLETE | MODIFY |
| EXP-002 | Data Scaling & Early Stopping | 2026-09-11 | COMPLETE | MODIFY |

---

## Failure Database

*(Records all bugs and failed experiments with root cause analysis)*

| BUG ID | EXP ID | Symptom | Resolution |
|--------|--------|---------|-----------|
| BUG-001 | EXP-001 | Catastrophic overfitting: val loss 5.42→8.63 | Fixed in EXP-002 by 11x data scaling |
| BUG-002 | EXP-001 | RoPE broadcasting error in attention | Fixed positional.py RoPE broadcasting logic |
| BUG-003 | EXP-001 | BPE tokenizer byte range error | Fixed tokenizer encoding logic |
| BUG-004 | EXP-001 | Dataset split produced 0 train sequences | Fixed token-level split instead of document-level |
| BUG-005 | EXP-002 | Best checkpoint (step 3500) deleted by keep_last_n=3 | Need dedicated best-model preservation |

---

## Architecture Decision Records (ADRs)

| ADR ID | Decision | Date | Status |
|--------|----------|------|--------|
| ADR-001 | Python 3.14 + PyTorch CPU-only | 2026-09-11 | ACCEPTED |
| ADR-002 | YAML for all configuration | 2026-09-11 | ACCEPTED |
| ADR-003 | BPE tokenizer as primary | 2026-09-11 | ACCEPTED |
| ADR-004 | RoPE positional encoding | 2026-09-11 | ACCEPTED |
| ADR-005 | AdamW optimizer | 2026-09-11 | ACCEPTED |

---

## Performance Benchmark Results

| Model | Params | Dataset | Trained Tokens | Val Loss | Perplexity | Tokens/sec |
|-------|--------|---------|----------------|----------|------------|------------|
| EXP-001 tiny | 395K | D1 (~3.4K) | 1.7M | 8.63 | 5605 | 4791 |
| EXP-002 tiny | 395K | D1 (~38K) | 19M | 4.83 | 125 | 4649 |

---

## Research Questions

| # | Question | Priority | Status |
|---|----------|----------|--------|
| 1 | What is the minimal viable model size on this hardware? | CRITICAL | OPEN |
| 2 | Can a tiny model learn coherent text generation? | CRITICAL | PARTIAL (EXP-002 improved) |
| 3 | What tokenizer achieves best compression/memory tradeoff? | HIGH | OPEN |
| 4 | What is the optimal LR for CPU training? | MEDIUM | OPEN |
| 5 | How does context length affect convergence? | MEDIUM | OPEN |
| 6 | What tokens/param ratio yields target perplexity < 12? | CRITICAL | OPEN |
| 7 | Does early stopping work with current patience? | HIGH | PARTIAL (not triggered) |

---

## Key Insights

1. **Tokens/param ratio is critical**: EXP-001 had 8.6 tokens/param → catastrophic overfitting. EXP-002 had 96 tokens/param → stable but still high perplexity. Target likely >150 tokens/param.

2. **Data scaling fixed divergence**: 11x data eliminated catastrophic val loss increase.

3. **Early stopping needs tuning**: Patience=500 eval intervals (125K steps) too high; val loss still improving at 5000 steps.

4. **Best model preservation**: keep_last_n=3 deletes best checkpoint; need dedicated best.pt preservation.

5. **Generation quality correlates with val loss**: EXP-002 (val 4.83) generates recognizable words but incoherent syntax; need val < 2.5 for fluency.