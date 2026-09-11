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

(None yet)

## Experiment Index

| EXP ID | Title | Date | Status | Decision |
|--------|-------|------|--------|----------|
| (none) | | | | |

---

## Failure Database

*(Records all bugs and failed experiments with root cause analysis)*

| BUG ID | EXP ID | Symptom | Resolution |
|--------|--------|---------|-----------|
| (none) | | | |

---

## Architecture Decision Records (ADRs)

| ADR ID | Decision | Date | Status |
|--------|----------|------|--------|
| ADR-001 | Python 3.14 + PyTorch CPU-only | 2026-09-11 | ACCEPTED |
| ADR-002 | YAML for all configuration | 2026-09-11 | ACCEPTED |
| ADR-003 | BPE tokenizer as primary | PENDING | PENDING |
| ADR-004 | RoPE positional encoding | PENDING | PENDING |
| ADR-005 | AdamW optimizer | PENDING | PENDING |

---

## Performance Benchmark Results

| Model | Params | Dataset | Trained Tokens | Val Loss | Perplexity | Tokens/sec |
|-------|--------|---------|----------------|----------|------------|------------|
| (none) | | | | | | |

---

## Research Questions

| # | Question | Priority | Status |
|---|----------|----------|--------|
| 1 | What is the minimal viable model size on this hardware? | CRITICAL | OPEN |
| 2 | Can a tiny model learn coherent text generation? | CRITICAL | OPEN |
| 3 | What tokenizer achieves best compression/memory tradeoff? | HIGH | OPEN |
| 4 | What is the optimal LR for CPU training? | MEDIUM | OPEN |
| 5 | How does context length affect convergence? | MEDIUM | OPEN |
