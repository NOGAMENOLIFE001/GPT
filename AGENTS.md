# AGENTS.md — AI From Scratch Project

## Project Overview

**Project Name:** AIOS (Artificial Intelligence Operating System / собственный ИИ с нуля)

**Philosophy:** Build a complete AI system from scratch, one experiment at a time. No external AI APIs. Every capability must be proven by experiment.

**Constraints:**
- No external AI inference APIs (OpenAI, Gemini, Claude, etc.)
- Own model, tokenizer, training pipeline, inference engine
- Hardware: CPU-only (AMD Ryzen 5 5500U), 5.8 GB RAM, no GPU
- Python 3.14 + PyTorch 2.13.0 (CPU) + NumPy 2.5.2

## Development Workflow

Every experiment follows:
`HYPOTHESIS → DESIGN → IMPLEMENTATION → TRAIN → EVALUATION → ANALYSIS → DECISION → NEXT EXPERIMENT`

Each experiment lives in `experiments/EXP-XXX/` with:
- `config.yaml` — full reproducibility config
- `results.json` — metrics
- `report.md` — analysis

## Coding Standards

- All config values in `configs/` — never hardcode
- Every component has unit tests in `tests/`
- Use type hints (Python 3.14)
- Simple working solutions over complex theoretical ones
- Document failures as learning, not setbacks

## Commands

- `python -m scripts.train <experiment_id>` — run training
- `python -m scripts.evaluate <model_checkpoint>` — run evaluation
- `python -m scripts.inference <model_checkpoint> --prompt "..."` — run inference
- `python -m pytest tests/ -v` — run all tests

## Key Directories

| Dir | Purpose |
|-----|---------|
| `core/` | Core utilities, config loader, logging, seeds |
| `model/` | Neural model components (attention, transformer, etc.) |
| `tokenizer/` | Custom tokenizer implementations |
| `training/` | Training loop, optimizer, scheduler, dataloader |
| `inference/` | Inference engine, decoding strategies |
| `agent/` | Agent loop, planning, reasoning |
| `memory/` | Short-term, long-term, episodic, semantic memory |
| `tools/` | Tool interface (calculator, filesystem, code executor) |
| `evaluation/` | Benchmarks, metrics, self-evaluation |
| `datasets/` | Data pipeline, cleaning, sharding |
| `experiments/` | Individual experiment folders (EXP-001, etc.) |
| `configs/` | YAML configuration files |
| `logs/` | Training logs, experiment logs |
| `checkpoints/` | Model checkpoints |
| `tests/` | Unit and integration tests |
| `docs/` | Architecture, research log, documentation |
