# AIOS Development & Experiment Roadmap

## Phase 0 — Project Foundation (v0.0.1)
| ID | Milestone | Status |
|----|-----------|--------|
| P0-1 | Directory structure | DONE |
| P0-2 | Configuration system (YAML) | DONE |
| P0-3 | Project architecture document | DONE |
| P0-4 | Development roadmap | DONE |
| P0-5 | Versioning strategy | DONE |
| P0-6 | Initial baseline definition | DONE |
| P0-7 | EXP-001 specification | DONE |
| P0-8 | Git repository initialization | PENDING |

## Phase 1 — Tokenizer (v0.0.2)
| Exp ID | Experiment | Hypothesis | Priority |
|--------|-----------|-----------|----------|
| T-001 | Character tokenizer | Char-level tokenizer provides perfect vocabulary coverage but long sequences | High |
| T-002 | Byte tokenizer | Byte-level tokenizer supports all languages/unicode without unknown tokens | High |
| T-003 | BPE tokenizer | BPE with 1K vocab achieves good compression-speed tradeoff | High |
| T-004 | Vocab size sweep | Larger vocab reduces sequence length but increases memory | Medium |

## Phase 2 — Model Components (v0.0.3)
| Exp ID | Experiment | Hypothesis | Priority |
|--------|-----------|-----------|----------|
| M-001 | Tiny LM (1M params) | Minimal transformer can learn next-token prediction | Critical |
| A-001 | Attention head count | More heads improve representation but have diminishing returns | Medium |
| P-001 | Positional encoding | RoPE outperforms learned/sinusoidal on short contexts | Medium |

## Phase 3 — First Training (v0.1.0)
| Exp ID | Experiment | Hypothesis | Priority |
|--------|-----------|-----------|----------|
| EXP-001 | Baseline training | Tiny LM can be trained and generates coherent text | Critical |
| TR-001 | LR sweep | LR 3e-4 gives best convergence for tiny model | Medium |
| TR-002 | Batch size sweep | Smaller batch with accumulation works better on CPU | Medium |

## Phase 4 — Scaling (v0.2.0)
| Exp ID | Experiment | Hypothesis |
|--------|-----------|-----------|
| SC-001 | Scale to 10M params | Larger model shows measurable perplexity improvement |
| SC-002 | Data quality experiments | Clean data improves convergence speed |

## Phase 5 — Inference Engine (v0.3.0)
| Exp ID | Experiment | Hypothesis |
|--------|-----------|-----------|
| IN-001 | Decoding strategies | Top-p sampling gives best diversity-coherence tradeoff |
| IN-002 | Streaming generation | Streaming enables interactive applications |

## Phase 6 — Instruction Following (v0.4.0)
| Exp ID | Experiment | Hypothesis |
|--------|-----------|-----------|
| INSTR-001 | SFT on instructions | Model learns to follow format and intent |
| INSTR-002 | Conversation formatting | Role-based chat format improves behavior |

## Phase 7 — Agent System (v1.0.0)
| Exp ID | Experiment | Hypothesis |
|--------|-----------|-----------|
| AG-001 | Basic agent loop | Observe-Plan-Act loop enables tool use |
| AG-002 | Memory integration | Memory improves task completion |
| AG-003 | Self-evaluation | Critic step reduces errors |

## Phase 8 — Multimodal (v3.0.0)
| Exp ID | Experiment | Hypothesis |
|--------|-----------|-----------|
| MV-001 | Vision encoder | Image understanding improves with projection |
| MV-002 | Audio processing | Audio-to-text works as retrieval task |

## Experiment Numbering
- T-XXX: Tokenizer experiments
- M-XXX: Model architecture experiments
- A-XXX: Attention experiments
- P-XXX: Positional encoding experiments
- TR-XXX: Training hyperparameter experiments
- EXP-XXX: Major training experiments
- SC-XXX: Scaling experiments
- IN-XXX: Inference experiments
- INSTR-XXX: Instruction following experiments
- AG-XXX: Agent experiments
- MV-XXX: Multimodal experiments

## Reproducibility Requirements
For every experiment:
1. Fixed random seed (config.yaml)
2. Fixed dataset version
3. Fixed tokenizer version
4. Full config in experiments/EXP-XXX/config.yaml
5. Results in experiments/EXP-XXX/results.json
6. Analysis in experiments/EXP-XXX/report.md
7. Research log entry in docs/RESEARCH_LOG.md
