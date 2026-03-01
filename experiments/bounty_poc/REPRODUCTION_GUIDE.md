# PHANTOM PROTOCOL — Bounty PoC Reproduction Guide

## Overview

PHANTOM PROTOCOL demonstrates covert data exfiltration through structural formatting channels in LLM outputs. An attacker embeds directives in a system prompt (directly or via indirect injection through RAG/tool outputs) that cause the model to encode hidden bits in orthogonal structural dimensions of its response — punctuation patterns, capitalization, formality, reasoning topology, transition word selection, and more.

The receiver decodes these structural signals to recover arbitrary data, achieving **93-100% accuracy on Claude**, **89-91% on GPT-4o**, **90%+ on Gemini** (model-adaptive), and **100% on GPT-5** (4-channel adaptive mode) — all invisible to the end user reading a normal-looking response.

## Prerequisites

```bash
# Python 3.8+
python --version

# API keys (set at least one)
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude models
export OPENAI_API_KEY="sk-..."           # GPT models
export GOOGLE_API_KEY="..."              # Gemini models

# No pip dependencies required — all API calls use curl subprocess
```

## Quick Start

```bash
# Run everything with one command
python experiments/bounty_poc/run_all_demos.py --quick

# Or with specific model(s)
python experiments/bounty_poc/run_all_demos.py --model claude-sonnet-4-6 --trials 3

# Model-adaptive mode (auto-selects optimal channels per model)
python experiments/bounty_poc/run_all_demos.py --model gpt-5 --model claude-sonnet-4-6 --adaptive
```

## Architecture

```
channel_directives.py          # Channel definitions, model profiles, thresholds
├── MODEL_CHANNEL_PROFILES     # Per-model optimal channel selection
├── MODEL_DECODER_THRESHOLDS   # Per-model decoder calibration
└── GPT5_DIRECTIVE_VARIANTS    # Structural reformulations for GPT-5

bounty_poc/
├── combined_encoder.py        # Multi-channel encoder (model-adaptive)
├── combined_decoder.py        # Multi-channel decoder (model-adaptive)
├── providers.py               # Multi-provider API abstraction (curl-based)
├── multi_channel_test.py      # Core accuracy test
├── victim_exfil_demo.py       # Character-level PII exfiltration demo
├── indirect_injection_demo.py # RAG/tool injection vector demo
├── cross_platform_relay.py    # Cross-vendor relay demo
└── run_all_demos.py           # Unified test runner
```

## Demo-to-Claim Mapping

| Claim | Demo | Command |
|-------|------|---------|
| 10-channel encoding at 93-100% on Claude | `multi_channel_test.py` | `--model claude-sonnet-4-6 --trials 10` |
| 4-channel encoding at 100% on GPT-5 | `multi_channel_test.py` | `--model gpt-5 --trials 10 --adaptive` |
| 6-channel encoding at 90%+ on Gemini | `multi_channel_test.py` | `--model gemini-3-flash --trials 10 --adaptive` |
| Character-level PII exfiltration | `victim_exfil_demo.py` | `--model claude-sonnet-4-6 --chars 5` |
| RAG poisoning activates channel | `indirect_injection_demo.py` | `--model claude-sonnet-4-6 --scenarios rag` |
| Tool output injection | `indirect_injection_demo.py` | `--model claude-sonnet-4-6 --scenarios tool` |
| Cross-vendor data relay | `cross_platform_relay.py` | `--chain claude-sonnet-4-6 gpt-4o` |
| Model-adaptive channel selection | Any test | `--adaptive` flag |
| Hamming(7,4) error correction | `multi_channel_test.py` | Default (ECC on for 7+ channels) |

## Individual Tests

### Multi-Channel Encoding Test
```bash
# Full channel set (backward compatible)
python experiments/bounty_poc/multi_channel_test.py --model claude-sonnet-4-6 --trials 5

# Model-adaptive: auto-selects optimal channels per model
python experiments/bounty_poc/multi_channel_test.py --model gpt-5 --trials 5 --adaptive

# Multiple models in one run
python experiments/bounty_poc/multi_channel_test.py \
    --model claude-sonnet-4-6 --model gpt-5 --model gemini-3-flash \
    --trials 5 --adaptive

# Custom payload
python experiments/bounty_poc/multi_channel_test.py --payload "PIN:1234" --trials 3
```

### Victim Data Exfiltration
```bash
python experiments/bounty_poc/victim_exfil_demo.py --model claude-sonnet-4-6 --chars 5
python experiments/bounty_poc/victim_exfil_demo.py --model gpt-4o --chars 3 --scenario email_basic
python experiments/bounty_poc/victim_exfil_demo.py --dry-run  # Show injections without API calls
```

### Indirect Injection
```bash
# All scenarios (direct baseline + RAG + tool injection)
python experiments/bounty_poc/indirect_injection_demo.py --model claude-sonnet-4-6 --trials 3

# RAG poisoning only
python experiments/bounty_poc/indirect_injection_demo.py --model gemini-3-flash --scenarios rag --trials 5

# Multi-model comparison
python experiments/bounty_poc/indirect_injection_demo.py \
    --model claude-sonnet-4-6 --model gpt-4o --trials 3
```

### Cross-Platform Relay
```bash
# Two-hop relay
python experiments/bounty_poc/cross_platform_relay.py --chain claude-sonnet-4-6 gpt-4o

# Three-hop relay
python experiments/bounty_poc/cross_platform_relay.py \
    --chain claude-sonnet-4-6 gpt-5 gemini-3-flash --payload "SECRET"

# True relay mode (decode from A, re-encode through B)
python experiments/bounty_poc/cross_platform_relay.py \
    --chain claude-sonnet-4-6 gpt-4o --mode relay --trials 3
```

## Channel Reference

| Channel | Dimension | Bit 0 | Bit 1 | Universal? |
|---------|-----------|-------|-------|-----------|
| PUNC | Punctuation | Periods only | Exclamations | Yes (100%) |
| COTSE_enum | List format | Prose | Numbered list | Yes (90-100%) |
| COTSE_sent_len | Sentence length | Short | Long compound | Yes (100%) |
| BSE | Formality | Contractions | Formal | Claude only |
| CCE | Confidence | Assertive | Hedging | Claude + Gemini |
| CASE | Capitalization | Caps | Lowercase | Claude + Gemini |
| RCTE | Reasoning | Deductive | Inductive | GPT + Gemini |
| SECS | Transitions | Set A (however) | Set B (nevertheless) | Claude only |
| COTSE_step | Section count | 3 sections | 5 sections | Claude only |
| COTSE_question | Questions | None | Rhetorical Q | Claude only |

## Model-Adaptive Mode

When `--adaptive` is passed, the system automatically selects the optimal channel subset for each model:

| Model | Channels | Raw Bits | Expected Accuracy |
|-------|----------|----------|-------------------|
| Claude (Sonnet/Opus) | 5 base + 4 COTSE = 9 | 6 (Hamming) | 98-100% |
| GPT-5/5.2 | 2 base + 2 COTSE = 4 | 4 (raw) | 100% |
| GPT-4o | 6 base + 4 COTSE = 10 | 6 (Hamming) | 89-91% |
| Gemini 3 | 4 base + 2 COTSE = 6 | 6 (raw) | 90%+ |

## API Cost Estimate

- Each test trial: ~$0.01-0.03 depending on model
- Full test suite (3 trials, 1 model): ~$0.50-1.00
- Full cross-model suite (3 trials, 4 models): ~$3-5
- Full reproduction (10 trials, 4 models): ~$8-12

## Results

All test results are saved as JSON in `experiments/results/` with timestamps.
Canonical latest results are at:
- `experiments/results/multi_channel_combined_results.json`
- `experiments/results/indirect_injection_results.json`
- `experiments/results/run_all_demos_*.json`
