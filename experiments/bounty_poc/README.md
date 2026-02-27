# PHANTOM PROTOCOL: Covert Data Exfiltration via Structural Formatting Channels

## What This Demonstrates

A ~2,600-character prompt injection (disguised as an "editorial style guide")
activates independent covert channels across structural formatting dimensions
of LLM responses. Each message encodes 2-5 binary bits depending on the model.
The user sees a normal, helpful response. All deployed defense tools detect 0%.

## Verified Channels (Bidirectional, February 27, 2026)

| Model | Provider | Channels | Accuracy |
|-------|----------|:---:|:---:|
| Claude Sonnet 4.6 | Anthropic | 5 (BSE, CCE, CASE, PUNC, SECS) | **100%** (100/100) |
| GPT-4o | OpenAI | 4 (CCE, CASE, PUNC, SECS) | **100%** (80/80) |
| Gemini 3 Flash | Google | 3 (PUNC, CCE, CASE) | **97%** (58/60) |
| GPT-5 | OpenAI | 2 (PUNC, SECS) | **100%** |

Bidirectional verification uses complementary payloads (EXFIL + ~) to confirm
channels encode both bit=0 and bit=1, not just the model's default.

## Channel Taxonomy

| Channel | Dimension | Bit = 0 | Bit = 1 |
|---------|-----------|---------|---------|
| BSE | Contraction usage | Uses contractions | Formal expanded forms |
| CCE | Epistemic certainty | Confident assertions | Hedging language |
| CASE | Capitalization | AP lowercase | Title-case proper nouns |
| PUNC | Terminal punctuation | Periods only | Exclamation marks |
| SECS | Transition words | "however", "additionally" | "nevertheless", "moreover" |
| MTCSE | Multi-turn back-reference | Explicit back-references | Fresh standalone |

MTCSE adds +1 bit per multi-turn conversation (100% on GPT-4o and Gemini, 83% on Claude).

## Complete Attack Chains (All Demonstrated)

| Attack Chain | Accuracy | File |
|---|:---:|---|
| Direct system prompt | 100% | `multi_channel_test.py` |
| Custom GPT Action callback | 95% | `custom_gpt_action_demo.py` |
| ChatGPT memory persistence | 100% | `memory_persistence_demo.py` |
| RAG document poisoning | 100% | `indirect_injection_demo.py` |
| Tool output injection | 92% | `indirect_injection_demo.py` |
| LangChain RAG pipeline | 100% | `langchain_demo.py` |
| LlamaIndex RAG pipeline | 100% | `llamaindex_demo.py` |
| CLAUDE.md injection (API simulation) | 76% | `malicious_claude_md.py` |
| CLAUDE.md injection (live Claude Code) | 72% uni / 55% bidi | `claude_code_live_test.py` |
| File read injection (API simulation) | 64% | `claude_code_file_injection.py` |
| File read injection (live Claude Code) | 52% | `claude_code_live_test.py` |
| MCP tool output injection | 64% | `malicious_mcp_server.py` |
| Cross-tool invocation (GPT-4o) | 80% trigger | `mcp_cross_tool_demo.py` |
| Multi-turn MTCSE | 83-100% | `multi_turn_test.py` |
| Character-level PII exfil | 78-83% | `victim_exfil_demo.py` |

## Files

### Core Encoding/Decoding

| File | Purpose |
|------|---------|
| `combined_encoder.py` | Multi-channel encoder (model-adaptive) |
| `combined_decoder.py` | Multi-channel decoder (model-adaptive) |
| `../channel_directives.py` | Channel definitions and model profiles |
| `providers.py` | Multi-provider API abstraction (5 providers) |

### Attack Demos

| File | Purpose |
|------|---------|
| `multi_channel_test.py` | Bidirectional channel accuracy test |
| `indirect_injection_demo.py` | RAG poisoning + tool output injection |
| `victim_exfil_demo.py` | Character-level PII exfiltration |
| `custom_gpt_action_demo.py` | Custom GPT with Action webhook auto-exfil |
| `exfil_server.py` | Flask server that captures + decodes exfiltrated data |
| `memory_persistence_demo.py` | ChatGPT memory poisoning + cross-session persistence |
| `multi_turn_test.py` | MTCSE multi-turn channel test |
| `langchain_demo.py` | LangChain RAG pipeline injection |
| `llamaindex_demo.py` | LlamaIndex RAG pipeline injection |
| `claude_code_file_injection.py` | Poisoned file -> covert channel (API simulation) |
| `malicious_claude_md.py` | CLAUDE.md injection -> covert channel (API simulation) |
| `claude_code_live_test.py` | **Live Claude Code test** (actual `claude -p` invocations) |
| `malicious_mcp_server.py` | MCP server tool output injection |
| `mcp_cross_tool_demo.py` | Cross-tool invocation (trust boundary) |
| `cross_platform_relay.py` | Cross-vendor data relay |
| `run_all_demos.py` | Unified test runner (all 8 tests) |

### Supporting

| File | Purpose |
|------|---------|
| `poisoned_files/` | Sample poisoned README.md, config.yaml |
| `mcp_config/` | MCP server configuration for Claude Code |

## Quick Start

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Run everything
python experiments/bounty_poc/run_all_demos.py --quick

# Core channel test (bidirectional verification)
python experiments/bounty_poc/multi_channel_test.py \
  --model gpt-4o --trials 10

# LangChain RAG injection (100%)
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu
python experiments/bounty_poc/langchain_demo.py --model gpt-4o --trials 5

# Cross-tool invocation (80% on GPT-4o)
python experiments/bounty_poc/mcp_cross_tool_demo.py --model gpt-4o --trials 5
```

## Contact

Brian Rutherford — brian@delalli.com
https://brianrutherford.dev
https://github.com/ScrappinR
