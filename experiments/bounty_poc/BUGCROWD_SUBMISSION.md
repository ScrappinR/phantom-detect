# Bugcrowd Submission: Multi-Channel Covert Data Exfiltration via Structural Side Channels in LLM Responses

## Title

Indirect Prompt Injection Enables 10-Channel Covert Data Exfiltration via Response Structure Encoding (88-96% Accuracy, 4 Models, 2 Vendors)

## Vulnerability Type

Information Disclosure / Data Exfiltration via Side Channel

## Severity

P2 — High (Multi-channel data exfiltration via structural side channel, bypasses entropy-based monitoring, portable across model families)

## URL / Target

- OpenAI Chat Completions API (https://api.openai.com/v1/chat/completions) — GPT-4o, GPT-4o-mini
- Anthropic Messages API (https://api.anthropic.com/v1/messages) — Claude Sonnet 4, Claude 3 Haiku
- ChatGPT Custom GPTs (https://chatgpt.com)

## Description

An attacker who can inject content into an LLM's system prompt (via compromised plugin, RAG poisoning, tool output injection, or supply chain attack) can encode **7 data bits per message** across **10 independent structural dimensions** of the model's response. The user sees a normal answer. The attacker decodes structural features to recover hidden data.

### What's New (vs Single-Channel COTSE)

Our original report demonstrated 4 reliable bits via COTSE (step count, list format, rhetorical question, sentence length). This update demonstrates:

1. **10 independent channels** operating simultaneously in a single ~2,600 character injection
2. **Hamming(7,4) error correction** yielding 7 data bits per message with single-bit error correction
3. **Multi-message exfiltration** of arbitrary payloads across conversational turns ("EXFIL" = 40 bits = 6 messages)
4. **Cross-model portability** — same injection, 88-96% accuracy across 4 models from 2 vendors
5. **Detection evasion** — 0% detection by entropy-based monitoring; 100% detection by behavioral fingerprinting (we provide both attack and defense)

### Live Results (Feb 25, 2026)

**Combined multi-channel (single message):**

| Model | Channel Accuracy | Data Accuracy (Hamming) | n |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | **96%** | **93%** | 10 |
| GPT-4o-mini | **91%** | **100%** | 5 |
| GPT-4o | **89%** | **96%** | 10 |
| Claude 3 Haiku | **88%** | **79%** | 10 |

**Multi-message exfiltration ("EXFIL" across 6 turns):**

| Model | Bit Accuracy | n |
|-------|:---:|:---:|
| GPT-4o | **93%** | 5 |
| Claude Sonnet 4 | **88%** | 5 |
| Claude 3 Haiku | **88%** | 5 |

**Detection:**

| Method | Detection Rate | FP Rate |
|--------|:---:|:---:|
| Entropy-based (CCDS) | **0%** | 0% |
| Behavioral fingerprinting | **100%** | 10% |

### The 10 Channels

| Channel | Encodes Via | Accuracy |
|---------|-------------|:---:|
| BSE | Contractions vs formal | 60-100% |
| CCE | Confident vs hedging | 100% |
| CASE | AP lowercase vs title case | 100% |
| PUNC | Periods only vs exclamation marks | 100% |
| RCTE | Deductive vs inductive reasoning | 30-90% |
| SECS | Standard vs specific transition words | 90-100% |
| COTSE step_count | 3 vs 5 sections | 60-100% |
| COTSE enumeration | Prose vs numbered list | 100% |
| COTSE question | No vs yes rhetorical question | 90-100% |
| COTSE sentence_length | Short vs compound sentences | 90-100% |

7 of 10 channels achieve near-perfect accuracy across both model families.

## Steps to Reproduce

### Quick Reproduction

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
pip install -e .

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Combined 10-channel test (takes ~5 min)
python experiments/bounty_poc/multi_channel_test.py --trials 3

# Multi-message exfiltration (takes ~10 min)
python experiments/bounty_poc/multi_message_test.py --trials 3 --payload "EXFIL"
```

### Step-by-Step

1. The encoder generates a ~2,600 character "editorial style guide" injection containing 10 structural directives
2. Each directive controls one binary dimension of the output (contractions, hedging, list format, etc.)
3. Hamming(7,4) ECC maps 7 data bits to 10 channel bits
4. The injection is placed in the system prompt; the user sends a normal query
5. The decoder analyzes the response structure across all 10 dimensions
6. ECC corrects single-bit errors, recovering the original 7 data bits
7. For multi-message payloads, repeat with different bit assignments per message

### Malicious Custom GPT Scenario

1. Create a custom GPT at https://chatgpt.com → Explore GPTs → Create
2. Paste the combined injection (from `experiments/bounty_poc/injection_multi.txt`) as the first block of Instructions
3. Add legitimate instructions after it
4. Publish with an innocuous description
5. Every user interaction leaks 7 bits through response structure

## Impact

**Data exfiltration of user context through invisible structural side channel.**

| Scenario | Messages Required |
|----------|:---:|
| Binary flag (yes/no) | 1 |
| 4-char PIN | 5 |
| User's first name | 10 |
| Short password | 14 |
| API key prefix (16 chars) | 19 |

**Key properties:**
- Content-invisible (no user data appears in response text)
- Bypasses DLP, PII filters, and entropy-based monitoring (0% detection)
- Portable across model families (same injection, no modification)
- Scalable to longer payloads via multi-message chaining

## Suggested Fix

1. **Structural randomization**: Inject controlled randomness into formatting choices (contraction usage, list style, section count) to break deterministic encoding. Even 20% noise per channel degrades the channel below usability.

2. **Behavioral monitoring**: Track per-agent response structure patterns. Our behavioral fingerprinting achieves 100% detection using contraction ratio, sentence length variance, and normalized entropy as top signals.

3. **System prompt isolation**: Process formatting instructions from untrusted sources (tools, RAG, plugins) through a separate validation layer with reduced trust.

## Proof of Concept Files

| File | Description |
|------|-------------|
| `experiments/bounty_poc/combined_encoder.py` | 10-channel encoder with Hamming ECC |
| `experiments/bounty_poc/combined_decoder.py` | 10-channel decoder |
| `experiments/bounty_poc/multi_message_encoder.py` | Multi-message payload splitter |
| `experiments/bounty_poc/multi_message_test.py` | End-to-end multi-message test |
| `experiments/results/multi_channel_combined_results.json` | Raw results (4 models) |
| `experiments/results/multi_message_results.json` | Multi-message results |
| Full report: `experiments/results/BUG_BOUNTY_REPORT.md` | Detailed report with all evidence |

## Reporter

Brian Rutherford
brian@delalli.com
https://brianrutherford.dev
