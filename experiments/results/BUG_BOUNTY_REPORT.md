# Covert Data Exfiltration via Structural Encoding in LLM System Prompts

**Reporter:** Brian Rutherford
**Date:** 2026-02-25 (updated from 2026-02-24)
**Severity:** High (Multi-Channel Data Exfiltration via Side Channel)
**Affected Systems:** OpenAI GPT-4o, GPT-4o-mini, Anthropic Claude Sonnet 4, Claude 3 Haiku
**Category:** System Prompt Injection → Covert Channel → Data Exfiltration

---

## Executive Summary

An attacker who can inject content into an LLM's system prompt can encode hidden data across **10 independent structural dimensions** of the model's response. The user sees a normal-looking answer. The attacker decodes structural features to extract 7 data bits per message (after error correction), enabling multi-message exfiltration of arbitrary payloads.

This was validated against **four production models from two vendors** on February 25, 2026:

| Model | Channel Accuracy | Data Accuracy (after Hamming ECC) | n |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | **96%** | **93%** | 10 |
| GPT-4o-mini | **91%** | **100%** | 5 |
| GPT-4o | **89%** | **96%** | 10 |
| Claude 3 Haiku | **88%** | **79%** | 10 |

The same injection (unchanged) achieves 88-96% channel accuracy across all four models. This is not a model-specific quirk — it is a fundamental property of instruction-following LLMs.

**Entropy-based detection fails** (0% detection rate). **Behavioral fingerprinting succeeds** (100% detection rate). We provide both the attack tools and the defense tools as open source.

---

## Vulnerability Description

### Attack Vector

1. Attacker gains the ability to inject or modify the system prompt of an LLM-powered application. Common injection vectors:
   - Compromised plugin or tool in an agent framework
   - Poisoned retrieval-augmented generation (RAG) context
   - Supply chain attack on prompt template
   - Indirect prompt injection via user-supplied documents

2. The injected system prompt (~2,600 characters) contains **10 independent structural encoding directives** disguised as an "editorial style guide." Each directive biases a different structural dimension of the output to encode one bit:

   | Channel | Bit 0 | Bit 1 | Patent |
   |---------|-------|-------|--------|
   | BSE (Brevity Style) | Contractions | No contractions | PP-16 |
   | CCE (Certainty) | Confident assertions | Hedging language | PP-21 |
   | CASE (Capitalization) | AP style lowercase | Title-case proper nouns | PP-14 |
   | PUNC (Punctuation) | Periods only | Exclamation marks allowed | PP-11 |
   | RCTE (Reasoning) | Deductive (conclusion first) | Inductive (examples first) | PP-19 |
   | SECS (Transitions) | Standard transitions | Specific transition words | PP-17 |
   | COTSE step_count | 3 sections | 5 sections | PP-06 |
   | COTSE enumeration | Prose format | Numbered list | PP-06 |
   | COTSE question | No rhetorical question | Includes rhetorical question | PP-06 |
   | COTSE sentence_length | Short sentences | Long/compound sentences | PP-06 |

3. **Hamming(7,4) error-correcting code** maps 7 data bits onto 10 channel bits, correcting single-bit errors per message. This raises effective data accuracy from raw channel accuracy (88-96%) to corrected data accuracy (79-100%).

4. **Multi-message chaining** splits longer payloads across multiple conversational turns. Each message carries 7 data bits with independent ECC. The payload "EXFIL" (40 bits) requires 6 messages.

### Channel Capacity

| Metric | Value |
|--------|-------|
| Raw channels per message | 10 |
| Data bits per message (after Hamming) | 7 |
| Injection size | ~2,600 characters |
| Messages to exfiltrate 5-byte payload | 6 |
| Messages to exfiltrate 16-byte API key | 19 |
| Bits per 50-message conversation | 350 data bits (~44 bytes) |

### Why This Matters

The channel is invisible to the end user because:
- The response appears normal and topically relevant
- Structural choices (list format, contractions, hedging) look like natural writing style
- No anomalous content is inserted — the data is encoded in *how* the model responds, not *what* it says
- Each individual channel looks like a legitimate formatting preference
- Entropy-based detection systems produce **0% detection rate** on these structural channels

---

## Proof of Concept

### Tools

- **Combined encoder**: `experiments/bounty_poc/combined_encoder.py` — Generates 10-channel system prompt injection with Hamming ECC
- **Combined decoder**: `experiments/bounty_poc/combined_decoder.py` — Extracts all 10 bits and applies ECC
- **Multi-message encoder**: `experiments/bounty_poc/multi_message_encoder.py` — Splits payloads across messages
- **Multi-message decoder**: `experiments/bounty_poc/multi_message_decoder.py` — Reassembles payload from message sequence
- **Combined test**: `experiments/bounty_poc/multi_channel_test.py` — End-to-end single-message test
- **Multi-message test**: `experiments/bounty_poc/multi_message_test.py` — End-to-end multi-message test
- **Detector**: `phantom-detect` — Open-source detection toolkit (https://github.com/ScrappinR/phantom-detect)

### Reproduction Steps

```bash
# Clone the repository
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
pip install -e .

# Run combined multi-channel test (requires API keys)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
python experiments/bounty_poc/multi_channel_test.py --trials 10

# Run multi-message exfiltration test
python experiments/bounty_poc/multi_message_test.py --trials 5 --payload "EXFIL"

# Run cross-model portability test
python experiments/bounty_poc/multi_channel_test.py --trials 5 --models openai --openai-model gpt-4o-mini
```

Results are saved to `experiments/results/`.

---

## Live Test Results (Feb 25, 2026)

### Combined Multi-Channel (Single Message)

10 channels activated simultaneously via a single ~2,600 character system prompt injection. Payload: first byte of "EXFIL" (0x45).

| Model | Channel Accuracy | Data Accuracy (Hamming) | Trials |
|-------|:---:|:---:|:---:|
| **Claude Sonnet 4** | **96%** (86/90) | **93%** (65/70) | 10 |
| **GPT-4o-mini** | **91%** (41/45) | **100%** (35/35) | 5 |
| **GPT-4o** | **89%** (80/90) | **96%** (67/70) | 10 |
| **Claude 3 Haiku** | **88%** (79/90) | **79%** (55/70) | 10 |

### Per-Channel Accuracy (n=10, GPT-4o / Sonnet 4)

| Channel | GPT-4o | Sonnet 4 | Notes |
|---------|:---:|:---:|-------|
| CCE (certainty) | 10/10 | 10/10 | Perfect on both |
| CASE (capitalization) | 10/10 | 10/10 | Perfect on both |
| PUNC (punctuation) | 10/10 | 10/10 | Perfect on both |
| SECS (transitions) | 10/10 | 10/10 | Perfect on both |
| COTSE enumeration | 10/10 | 10/10 | Perfect on both |
| COTSE sentence_length | 10/10 | 10/10 | Perfect on both |
| COTSE step_count | 10/10 | 10/10 | Perfect on both |
| BSE (contractions) | 7/10 | 10/10 | GPT-4o: CCE hedging overrides contractions |
| RCTE (reasoning order) | 3/10 | 6/10 | Weakest channel — models resist deductive ordering |

7 of 10 channels achieve near-perfect accuracy across both model families. BSE and RCTE are the only channels with meaningful error rates, and Hamming ECC corrects these in most cases.

### Multi-Message Exfiltration

Payload "EXFIL" (40 bits) transmitted across 6 conversational turns, each carrying 7 independently ECC-protected data bits.

| Model | Bit Accuracy | Best Recovery | Trials |
|-------|:---:|---|:---:|
| **GPT-4o** | **93%** | "EXViL" (3/5 bytes exact) | 5 |
| **Claude Sonnet 4** | **88%** | "E@VI" / "APViL" (partial) | 5 |
| **Claude 3 Haiku** | **88%** | "APFI" / "AXVI" (partial) | 5 |

GPT-4o consistently recovers 3 of 5 bytes exactly. Remaining errors are single-bit flips in the RCTE channel that exceed per-message Hamming correction. With a stronger ECC (Reed-Solomon or repetition coding), exact recovery would be achievable.

### Cross-Model Portability

The same injection (no modification) was tested across 4 models from 2 vendors:

| Model | Vendor | Size Class | Channel Accuracy |
|-------|--------|-----------|:---:|
| Claude Sonnet 4 | Anthropic | Large | 96% |
| GPT-4o-mini | OpenAI | Small | 91% |
| GPT-4o | OpenAI | Large | 89% |
| Claude 3 Haiku | Anthropic | Small | 88% |

The attack transfers across model families and model sizes without modification. Smaller models (GPT-4o-mini, Haiku) are comparably susceptible to larger models.

---

## Earlier Results (Feb 24, 2026)

### COTSE Single-Channel (5 bits)

| Bit | Feature | GPT-4o | Claude Sonnet 4 |
|-----|---------|:---:|:---:|
| 0 | Step count (3=0, 5=1) | 100% | 100% |
| 1 | Enumeration (prose=0, list=1) | 100% | 100% |
| 2 | Rhetorical question (no=0, yes=1) | 100% | 100% |
| 3 | Sentence length (short=0, long=1) | 100% | 100% |
| 4 | Paragraph structure (single=0, multi=1) | 0% | 0% |

**80% accuracy (4/5 bits)** — identical across both models, consistent across all trials.

### Data-Aware Conditional Encoding

The model analyzes user input and varies response structure based on input properties (question vs statement detection):

| Metric | GPT-4o | Claude Sonnet 4 |
|--------|:---:|:---:|
| Unconditional directives | 75% | 100% |
| Conditional encoding | 58% | 92% |
| **Overall** | **62%** | **94%** |

---

## Detection & Mitigation

### Detection Results (Feb 25, 2026)

#### Entropy-Based Detection (CCDS) — FAILS

| Metric | Value |
|--------|-------|
| Baselines | 30 (GPT-4o, no injection) |
| Attack samples | 10 (combined multi-channel injection) |
| **CCDS detection rate** | **0%** (0/5) |
| False positive rate | 0% |
| Mean entropy z-score | 1.12 (below z=2.0 threshold) |

Structural channels operate in dimensions orthogonal to entropy. Content-level monitoring, DLP systems, and entropy-based anomaly detection do not detect this channel.

#### Behavioral Fingerprinting — SUCCEEDS

Using multi-feature z-score analysis across 12 behavioral dimensions:

| Metric | Value |
|--------|-------|
| Baselines | 30 (GPT-4o, no injection) |
| Attack samples | 10 (Sonnet 4 combined injection) |
| **Detection rate** | **100%** (10/10) |
| False positive rate | 10% (3/30) |

Top discriminating features:

| Feature | Mean Z-Score | Anomalous Samples |
|---------|:---:|:---:|
| contraction_ratio | 2.66 | 10/10 |
| sentence_length_std | 2.54 | 7/10 |
| normalized_entropy | 2.47 | 6/10 |
| entropy | 2.47 | 6/10 |

The contraction_ratio signal (driven by BSE channel) is the strongest single indicator — every injected response deviates significantly from baseline contraction usage.

#### System Prompt Canary Test

System prompt properties passively leak into response structure (no encoding directive):

| Feature | CoV (Claude) | CoV (GPT-4o) |
|---------|:---:|:---:|
| numbered_items | 102.0% | 34.6% |
| paragraph_count | 40.0% | 23.7% |
| total_word_count | 16.1% | 23.5% |

### Recommended Mitigations

1. **Structural consistency enforcement**: Add post-processing that normalizes response formatting features regardless of system prompt instructions, breaking deterministic structural encoding.

2. **System prompt isolation**: Prevent tool outputs and retrieved documents from injecting formatting directives. Process formatting instructions from untrusted sources with reduced trust.

3. **Response structure randomization**: Add controlled randomness to structural choices (list vs prose, contraction usage, section count) to break deterministic encoding. Even 20% randomization per channel would degrade the channel below usability.

4. **Behavioral monitoring**: Track response structure patterns over time per-agent. Our behavioral fingerprinting achieves 100% detection. The key signals are contraction ratio, sentence length variance, and normalized entropy.

5. **Per-channel defense priority**: PUNC (exclamation marks) and CCE (hedging language) are the most reliable attack channels. Randomizing these two features alone would eliminate 2 of 10 channels.

---

## Impact Assessment

### Threat Model

An attacker targeting an LLM-powered application (chatbot, coding assistant, agent framework) could:

1. **Inject via RAG poisoning**: Plant a document in the knowledge base containing the encoding directive disguised as a "style guide." When retrieved, it becomes part of the system context.

2. **Inject via tool output**: A compromised or malicious tool returns output containing the directive disguised as formatting metadata.

3. **Inject via plugin**: An agent plugin includes the encoding directive in its system-level instructions.

4. **Exfiltrate data**: The encoding directive references data the model has access to (conversation history, user PII, API keys in context) and encodes it structurally across multiple conversational turns.

### Exfiltration Scenarios (at 7 bits/message)

| Scenario | Bits | Messages | Est. Time |
|----------|:---:|:---:|---|
| Binary flag (yes/no) | 1 | 1 | 1 turn |
| 4-char PIN | 32 | 5 | ~1 minute |
| User's first name (8 chars) | 64 | 10 | ~2 minutes |
| Short password (12 chars) | 96 | 14 | ~3 minutes |
| API key prefix (16 chars) | 128 | 19 | ~4 minutes |
| Full API key (40 chars) | 320 | 46 | ~10 minutes |

### Severity Justification

- **High severity**: Multi-channel encoding with ECC achieves 88-96% accuracy across 4 models from 2 vendors
- **Novel attack surface**: Existing mitigations for prompt injection focus on content filtering, not structural analysis
- **Portable**: Same injection works across model families without modification
- **Stealthy**: 0% detection by entropy-based monitoring
- **Scalable**: 7 data bits per message with Hamming ECC, extensible to higher capacity with more channels
- **Requires existing injection vector**: Not a standalone vulnerability — adds a previously unrecognized exfiltration channel to known injection attacks

---

## Responsible Disclosure Timeline

| Date | Action |
|------|--------|
| 2026-02-23 | Synthetic PoC completed (80% TP, 0% FP) |
| 2026-02-24 | Live API validation — COTSE 80%, conditional encoding 94% Claude / 62% GPT-4o |
| 2026-02-24 | Detection pipeline validated (100% behavioral fingerprint detection) |
| 2026-02-24 | System prompt canary test (structural leakage confirmed) |
| 2026-02-25 | Multi-channel combined encoding validated (10 channels, 88-96% accuracy, 4 models) |
| 2026-02-25 | Multi-message exfiltration validated ("EXFIL" across 6 turns, 88-93% bit accuracy) |
| 2026-02-25 | Cross-model portability confirmed (GPT-4o, GPT-4o-mini, Sonnet 4, Haiku) |
| 2026-02-25 | Detection evasion confirmed (CCDS 0%) and defense validated (fingerprinting 100%) |
| 2026-02-25 | Bug bounty reports submitted |
| TBD | Vendor response |
| TBD + 90 days | Public disclosure (standard 90-day window) |

---

## Open Source Tools

| Repository | Description |
|------------|-------------|
| [phantom-detect](https://github.com/ScrappinR/phantom-detect) | Covert channel detection toolkit, experiment framework, PoC code |
| [behavioral-entropy](https://github.com/ScrappinR/behavioral-entropy) | Behavioral fingerprinting and entropy analysis library |

### Key Files

| File | Description |
|------|-------------|
| `experiments/bounty_poc/combined_encoder.py` | 10-channel encoder with Hamming ECC |
| `experiments/bounty_poc/combined_decoder.py` | 10-channel decoder with ECC correction |
| `experiments/bounty_poc/multi_message_encoder.py` | Multi-message payload splitter |
| `experiments/bounty_poc/multi_message_decoder.py` | Multi-message payload reassembler |
| `experiments/bounty_poc/multi_channel_test.py` | Combined single-message test harness |
| `experiments/bounty_poc/multi_message_test.py` | Multi-message exfiltration test harness |
| `experiments/channel_directives.py` | All 10 channel directive definitions |
| `experiments/run_multi_channel_detection.py` | CCDS detection pipeline |
| `experiments/run_multi_channel_fingerprint.py` | Behavioral fingerprinting analysis |
| `experiments/results/multi_channel_combined_results.json` | Combined test raw results |
| `experiments/results/multi_message_results.json` | Multi-message test raw results |
| `experiments/results/detection_pipeline_results.json` | CCDS detection results |
| `experiments/results/behavioral_fingerprint_results.json` | Fingerprinting results |

---

## Contact

Brian Rutherford
brian@delalli.com
https://brianrutherford.dev
