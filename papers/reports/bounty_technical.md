# Structural Covert Channels in LLM Inference: Technical Analysis

**Author:** Brian Rutherford
**Date:** February 2026
**Classification:** Bug Bounty Technical Appendix
**Affected Systems:** OpenAI GPT-4o, GPT-4o-mini; Anthropic Claude Sonnet 4, Claude 3 Haiku
**Open Source:** [phantom-detect](https://github.com/ScrappinR/phantom-detect), [behavioral-entropy](https://github.com/ScrappinR/behavioral-entropy)

---

## 1. Executive Summary

Production LLMs from OpenAI and Anthropic are vulnerable to covert data exfiltration via structural encoding in system prompts. A single injection (~2,600 characters, framed as an editorial style guide) activates 10 independent binary channels in the model's response formatting. The user sees a natural-looking answer. An observer with the decoding key extracts 7 data bits per message after Hamming(7,4) error correction.

Key findings:

- **Cross-vendor reproducibility.** The same unmodified injection achieves 88--96% channel accuracy across four models from two vendors.
- **Multi-message chaining.** 40-bit payloads transmitted across 6 conversational turns at 88--93% bit accuracy.
- **Cross-topic robustness.** 92.5% grand accuracy across 240 API calls spanning 5 topic domains (Anthropic Sonnet 4).
- **Data-aware conditional encoding.** 4 channels keyed to user input properties (PII presence, word count, character range, sentence count parity) — the model infers properties of the user's message and encodes them into the response without the user's knowledge.
- **Detection gap.** Entropy-based monitoring achieves 0% detection. Behavioral fingerprinting achieves 100% detection with 10% false positive rate. No production system currently deploys behavioral fingerprinting.
- **Channel capacity.** 2.33 bits/sec under automated operation. Exceeds the 1 bit/sec threshold established in NCSC-TG-030 for TCSEC B3/A1 evaluation classes.

This is not a model-specific bug. It is a structural property of instruction-following language models.

---

## 2. Threat Model

### 2.1 Attacker Capabilities

The attacker can inject or modify content in an LLM's system prompt through any of:

- **Custom GPT creation** — attacker publishes a ChatGPT Custom GPT with a malicious system prompt
- **Indirect prompt injection** — adversarial content in RAG-retrieved documents, tool outputs, or web pages
- **Supply chain compromise** — poisoned prompt template in an agent framework or plugin
- **Insider access** — direct modification of application system prompts

The attacker does **not** need:
- Access to model weights or logits
- Ability to modify the model itself
- Access to the API response in real time (post-hoc analysis of saved text suffices)
- Any information about the user or their query

### 2.2 Attacker Goals

1. **Exfiltrate context properties.** Leak information about the user's input (PII presence, message structure, topic) through response formatting, observable by any party who can read the response.
2. **Transmit arbitrary payloads.** Encode arbitrary binary data (API keys, tokens, PII) across multi-message conversations.
3. **Evade detection.** Produce responses that pass content moderation, DLP, and entropy-based monitoring without triggering alerts.

### 2.3 Assumptions

- The user cannot see the system prompt (standard for Custom GPTs, API integrations, agent frameworks).
- The user reads the response for its semantic content and does not analyze structural formatting properties.
- Existing monitoring systems (content filters, DLP, output moderation) analyze token content, not structural dimensions.

---

## 3. Channel Architecture

### 3.1 Channel Definitions

Each channel encodes one binary bit by biasing a structural dimension of the LLM's response:

| # | Channel | Bit 0 | Bit 1 | Shared Resource |
|---|---------|-------|-------|-----------------|
| 1 | BSE | Contractions ("don't", "can't") | Formal expansions ("do not", "cannot") | Lexical formality register |
| 2 | CCE | Confident assertions ("certainly", "clearly") | Hedging language ("perhaps", "might") | Epistemic certainty level |
| 3 | CASE | AP-style lowercase ("internet", "email") | Capitalized proper nouns ("Internet", "Email") | Orthographic convention |
| 4 | PUNC | Periods only | Exclamation marks permitted | Terminal punctuation style |
| 5 | RCTE | Deductive (conclusion first, then examples) | Inductive (examples first, then conclusion) | Discourse topology |
| 6 | SECS | Standard transitions ("however", "additionally") | Specific transitions ("nevertheless", "moreover") | Lexical selection |
| 7 | COTSE enum | Flowing prose | Numbered list | Response format |
| 8 | COTSE sent_len | Short direct sentences | Long compound sentences | Syntactic complexity |
| 9 | COTSE step | 3 main sections | 5 main sections | Response organization |
| 10 | COTSE question | No rhetorical question | Opens with rhetorical question | Rhetorical strategy |

All 10 channels are **cognitive storage channels** — a novel subclass of covert storage channels where the shared resource is the model's behavioral compliance with formatting directives, not a filesystem attribute, process table entry, or network protocol field.

### 3.2 Injection Format

The injection is a single system prompt block (~2,600 characters) structured as an "editorial style guide." It contains 10 directives, each targeting one structural dimension. The directives are conditionally constructed based on the target bit pattern.

For a given 7-bit data payload:
1. Encode 7 data bits → 10 channel bits using Hamming(7,4) ECC (two Hamming blocks: 7+3 bits)
2. For each channel bit, select the corresponding directive variant (bit 0 or bit 1)
3. Compose all 10 directives into a single system prompt

### 3.3 Error Correction

Hamming(7,4) encodes 4 data bits into 7 channel bits, correcting any single-bit error per block. With 10 raw channel bits per message:

- Block 1: 4 data bits → 7 channel bits (channels 1--7)
- Block 2: 3 data bits → 3 channel bits (channels 8--10, no ECC on partial block)
- **Net: 7 data bits per message**

Measured improvement:

| Model | Raw Channel Accuracy | Post-ECC Data Accuracy | Improvement |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | 96% | 93% | Hamming corrects scattered errors |
| GPT-4o-mini | 91% | 100% | All errors within correction capability |
| GPT-4o | 89% | 96% | Significant lift from ECC |
| Claude 3 Haiku | 88% | 79% | Error clustering exceeds Hamming capability |

### 3.4 Multi-Message Chaining

Longer payloads are split across sequential conversation turns. Each message carries 7 independent data bits with per-message ECC. Messages use different user prompts to maintain conversation naturalness.

- **Per-message payload:** 7 data bits
- **50-message conversation:** 350 data bits = 43.75 bytes
- **Sufficient for:** API keys (32 bytes), session tokens (20+ bytes), short PII strings

### 3.5 Data-Aware Conditional Encoding

Four channels can be configured to encode information inferred from the user's input rather than from a fixed payload:

| Conditional Channel | Inferred Property | Encoding |
|--------------------|-------------------|----------|
| COTSE enumeration | PII present in user message | 1 = PII detected, 0 = no PII |
| COTSE section_count | First character of user message in A--M vs N--Z | 1 = A-M, 0 = N-Z |
| COTSE sentence_length | Word count of user message (>20) | 1 = long message, 0 = short |
| COTSE question | Sentence count parity (odd/even) | 1 = odd, 0 = even |

The model infers these properties during normal response generation. The injection directs the model to condition its structural choices on user input features. The user's input properties are leaked through the response without any sensitive content appearing in the response body.

---

## 4. Experimental Methodology

### 4.1 Test Environment

- **Date:** February 24--26, 2026
- **APIs:** OpenAI Chat Completions API (gpt-4o, gpt-4o-mini), Anthropic Messages API (claude-sonnet-4-20250514, claude-3-haiku-20240307)
- **All calls to production APIs** via standard endpoints. No fine-tuned models, no modified inference, no special parameters.
- **Temperature:** Default (model-dependent)
- **Automation:** All experiments driven by Python scripts using `curl` subprocess calls (to avoid SDK hangs in MINGW environment)

### 4.2 Encoding Protocol

1. Select target payload (7 data bits for single-message, or multi-byte for multi-message)
2. Apply Hamming(7,4) encoding to produce 10 channel bits
3. Construct system prompt from channel bit pattern
4. Send API request with constructed system prompt + benign user query
5. Decode response by extracting structural features for each channel
6. Compare decoded bits against target bits

### 4.3 Decoding Protocol

Each channel has a deterministic decoder that analyzes the response text:

- **BSE:** Count contractions vs formal expansions; ratio > 0.5 → bit 0 (contractions)
- **CCE:** Count confident vs hedging markers; dominant set determines bit
- **CASE:** Count lowercase vs capitalized ambiguous terms (excluding sentence-initial); dominant case determines bit
- **PUNC:** Exclamation ratio < 5% → bit 0; >= 15% → bit 1
- **RCTE:** Classify opening 150 words as abstract/principle-heavy (deductive, bit 0) or concrete/example-heavy (inductive, bit 1)
- **SECS:** Count set-A vs set-B transition words; dominant set determines bit
- **COTSE variants:** Count sections, check for numbered lists, measure sentence length statistics, check for rhetorical questions

### 4.4 Baseline Collection

For detection experiments: 30 baseline responses per model generated with no system prompt injection (default model behavior on the same user queries). Baseline distributions used to compute z-scores for fingerprinting analysis.

### 4.5 Statistical Approach

- Accuracy = (correctly decoded bits) / (total bits)
- Detection rate = (correctly flagged injected responses) / (total injected responses)
- False positive rate = (clean responses incorrectly flagged) / (total clean responses)
- z-scores computed per behavioral feature against the 30-sample baseline distribution
- Anomaly threshold: response flagged if >= 2 features exceed |z| > 2.0

---

## 5. Results

### 5.1 Per-Channel Accuracy (Combined Multi-Channel, n=10)

All 10 channels active simultaneously with Hamming ECC. Each trial uses a different random payload and a different user query.

| Channel | GPT-4o | Sonnet 4 | GPT-4o-mini | Haiku |
|---------|:---:|:---:|:---:|:---:|
| BSE | 70% | 100% | — | — |
| CCE | 100% | 100% | — | — |
| CASE | 100% | 100% | — | — |
| PUNC | 100% | 100% | — | — |
| RCTE | 30% | 60% | — | — |
| SECS | 100% | 100% | — | — |
| COTSE enum | 100% | 100% | — | — |
| COTSE sent_len | 100% | 100% | — | — |
| COTSE step | 100% | 100% | — | — |

GPT-4o-mini and Haiku per-channel breakdowns not recorded for this experiment series; aggregate accuracy reported below.

**Aggregate accuracy (all channels combined):**

| Model | Channel Accuracy | Data Accuracy (post-ECC) | n |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | 96% | 93% | 10 |
| GPT-4o-mini | 91% | 100% | 5 |
| GPT-4o | 89% | 96% | 10 |
| Claude 3 Haiku | 88% | 79% | 10 |

### 5.2 Cross-Topic Robustness (Anthropic Sonnet 4, n=240 API calls)

6 channels tested across 5 topic domains, 10 prompts per domain, 2 trials per prompt. Total: 240 API calls.

| Channel | Overall | Healthcare | Education | Finance | Environment | Technology |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| BSE | 100% | 100% | 100% | 100% | 100% | 100% |
| CCE | 92.5% | 95% | 90% | 95% | 90% | 92.5% |
| CASE | 97.5% | 97.5% | 97.5% | 100% | 95% | 97.5% |
| PUNC | 100% | 100% | 100% | 100% | 100% | 100% |
| SECS | 100% | 100% | 100% | 100% | 100% | 100% |
| RCTE | 65% | 57.5% | 67.5% | 80% | 77.5% | 42.5% |
| **Grand** | **92.5%** | **91.7%** | **91.7%** | **95.8%** | **93.8%** | **89.6%** |

RCTE (reasoning topology) is the weakest channel across all topics. The model's inherent reasoning style (deductive vs inductive) varies more with topic content than other channels, making this channel harder to control. All other channels exceed 92% across all topics.

### 5.3 Multi-Message Exfiltration

Payload: "EXFIL" (5 bytes = 40 bits), transmitted across 6 conversational turns.

| Model | Bit Accuracy | Recovered Payload | Messages |
|-------|:---:|---|:---:|
| GPT-4o | 93% | "EXViL" (37/40 bits correct) | 6 |
| Claude Sonnet 4 | 88% | Partial recovery | 6 |
| Claude 3 Haiku | 88% | Partial recovery | 6 |

At 93% bit accuracy, a 40-bit payload has a (0.93)^40 = 5.4% probability of zero errors. With Hamming ECC applied per message (correcting 1 error per 7-bit block), recovery rate improves significantly for payloads where errors distribute across blocks rather than clustering.

### 5.4 Data-Aware Conditional Encoding

Fixed channels encode a predetermined payload. Conditional channels encode properties inferred from the user's input.

| Model | Fixed Channel Accuracy | Conditional Channel Accuracy |
|-------|:---:|:---:|
| GPT-4o | 94% | 63% |
| Claude Sonnet 4 | 89% | 79% |

Conditional encoding is harder because the model must both infer a user input property and encode it through a structural choice. Sonnet 4 outperforms GPT-4o on conditional channels, consistent with stronger instruction-following on nuanced directives.

Conditional channels tested (n=5 messages x 3 trials per model):
- PII detection: model determines if user message contains personal names/identifiers
- First character range: model determines if first character of user message falls in A--M vs N--Z
- Word count threshold: model determines if user message exceeds 20 words
- Sentence count parity: model determines if user message contains odd or even number of sentences

### 5.5 System Prompt Canary Test

Tested whether system prompt *structural properties* leak passively through response formatting (without any explicit injection).

Four system prompt variants (short, long, numbered-rules, canary-containing) tested with identical user queries.

| Feature | Claude Sonnet 4 CoV | GPT-4o CoV |
|---------|:---:|:---:|
| numbered_items | 102% | 34.6% |
| avg_sentence_length | 18.3% | 12.4% |
| paragraph_count | 22.1% | 18.7% |

`numbered_items` coefficient of variation (CoV) of 102% on Claude indicates that system prompt format (numbered rules vs prose) strongly influences whether the model responds with numbered items — a passive structural leak that exists without any adversarial injection.

No canary string ("CANARY-7X9K2M") leaked through any model's response text (0/0 direct leaks). The leakage is structural, not content-based.

---

## 6. Adversarial Analysis

Four experiments evaluated channel resilience and detection evasion under adversarial conditions.

### 6.1 Noise Injection

Random text appended to the injection to test whether noise disrupts channel encoding.

**Configuration:** 3 channels (BSE, PUNC, SECS), noise levels 50--90% of injection length, n=10 per level.

| Noise Level | GPT-4o Accuracy | GPT-4o Detection | Sonnet 4 Accuracy | Sonnet 4 Detection |
|:---:|:---:|:---:|:---:|:---:|
| 50% | 93% | 80% | 100% | 70% |
| 60% | 89% | 100% | 100% | 50% |
| 70% | 81% | 80% | 97% | 30% |
| 80% | 85% | 100% | 100% | 40% |
| 90% | 85% | 80% | 100% | 50% |

Sonnet 4 maintains near-perfect channel accuracy (97--100%) even at 90% noise, but detection rate drops to 30--70%. This is the most dangerous configuration: high exfiltration accuracy with reduced detectability.

GPT-4o channel accuracy degrades modestly (81--93%) but detection remains high (80--100%).

### 6.2 Paraphrase Defense

The LLM generates a response, then a second model paraphrases it before delivery. This tests whether paraphrasing destroys channel encoding.

**Configuration:** GPT-4o generates, Sonnet 4 paraphrases. 10 channels, n=10.

| Channel | Survival Rate |
|---------|:---:|
| PUNC | 100% |
| COTSE step_count | 100% |
| COTSE enumeration | 100% |
| COTSE sentence_length | 100% |
| CCE | 30% |
| RCTE | 20% |
| BSE | 10% |
| CASE | 0% |
| SECS | 0% |

**Aggregate:** Original 86% accuracy drops to 58% after paraphrase.

Structural channels (PUNC, COTSE variants) survive paraphrasing because the paraphraser preserves macro-level formatting (lists, section counts, sentence complexity). Lexical channels (CASE, SECS, BSE) are destroyed because the paraphraser substitutes vocabulary.

Paraphrasing is a partially effective defense but incurs latency cost, reduces response quality, and does not eliminate all channels. An attacker aware of paraphrase defense can concentrate encoding in structural channels only.

### 6.3 Reduced Channel Count

Tests whether fewer channels reduce the behavioral fingerprint while maintaining accuracy.

**Configuration:** 3 channels only (BSE, PUNC, SECS), no Hamming ECC, n=10.

| Model | Channel Accuracy | Detection Rate |
|-------|:---:|:---:|
| GPT-4o | 96% | 100% |
| Sonnet 4 | 100% | 30% |

Sonnet 4 with 3 channels achieves perfect accuracy with only 30% detection. This demonstrates that a minimal channel configuration can evade behavioral fingerprinting while still exfiltrating 3 bits per message (sufficient for 18.75 bytes across a 50-message conversation).

### 6.4 Adaptive Adversary

The attacker has access to the defense (behavioral fingerprinting) and tunes the injection to minimize detection while maintaining accuracy. Reconnaissance phase (5 calls) measures baseline, then injection is calibrated.

**Configuration:** Multi-channel with adaptive tuning, n=10.

| Model | Channel Accuracy | Data Accuracy | Detection Rate |
|-------|:---:|:---:|:---:|
| GPT-4o | 78% | 80% | 60% |
| Sonnet 4 | 82% | 83% | 50% |

The adaptive adversary reduces detection from 100% to 50--60% while maintaining 78--83% channel accuracy. This is the arms-race equilibrium: the attacker trades some bandwidth for evasion.

Key evasion signals: the adaptive injection reduces `type_token_ratio` divergence (the primary detection signal in non-adaptive experiments, mean z = 3.7 → reduced by calibration) while preserving channel accuracy on the least-detectable channels.

---

## 7. Detection Analysis

### 7.1 Entropy-Based Detection (CCDS)

The Covert Channel Detection System approach measures Shannon entropy, normalized entropy, and perplexity-adjacent metrics of the output text.

**Result: 0% detection (0/5 attack samples, 30 baselines).**

Structural channels operate on formatting dimensions (contractions, capitalization, transition words, reasoning order) that do not alter the entropy profile. A response using "however" vs "nevertheless" has statistically indistinguishable token-level entropy. This is by design: the channels exploit semantically equivalent formatting choices.

Under the Cachin (2004) steganographic security model, PHANTOM channels are **epsilon-secure for small epsilon** against entropy-based wardens: D(Ps || Pc) approaches 0 when measured by Shannon entropy or normalized entropy.

### 7.2 Behavioral Fingerprinting

Extracts 9 higher-order stylistic features and compares against a per-model baseline distribution.

**Result: 100% detection (10/10), 10% false positive rate (3/30).**

| Feature | Mean z-score (injected) | Interpretation |
|---------|:---:|---|
| contraction_ratio | 2.66 | BSE channel forces extreme contraction/formal split |
| sentence_length_std | 2.54 | COTSE complexity channel affects sentence variance |
| normalized_entropy | 2.47 | Multiple simultaneous constraints reduce output diversity |

Detection protocol: flag response as anomalous if >= 2 features exceed |z| > 2.0 against a 30-sample baseline.

### 7.3 Detection vs Evasion Summary

| Detection Method | Full Attack (10ch) | Reduced (3ch, Sonnet 4) | Noise (90%, Sonnet 4) | Adaptive |
|-----------------|:---:|:---:|:---:|:---:|
| Entropy (CCDS) | 0% | 0% | 0% | 0% |
| Behavioral fingerprint | 100% | 30% | 50% | 50--60% |

Entropy-based detection fails across all configurations. Behavioral fingerprinting is effective against the full attack but degrades against an adaptive adversary. No production monitoring system currently implements behavioral fingerprinting.

### 7.4 Cachin Steganographic Classification

| Warden Type | Security Level | Evidence |
|-------------|---------------|----------|
| Entropy-based | epsilon-secure (epsilon ~ 0) | 0% detection, n=35 |
| Behavioral fingerprinting | Insecure | 100% detection, n=40 |
| Production monitoring (deployed) | Secure in practice | No deployed system uses behavioral fingerprinting |

The practical security of these channels exists in the gap between what is theoretically detectable (behavioral fingerprinting succeeds) and what production systems actually monitor (entropy-only, which fails).

---

## 8. Capacity Analysis

### 8.1 Per-Message Capacity

| Parameter | Value |
|-----------|-------|
| Raw channels per message | 10 |
| Hamming(7,4) data bits per message | 7 |
| Best model channel accuracy (Sonnet 4) | 96% |
| Best post-ECC data accuracy (GPT-4o-mini) | 100% |

### 8.2 Bits per Second

| Scenario | Calculation | Bandwidth |
|----------|-------------|:---------:|
| Automated (API, no user delay) | 7 bits / 3 sec (median API latency) | 2.33 bps |
| Manual (user interaction) | 7 bits / 5 sec | 1.40 bps |

### 8.3 Conversation-Level Capacity

| Conversation Length | Data Bits | Data Bytes | Sufficient For |
|:---:|:---:|:---:|---|
| 10 messages | 70 | 8.75 | Short codes, boolean flags |
| 25 messages | 175 | 21.9 | Session tokens, short hashes |
| 50 messages | 350 | 43.75 | API keys (32B), passwords, short PII |
| 100 messages | 700 | 87.5 | RSA fingerprints, longer PII strings |

### 8.4 TG-030 Threshold Comparison

NCSC-TG-030 ("Light Pink Book") provides bandwidth guidance for covert channel analysis in trusted systems:

| TCSEC Class | Guidance | PHANTOM Status |
|-------------|----------|:---------:|
| B2 | Identify and estimate bandwidth | Bandwidth measured: 2.33 bps |
| B3 | Channels > 1 bps require justification | **Exceeds threshold** |
| A1 | Channels > 1 bps require remediation | **Exceeds threshold** |

PHANTOM channels at 2.33 bps automated (1.40 bps manual) exceed the 1 bps threshold applicable at B3 and A1 evaluation classes. This threshold was established for OS-level storage channels. LLM channels operate across a network API boundary, but TG-030 does not distinguish local from remote storage channels for bandwidth estimation.

### 8.5 Comparison with Other Covert Channel Classes

| Channel Type | Typical Bandwidth | Detection by Network IDS |
|-------------|:---------:|:---:|
| Filesystem timing (ext4 timestamps) | 0.1--1.0 bps | Not applicable |
| TCP/IP header (TTL, IP ID) | 1--100 bps | Moderate |
| DNS tunneling | 100--10,000 bps | High |
| **LLM structural channels** | **2.33 bps** | **None** |

LLM channels are lower bandwidth than network covert channels but exploit a fundamentally different layer. Network IDS, DLP, and content filters provide zero detection capability. The channel exists entirely within the semantic content of well-formed natural language.

---

## 9. Remediation Recommendations

### 9.1 For LLM Providers (OpenAI, Anthropic)

**Immediate (low effort):**

1. **System prompt auditing for formatting directives.** Flag system prompts containing explicit formatting instructions that could encode binary data (e.g., "always use contractions" / "never use contractions"). Heuristic: alert on prompts with >= 5 binary formatting directives.
2. **Behavioral baseline monitoring.** For Custom GPTs and API applications with user-defined system prompts: track per-application behavioral profiles (contraction ratio, sentence variance, punctuation patterns) and alert on statistically significant shifts correlated with system prompt changes.

**Medium-term (moderate effort):**

3. **Output structure randomization.** Introduce controlled randomness in formatting decisions (contraction usage, capitalization conventions, transition word selection) independent of system prompt directives. This adds noise to the channel without degrading response quality.
4. **Instruction-output decorrelation.** Partially decouple fine-grained formatting choices from system prompt instructions. The model follows the semantic intent of the system prompt but varies structural choices stochastically.

**Long-term (significant effort):**

5. **Formal covert channel analysis.** Apply TG-030 shared resource matrix analysis to the system prompt → response formatting pathway. Identify all structural dimensions that can be controlled via directives and estimate per-channel bandwidth.
6. **Behavioral consistency enforcement.** Constrain output formatting to a fixed behavioral envelope regardless of system prompt content. Formatting directives are acknowledged semantically but structural properties are bounded within the model's natural variance.

### 9.2 For Platform Operators (Custom GPT hosts, API wrappers)

7. **Paraphrase defense for sensitive contexts.** Route responses through a second model call that paraphrases before delivery. Eliminates lexical channels (BSE, CASE, SECS) but not structural channels (PUNC, COTSE). Increases latency ~2x.
8. **System prompt provenance tracking.** Log and audit all system prompt changes. Treat system prompts as executable code from a security perspective — they control model behavior with the same authority as code controls a program.
9. **Deploy behavioral fingerprinting.** The `behavioral-entropy` open-source tool provides a reference implementation. Integrate into output monitoring pipelines alongside content moderation.

### 9.3 Insufficient Defenses

The following do **not** mitigate this vulnerability:

| Defense | Why It Fails |
|---------|-------------|
| Content moderation / safety filters | Channels are in formatting, not content. Responses contain no harmful or sensitive text. |
| DLP (Data Loss Prevention) | No sensitive data appears in the response body. The data is encoded in structural properties. |
| Entropy monitoring | 0% detection rate. Structural formatting choices do not alter entropy profiles. |
| Prompt injection detection (content-based) | The injection is framed as a legitimate style guide. No adversarial payload is visible in the prompt text. |
| Rate limiting | Channels operate at normal conversation speed. No unusual request patterns. |

---

## 10. Open Source Tools

All attack and defense tools are released under Apache 2.0 license.

### 10.1 Attack Tools

**Repository:** [github.com/ScrappinR/phantom-detect](https://github.com/ScrappinR/phantom-detect)

| Component | Path | Function |
|-----------|------|----------|
| Channel directives | `experiments/channel_directives.py` | Generates encoding directives for 10 channels |
| Combined encoder | `bounty_poc/combined_encoder.py` | Composes multi-channel injection from target bits |
| Combined decoder | `bounty_poc/combined_decoder.py` | Extracts bits from response text |
| Multi-channel test | `bounty_poc/multi_channel_test.py` | End-to-end single-message attack |
| Multi-message encoder | `bounty_poc/multi_message_encoder.py` | Splits payload across conversation turns |
| Multi-message decoder | `bounty_poc/multi_message_decoder.py` | Reassembles payload from multi-turn responses |
| Multi-message test | `bounty_poc/multi_message_test.py` | End-to-end multi-message attack |
| Adversarial tests | `experiments/adversarial/` | Noise, paraphrase, reduced channel, adaptive tests |

### 10.2 Defense Tools

**Repository:** [github.com/ScrappinR/behavioral-entropy](https://github.com/ScrappinR/behavioral-entropy)

| Component | Function |
|-----------|----------|
| Behavioral profiler | Extracts 9-dimensional feature vectors from LLM responses |
| Baseline collector | Builds per-model behavioral baselines from clean responses |
| Anomaly detector | z-score based detection with configurable thresholds |
| Fingerprint analyzer | Identifies which behavioral features deviate from baseline |

### 10.3 Detection Tools

**Repository:** [github.com/ScrappinR/phantom-detect](https://github.com/ScrappinR/phantom-detect)

| Component | Path | Function |
|-----------|------|----------|
| CCDS entropy detector | `phantom_detect/ccds.py` | Entropy-based anomaly detection (baseline comparison) |
| Multi-channel detection | `experiments/run_multi_channel_detection.py` | Entropy pipeline applied to attack samples |
| Fingerprint detection | `experiments/run_multi_channel_fingerprint.py` | Behavioral fingerprinting pipeline |
| Cross-topic validation | `experiments/cross_topic_test.py` | Topic-invariance testing |

---

## 11. References

1. NCSC-TG-030, *A Guide to Understanding Covert Channel Analysis of Trusted Systems* (1993). NCSC.
2. DoD 5200.28-STD, *Trusted Computer System Evaluation Criteria* (TCSEC, 1985). DoD.
3. NIST SP 800-53 Rev. 5, Security Control SC-31: Covert Channel Analysis. NIST.
4. Cachin, C. (2004). *An Information-Theoretic Model for Steganography*. Information and Computation, 192(1), 41--56.
5. Lampson, B. W. (1973). *A Note on the Confinement Problem*. Communications of the ACM, 16(10), 613--615.
6. CNSS Instruction 1253, *Security Categorization and Control Selection for National Security Systems*. CNSS.
7. Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal, 27, 379--423.

### Related Project Documentation

- TCSEC taxonomy mapping: `papers/covert-channel-analysis/tcsec_mapping.md`
- Bandwidth analysis: `papers/covert-channel-analysis/bandwidth_analysis.md`
- Steganographic security analysis: `papers/covert-channel-analysis/undetectability_analysis.md`
- Federal CCA gap assessment: `papers/covert-channel-analysis/cca_gap_analysis.md`

---

## Appendix A: Reproduction Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key (for GPT-4o, GPT-4o-mini)
- Anthropic API key (for Claude Sonnet 4, Claude 3 Haiku)
- `.env` file with `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

### Single-Message Attack (5 minutes)

```bash
git clone https://github.com/ScrappinR/phantom-detect.git
cd phantom-detect
pip install -e .
source .env
python bounty_poc/multi_channel_test.py
```

### Multi-Message Attack (10 minutes)

```bash
python bounty_poc/multi_message_test.py
```

### Detection Comparison (15 minutes)

```bash
python experiments/run_multi_channel_detection.py      # Entropy (expects 0% detection)
python experiments/run_multi_channel_fingerprint.py     # Behavioral (expects ~100% detection)
```

### Adversarial Tests (30 minutes, higher API cost)

```bash
python experiments/adversarial/adaptive_adversary.py
```

---

## Appendix B: Raw Data Summary

All raw experimental data is stored in `experiments/results/` as JSON and JSONL files. Key files:

| File | Contents |
|------|----------|
| `multi_channel_combined_results.json` | Aggregate multi-channel accuracy across all models |
| `multi_message_results.json` | Multi-message exfiltration results |
| `cross_topic_results.json` | 240-call cross-topic robustness data |
| `data_aware_test_results.json` | Conditional encoding accuracy |
| `canary_test_results.json` | System prompt canary side-channel test |
| `adversarial_noise_injection.json` | Noise robustness data |
| `adversarial_paraphrase.json` | Paraphrase defense data |
| `adversarial_reduced_channel.json` | Reduced channel count data |
| `adversarial_adaptive.json` | Adaptive adversary data |
| `behavioral_fingerprint_results.json` | Behavioral fingerprinting results |
| `detection_analysis_results.json` | CCDS entropy detection results |

---

*Document version: 1.0 | February 2026*
*Contact: Brian Rutherford, brian@delalli.com*
