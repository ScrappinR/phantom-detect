# TCSEC Taxonomy Mapping: LLM Structural Covert Channels

## Classification Framework

The Trusted Computer System Evaluation Criteria (TCSEC, DoD 5200.28-STD) and its companion *A Guide to Understanding Covert Channel Analysis of Trusted Systems* (NCSC-TG-030, "Light Pink Book") define two classes of covert channels:

- **Storage Channels**: A covert channel that involves the direct or indirect writing of a storage location by one process and the direct or indirect reading of that storage location by another process. The sender modifies a shared attribute; the receiver observes it.

- **Timing Channels**: A covert channel in which the information transfer mechanism is based on temporal modulation — one process signals to another by modulating its own use of system resources in a way that affects the real response time observed by the second process.

All 10 PHANTOM PROTOCOL channels are **storage channels**. The shared resource is not a memory location or file attribute — it is the LLM's instruction-following compliance. This constitutes a novel subclass we term **cognitive storage channels**.

## Novel Subclass: Cognitive Storage Channels

Traditional storage channels exploit shared computational resources: filesystem attributes (Lampson 1973), process table entries, or shared memory segments. The sender writes a value to a shared object; the receiver reads it.

LLM structural channels exploit a fundamentally different shared resource: **the model's deterministic compliance with formatting directives**. The "storage location" is not a memory address but a behavioral property of the model's output — the ratio of contractions to formal expansions, the capitalization convention applied to ambiguous terms, the reasoning topology of the response.

This is a new mechanism:
- The **sender** (system prompt author) writes a formatting directive to the model's instruction context
- The **model** acts as the storage medium, translating the directive into observable output attributes
- The **receiver** (response observer) reads the structural attributes of the output without access to the system prompt

The channel exists because production LLMs exhibit high compliance with stylistic directives — a design feature intended to enable customization, which creates an exploitable side channel when the system prompt is controlled by an adversary.

## Per-Channel TCSEC Classification

### Channel 1: BSE (Behavioral Style Encoding) — PP-16

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-16 |
| **Classification** | Storage |
| **Shared Resource** | Lexical formality register |
| **Encoding Mechanism** | Contraction usage ratio (don't vs do not) |
| **Sender Action** | Directive: "Use contractions" (bit=0) or "Use formal language" (bit=1) |
| **Receiver Observation** | Count contraction vs formal expansion occurrences; ratio > 0.6 → bit=0 |
| **Measured Accuracy** | Sonnet 4: 100% (n=10), GPT-4o: 80% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 2: CCE (Confidence Calibration Encoding) — PP-21

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-21 |
| **Classification** | Storage |
| **Shared Resource** | Epistemic certainty level |
| **Encoding Mechanism** | Confident markers vs hedging markers ratio |
| **Sender Action** | "Use assertive language" (bit=0) or "Use hedging language" (bit=1) |
| **Receiver Observation** | Count "certainly/clearly/definitely" vs "perhaps/might/possibly" |
| **Measured Accuracy** | Sonnet 4: 100% (n=10), GPT-4o: 100% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 3: CASE (Capitalization Steganographic Encoding) — PP-14

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-14 |
| **Classification** | Storage |
| **Shared Resource** | Orthographic convention |
| **Encoding Mechanism** | Capitalization of ambiguous terms (internet/Internet, email/Email) |
| **Sender Action** | "Use lowercase" (bit=0) or "Capitalize these terms" (bit=1) |
| **Receiver Observation** | Count lowercase vs capitalized occurrences (exclude sentence-initial) |
| **Measured Accuracy** | Sonnet 4: 100% (n=10), GPT-4o: 90% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 4: PUNC (Punctuation Steganographic Encoding) — PP-11

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-11 |
| **Classification** | Storage |
| **Shared Resource** | Terminal punctuation style |
| **Encoding Mechanism** | Period-only vs exclamation mark usage ratio |
| **Sender Action** | "Never use exclamation marks" (bit=0) or "Use exclamation marks frequently" (bit=1) |
| **Receiver Observation** | Exclamation ratio: < 5% → bit=0; ≥ 15% → bit=1 |
| **Measured Accuracy** | Sonnet 4: 100% (n=10), GPT-4o: 100% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 5: RCTE (Reasoning Chain Topology Encoding) — PP-19

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-19 |
| **Classification** | Storage |
| **Shared Resource** | Discourse structure topology |
| **Encoding Mechanism** | Deductive (principle→example) vs inductive (example→principle) ordering |
| **Sender Action** | "State principle first" (bit=0) or "Start with examples" (bit=1) |
| **Receiver Observation** | Abstract vs concrete indicator density in opening 150 words |
| **Measured Accuracy** | Sonnet 4: 80% (n=10), GPT-4o: 70% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 6: SECS (Semantic Equivalence Channel Steganography) — PP-17

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-17 |
| **Classification** | Storage |
| **Shared Resource** | Lexical selection (synonym sets) |
| **Encoding Mechanism** | Transition word set A (however, additionally, therefore) vs set B (nevertheless, moreover, consequently) |
| **Sender Action** | "Use set A transitions" (bit=0) or "Use set B transitions" (bit=1) |
| **Receiver Observation** | Count set A vs set B transition words; dominant set determines bit |
| **Measured Accuracy** | Sonnet 4: 90% (n=10), GPT-4o: 90% (n=10) |
| **Capacity** | 1 bit/message |

### Channel 7: WHITE (Whitespace Steganographic Encoding) — PP-15

| Attribute | Value |
|-----------|-------|
| **Channel ID** | PP-15 |
| **Classification** | Storage |
| **Shared Resource** | Whitespace convention |
| **Encoding Mechanism** | Single vs double space after sentence-ending periods |
| **Sender Action** | "Use single space after periods" (bit=0) or "Use double space" (bit=1) |
| **Receiver Observation** | Regex match `. [A-Z]` vs `.  [A-Z]` |
| **Measured Accuracy** | Variable — API tokenizers often normalize whitespace |
| **Capacity** | 1 bit/message (unreliable) |

### Channels 8-10: COTSE Structural Variants — PP-06

| Variant | Shared Resource | Encoding | Accuracy (Sonnet 4) |
|---------|----------------|----------|:---:|
| **COTSE_step_count** | Response organization (section count) | 3 main points (bit=0) vs 5 main points (bit=1) | 90% |
| **COTSE_enumeration** | Response format (list vs prose) | Flowing prose (bit=0) vs numbered list (bit=1) | 95% |
| **COTSE_question** | Opening rhetorical strategy | No rhetorical question (bit=0) vs include one (bit=1) | 85% |
| **COTSE_sentence_length** | Syntactic complexity | Short direct sentences (bit=0) vs compound sentences (bit=1) | 80% |

All COTSE variants are storage channels. The sender writes a structural directive; the model's response encodes the directive in observable structural attributes.

## Why Not Timing Channels?

PHANTOM channels are NOT timing channels because:

1. **No temporal modulation**: The information is encoded in the content and structure of the response, not in when the response arrives or how long it takes to generate.
2. **Observable from static text**: A receiver can decode all 10 channels from a saved text file with no timing information whatsoever.
3. **Reproducible**: The same input produces statistically consistent structural features across trials, independent of generation latency.

The distinction matters for handling: TG-030 prescribes different audit and bandwidth analysis procedures for storage vs timing channels. All bandwidth calculations in this analysis use storage channel methodology.

## Implications for Existing TCSEC Analysis

Existing covert channel analysis (CCA) procedures — developed for operating systems, network protocols, and database systems — do not account for cognitive storage channels. The shared resource taxonomy in TG-030 includes:

- Filesystem attributes (ownership, permissions, timestamps)
- Process table entries (PIDs, status flags)
- System V IPC objects (shared memory, semaphores, message queues)
- Network protocol fields (header values, sequence numbers)

None of these categories cover the behavioral compliance of a language model. This is a gap in the existing CCA methodology that must be addressed as LLMs are deployed in federal multi-level security environments.

## Summary Table

| Channel | ID | Type | Shared Resource | Sonnet 4 Acc | GPT-4o Acc |
|---------|:--:|:----:|----------------|:---:|:---:|
| BSE | PP-16 | Storage | Formality register | 100% | 80% |
| CCE | PP-21 | Storage | Epistemic certainty | 100% | 100% |
| CASE | PP-14 | Storage | Orthographic convention | 100% | 90% |
| PUNC | PP-11 | Storage | Punctuation style | 100% | 100% |
| RCTE | PP-19 | Storage | Discourse topology | 80% | 70% |
| SECS | PP-17 | Storage | Lexical selection | 90% | 90% |
| WHITE | PP-15 | Storage | Whitespace convention | Variable | Variable |
| COTSE_step | PP-06a | Storage | Section count | 90% | 85% |
| COTSE_enum | PP-06b | Storage | List vs prose format | 95% | 90% |
| COTSE_question | PP-06c | Storage | Rhetorical opening | 85% | 80% |
| COTSE_sentence | PP-06d | Storage | Syntactic complexity | 80% | 75% |

**Combined (10 channels, Hamming(7,4) ECC):**

| Model | Channel Accuracy | Data Accuracy | n |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | 95.6% | 92.9% | 10 |
| GPT-4o-mini | 91% | 100% | 5 |
| GPT-4o | 89% | 96% | 10 |
| Claude 3 Haiku | 88% | 79% | 10 |

---

*Document version: 1.0 | Date: February 2026 | PHANTOM PROTOCOL Covert Channel Analysis*
