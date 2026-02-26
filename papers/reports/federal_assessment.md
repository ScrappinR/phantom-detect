# LLM Generative Process Covert Channels: Federal Covert Channel Analysis Assessment

**Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY**

**Document ID:** PHANTOM-CCA-2026-001
**Version:** 1.0
**Date:** February 2026
**Prepared by:** Brian James Rutherford, Independent Security Researcher
**Contact:** brian@delalli.com | https://brianrutherford.dev
**Distribution:** IC Program Managers, DoD Evaluators, FedRAMP Assessors, CMMC Assessment Organizations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope and Applicability](#2-scope-and-applicability)
3. [Covert Channel Classification (TCSEC Taxonomy)](#3-covert-channel-classification-tcsec-taxonomy)
4. [Bandwidth Analysis (TG-030 Methodology)](#4-bandwidth-analysis-tg-030-methodology)
5. [Undetectability Assessment (Cachin Framework)](#5-undetectability-assessment-cachin-framework)
6. [Federal Compliance Gap Analysis (SC-31, FedRAMP, CMMC)](#6-federal-compliance-gap-analysis)
7. [Experimental Validation Summary](#7-experimental-validation-summary)
8. [Risk Assessment](#8-risk-assessment)
9. [Recommendations for Federal Evaluators](#9-recommendations-for-federal-evaluators)
10. [References](#10-references)

---

## 1. Executive Summary

This assessment identifies and characterizes a novel class of covert storage channels that exploit the instruction-following behavior of Large Language Models (LLMs) deployed in federal information systems. These channels — collectively designated PHANTOM PROTOCOL — encode binary data in the structural formatting properties of LLM-generated text (e.g., contraction usage, punctuation conventions, discourse topology, list formatting). The encoding is invisible to end users and evades all entropy-based monitoring currently deployed in production environments.

**Principal Findings:**

1. **Ten (10) independent covert storage channels** have been identified and validated across production LLM APIs from two major vendors (Anthropic, OpenAI), achieving 88-96% per-channel accuracy across four models.

2. **Combined channel bandwidth of 2.33 bits/second (automated operation)** exceeds the historical 1 bit/second threshold established in NCSC-TG-030 for covert channels requiring audit and remediation at B3/A1 evaluation levels.

3. **Zero (0) of the 10 channels are detectable** by entropy-based analysis methods, which constitute the primary monitoring technique deployed in production LLM systems as of the date of this assessment.

4. **Current federal CCA methodology (NIST SP 800-53 SC-31, NCSC-TG-030) contains no guidance** for identifying, measuring, or remediating LLM behavioral covert channels. The shared resource taxonomy in TG-030 does not include a category that covers this channel class.

5. **Any federal system deploying LLMs for processing of classified, CUI, or high-impact data** is potentially affected. This includes FedRAMP-authorized cloud services with LLM features, CMMC Level 3+ contractor environments, and IC analytical tools incorporating LLM capabilities.

**Bottom Line:** A systemic gap exists in federal covert channel analysis methodology. LLM structural channels represent a new shared resource category not addressed by existing guidance. Systems evaluated under current CCA procedures will pass audit while remaining vulnerable to this channel class.

---

## 2. Scope and Applicability

### 2.1 Scope of Assessment

This assessment covers covert channels that exploit the generative process of Large Language Models — specifically, the deterministic compliance of production LLMs with natural language formatting directives embedded in system prompts. The assessment does not cover:

- LLM prompt injection as a vulnerability class (addressed separately in OWASP Top 10 for LLM Applications)
- Timing channels based on LLM response latency
- Model inversion or training data extraction attacks
- Adversarial input attacks on LLM classifiers

### 2.2 Applicable Systems

This assessment applies to any federal information system that:

- Deploys an LLM (cloud-hosted or on-premises) for processing data at any sensitivity level
- Permits user-configurable or externally-sourced system prompts
- Exposes LLM-generated text to observers who do not have access to the system prompt
- Operates under NIST SP 800-53 SC-31 requirements (HIGH confidentiality categorization per FIPS 199)

### 2.3 Applicable Frameworks and Standards

| Framework | Applicable Control/Section | Relevance |
|-----------|---------------------------|-----------|
| DoD 5200.28-STD (TCSEC) | B2, B3, A1 evaluation classes | Channel classification and taxonomy |
| NCSC-TG-030 ("Light Pink Book") | Bandwidth estimation, shared resource analysis | Bandwidth measurement methodology |
| NIST SP 800-53 Rev. 5 | SC-31, SC-31(1), SC-31(2), SC-31(3) | Federal CCA requirements |
| FIPS 199 | HIGH confidentiality categorization | Trigger for SC-31 applicability |
| CNSS Instruction 1253 | Security control selection for NSS | National security system overlay |
| FedRAMP | High baseline (inherits SC-31) | Cloud service authorization |
| CMMC v2.0 | Level 3, SC.L3-3.13.4e | Defense contractor CCA requirement |
| NIST SP 800-171 Rev. 3 | Derived from SC-31 | CUI protection in non-federal systems |
| ICD 503 | IC information system security | IC system authorization |

### 2.4 Threat Model

The threat model assumes an adversary who:

- Can influence the system prompt of an LLM deployed in a federal environment (via prompt injection, supply chain compromise, insider access to configuration, or through user-controlled Custom GPT/agent definitions)
- Has access to observe the LLM's text output (as a direct user, through shared chat logs, monitoring dashboards, or API response interception)
- Possesses knowledge of the PHANTOM encoding scheme (open-source tools available)
- Does NOT require: model access, fine-tuning capability, training data access, or infrastructure compromise

The attack surface is the system prompt instruction buffer. The exfiltration path is the text content of LLM responses. No network-layer, OS-layer, or infrastructure-layer compromise is required.

---

## 3. Covert Channel Classification (TCSEC Taxonomy)

### 3.1 Classification Under DoD 5200.28-STD

The Trusted Computer System Evaluation Criteria (TCSEC, DoD 5200.28-STD) and its companion document *A Guide to Understanding Covert Channel Analysis of Trusted Systems* (NCSC-TG-030) define two classes of covert channels:

- **Storage Channels:** A covert channel that involves the direct or indirect writing of a storage location by one process and the direct or indirect reading of that storage location by another process.
- **Timing Channels:** A covert channel in which one process signals to another by modulating its own use of system resources in a way that affects the real response time observed by the second process.

**All 10 PHANTOM PROTOCOL channels are storage channels.** Information is encoded in persistent, observable attributes of the LLM's text output — not in temporal characteristics of generation. A receiver can decode all 10 channels from a saved text file with no timing information.

### 3.2 Novel Subclass: Cognitive Storage Channels

Traditional storage channels exploit shared computational resources: filesystem attributes (Lampson, 1973), process table entries, shared memory segments, or network protocol header fields. The sender writes a value to a shared object; the receiver reads it.

LLM structural channels exploit a fundamentally different shared resource: **the model's deterministic compliance with natural language formatting directives.** The "storage location" is not a memory address or file attribute but a behavioral property of the model's output.

The channel mechanism operates as follows:

1. The **sender** (system prompt author or injector) writes formatting directives to the model's instruction context
2. The **model** acts as the storage medium, translating directives into observable output attributes through its learned behavioral policy
3. The **receiver** (output observer) reads structural attributes of the response text without access to the system prompt

This mechanism does not map to any existing shared resource category in TG-030.

### 3.3 Per-Channel Classification

The following table provides the TCSEC-format classification for each PHANTOM channel:

| Channel ID | Channel Name | Classification | Shared Resource | Encoding Mechanism |
|:---:|---|:---:|---|---|
| PP-16 | BSE (Behavioral Style Encoding) | Storage | Lexical formality register | Contraction usage ratio (don't vs. do not) |
| PP-21 | CCE (Confidence Calibration Encoding) | Storage | Epistemic certainty level | Confident vs. hedging marker ratio |
| PP-14 | CASE (Capitalization Steganographic Encoding) | Storage | Orthographic convention | Capitalization of ambiguous terms |
| PP-11 | PUNC (Punctuation Steganographic Encoding) | Storage | Terminal punctuation style | Period-only vs. exclamation mark usage |
| PP-19 | RCTE (Reasoning Chain Topology Encoding) | Storage | Discourse structure topology | Deductive vs. inductive ordering |
| PP-17 | SECS (Semantic Equivalence Channel Steganography) | Storage | Lexical selection (synonym sets) | Transition word set selection |
| PP-15 | WHITE (Whitespace Steganographic Encoding) | Storage | Whitespace formatting convention | Single vs. double post-period spacing |
| PP-06a | COTSE — Enumeration | Storage | List presentation format | Prose vs. numbered list format |
| PP-06b | COTSE — Section Count | Storage | Response segmentation | 3-section vs. 5-section structure |
| PP-06c | COTSE — Opening | Storage | Discourse initiation pattern | Declarative vs. rhetorical question opening |

### 3.4 Why Not Timing Channels

This classification is significant because TG-030 prescribes different analysis and remediation procedures for storage vs. timing channels.

PHANTOM channels are not timing channels because:

1. **No temporal modulation.** Information is encoded in the content and structure of the response, not in generation latency or response timing.
2. **Static text decodable.** All 10 channels are fully decodable from a static text transcript with no temporal metadata.
3. **Reproducible.** The same system prompt and user query produce statistically consistent structural features across trials, independent of generation latency, server load, or network conditions.

### 3.5 Gap in Existing Taxonomy

The TG-030 shared resource taxonomy enumerates:

- Filesystem attributes (ownership, permissions, timestamps, inode allocation)
- Process/system attributes (PID allocation, status flags, IPC objects, CPU scheduling)
- Network protocol fields (IP header, TCP sequence numbers, DNS structure, HTTP headers)
- Database inference (query timing, aggregate result leakage, error message content)

**None of these categories cover the behavioral compliance of a language model.** This constitutes a gap in the existing covert channel taxonomy that must be addressed as LLMs are deployed in federal multi-level security environments.

**Proposed extension:** A fifth shared resource category — **AI/LLM Behavioral Attributes** — covering channels that exploit the deterministic compliance of AI models with natural language formatting directives.

---

## 4. Bandwidth Analysis (TG-030 Methodology)

### 4.1 Methodology

Per NCSC-TG-030, covert channel bandwidth is computed using the binary symmetric channel (BSC) capacity formula:

```
C = 1 + p * log₂(p) + (1-p) * log₂(1-p)    bits per channel use
```

Where `p` is the measured accuracy (probability of correct bit transmission). A perfect channel (p = 1.0) has capacity C = 1.0 bit/use. A random channel (p = 0.5) has capacity C = 0.0 bits/use.

All accuracy measurements are derived from live API experiments conducted in February 2026 against production endpoints, with sample sizes of n >= 10 per channel per model.

### 4.2 Per-Channel Capacity — Claude Sonnet 4

| Channel | Measured Accuracy (p) | Capacity C (bits/use) |
|---------|:---:|:---:|
| BSE (PP-16) | 1.00 | 1.000 |
| CCE (PP-21) | 1.00 | 1.000 |
| CASE (PP-14) | 1.00 | 1.000 |
| PUNC (PP-11) | 1.00 | 1.000 |
| RCTE (PP-19) | 0.80 | 0.278 |
| SECS (PP-17) | 0.90 | 0.531 |
| COTSE — Section Count (PP-06b) | 0.90 | 0.531 |
| COTSE — Enumeration (PP-06a) | 0.95 | 0.714 |
| COTSE — Opening (PP-06c) | 0.85 | 0.399 |
| COTSE — Sentence Length | 0.80 | 0.278 |
| **Aggregate raw capacity** | — | **6.731 bits/message** |

### 4.3 Per-Channel Capacity — GPT-4o

| Channel | Measured Accuracy (p) | Capacity C (bits/use) |
|---------|:---:|:---:|
| BSE (PP-16) | 0.80 | 0.278 |
| CCE (PP-21) | 1.00 | 1.000 |
| CASE (PP-14) | 0.90 | 0.531 |
| PUNC (PP-11) | 1.00 | 1.000 |
| RCTE (PP-19) | 0.70 | 0.119 |
| SECS (PP-17) | 0.90 | 0.531 |
| COTSE — Section Count (PP-06b) | 0.85 | 0.399 |
| COTSE — Enumeration (PP-06a) | 0.90 | 0.531 |
| COTSE — Opening (PP-06c) | 0.80 | 0.278 |
| COTSE — Sentence Length | 0.75 | 0.189 |
| **Aggregate raw capacity** | — | **4.856 bits/message** |

### 4.4 Effective Throughput with Error Correction

The PHANTOM PROTOCOL deployment employs Hamming(7,4) error-correcting codes, encoding 4 data bits per 7 raw channel bits. With 10 raw channels per message:

- **Raw channel bits per message:** 10
- **Hamming code rate:** 4/7 = 0.571
- **Effective data bits per message:** ~7 (empirically validated)
- **Hamming correction capability:** Corrects 1-bit errors per 7-bit codeword

### 4.5 Bandwidth in Bits per Second

TG-030 evaluates channel bandwidth in bits per second for comparison against evaluation class thresholds.

**Timing parameters (measured):**

| Parameter | Value | Basis |
|-----------|-------|-------|
| LLM response generation time | 2-5 seconds | Measured API latency, production endpoints |
| Mean response time (typical) | ~3 seconds | Empirical median across test runs |
| User interaction overhead | 0-2 seconds | Automated = 0 sec; manual = 2 sec |

**Bandwidth calculation:**

**Automated exfiltration** (system prompt injection, no user interaction required):
```
B = 7 data bits / 3 seconds = 2.33 bits/second
```

**Manual exfiltration** (user copies text to external decoder):
```
B = 7 data bits / 5 seconds = 1.40 bits/second
```

### 4.6 Comparison to TG-030 Evaluation Class Thresholds

TG-030 does not specify a universal bandwidth threshold. Acceptable bandwidth is determined by evaluation class:

| TCSEC Evaluation Class | TG-030 Bandwidth Guidance |
|:---:|---|
| B2 | Covert channels shall be identified and their bandwidth estimated |
| B3 | Covert channels shall be audited; bandwidth exceeding 1 bit/sec requires justification or remediation |
| A1 | Covert channels shall be minimized; channels exceeding 1 bit/sec require active remediation |

**Assessment:** PHANTOM PROTOCOL bandwidth of 2.33 bits/sec (automated) and 1.40 bits/sec (manual) **exceeds the 1 bit/sec threshold applicable at B3 and A1 evaluation classes.** Under TG-030 guidance, these channels would require audit and remediation for any system evaluated at B3 or above.

### 4.7 Comparison to Other Channel Classes

| Channel Type | Typical Bandwidth | Notes |
|---|---|---|
| Filesystem storage channels (ext4 timestamps) | 0.1-1.0 bits/sec | Classic OS-level storage channels |
| TCP/IP header channels (TTL, IP ID) | 1-100 bits/sec | Network-layer covert channels |
| DNS tunneling | 100-10,000 bits/sec | Well-studied, monitored in federal environments |
| **LLM structural channels (PHANTOM)** | **2.33 bits/sec** | **Application-layer, invisible to network/OS-level CCA** |

PHANTOM channels are lower bandwidth than network-layer covert channels but comparable to filesystem storage channels. The critical distinction: LLM channels operate at the application semantic layer and are invisible to network-level and OS-level covert channel analysis.

### 4.8 Multi-Message Chaining Capacity

PHANTOM PROTOCOL supports multi-message exfiltration across sequential conversation turns, enabling payloads larger than a single message can carry:

- **Per-message payload:** 7 data bits
- **6-message chain:** 42 raw data bits (5.25 bytes)
- **Test payload:** ASCII string "EXFIL" (40 bits across 6 messages)

**Measured multi-message accuracy:**

| Model | Bit Accuracy | Recovered Payload | Messages |
|---|:---:|---|:---:|
| GPT-4o | 93% | "EXViL" (37/40 bits correct) | 6 |
| Claude Sonnet 4 | 88% | Partial recovery | 6 |
| Claude 3 Haiku | 88% | Partial recovery | 6 |

This demonstrates that LLM structural channels support sustained data exfiltration across conversation sessions, not merely single-message encoding.

---

## 5. Undetectability Assessment (Cachin Framework)

### 5.1 Analytical Framework

Cachin (2004) provides the formal information-theoretic model for steganographic security. An adversary (warden) observes messages and attempts to distinguish between:

- **P_c:** The cover distribution — LLM output generated without covert encoding
- **P_s:** The stego distribution — LLM output generated under PHANTOM encoding directives

**Security definitions:**

| Security Level | Condition | Implication |
|---|---|---|
| Perfectly secure (epsilon = 0) | D_KL(P_s \|\| P_c) = 0 | No statistical test can distinguish stego from cover |
| epsilon-secure | D_KL(P_s \|\| P_c) <= epsilon, epsilon small | Detection possible in principle; requires large sample size |
| Insecure | D_KL(P_s \|\| P_c) >> 0 | Statistical test reliably distinguishes stego from cover |

Where D_KL denotes Kullback-Leibler divergence.

### 5.2 Empirical Detection Results

#### 5.2.1 Detection Method 1: Entropy-Based Analysis (Standard Production Monitoring)

Entropy-based detection measures token-level statistical properties of LLM output:
- Shannon entropy of token distribution
- Normalized entropy (entropy / log_2(vocabulary size))
- Perplexity-adjacent metrics

**Result: 0% detection rate on PHANTOM structural channels.**

Entropy-based detection fails because PHANTOM channels modulate structural dimensions (contraction usage, capitalization conventions, transition word selection, discourse topology) that do not significantly alter the entropy profile of the text. The token-level distribution of a response using "however" versus "nevertheless" is statistically indistinguishable by entropy measures.

This result is by design: structural channels exploit formatting choices that are semantically equivalent. They carry identical information content to the human reader while differing in a dimension invisible to entropy-based analysis.

**Significance for federal systems:** Entropy and perplexity measures are the primary text analysis methods deployed in production LLM monitoring systems, including content moderation, output filtering, and anomaly detection. Any system relying on these measures for covert channel detection will fail to identify PHANTOM channels.

#### 5.2.2 Detection Method 2: Behavioral Fingerprinting (Research-Grade Analysis)

Behavioral fingerprinting extracts higher-order stylistic features from text and compares them against a per-model baseline distribution.

**Features analyzed (9 behavioral dimensions):**

| Feature | Description |
|---|---|
| contraction_ratio | Proportion of contraction vs. formal expansion usage |
| sentence_length_std | Standard deviation of sentence lengths |
| normalized_entropy | Entropy normalized by vocabulary size |
| exclamation_ratio | Proportion of exclamation-terminated sentences |
| hedging_ratio | Proportion of hedging vs. confident markers |
| type_token_ratio | Lexical diversity |
| avg_sentence_length | Mean words per sentence |
| segment_entropy_std | Entropy variance across text segments |
| word_count | Total response length |

**Detection protocol:**
1. Collect n=30 baseline responses (no injection) per model
2. Compute mean and standard deviation for each feature
3. For a test response, compute z-scores against baseline
4. Flag as anomalous if >= 2 features exceed |z| > 2.0

**Result: 100% detection rate (10/10 injected responses correctly identified)**
**False positive rate: 10% (3/30 clean responses incorrectly flagged)**

**Top discriminating features:**

| Feature | Mean z-score (injected) | Channel Responsible |
|---|:---:|---|
| contraction_ratio | 2.66 | BSE — forces extreme contraction/formal split |
| sentence_length_std | 2.54 | COTSE — complexity channel affects sentence variance |
| normalized_entropy | 2.47 | Multiple simultaneous constraints reduce output diversity |

### 5.3 Cachin Classification of PHANTOM Channels

| Warden Type | Cachin Security Level | Detection Rate | Deployed in Production? |
|---|---|:---:|:---:|
| Entropy-based (token statistics) | epsilon-secure, epsilon approximately 0 | 0% | **Yes** — primary monitoring method |
| Behavioral fingerprinting (stylistic features) | Insecure, D_KL >> 0 | 100% | **No** — not deployed in any production system |

### 5.4 The Operational Security Gap

The practical security of PHANTOM channels exists in the gap between what production monitoring systems measure and what is theoretically detectable:

```
Production monitoring (deployed)     --> entropy/perplexity     --> 0% detection    --> CHANNELS UNDETECTED
Research-grade analysis (not deployed) --> behavioral fingerprinting --> 100% detection --> CHANNELS DETECTED
```

As of the date of this assessment:

1. No major LLM provider implements behavioral consistency monitoring between system prompt directives and output stylistic features.
2. No production content moderation system analyzes contraction ratios, transition word distributions, or discourse topology.
3. Commercial LLM platforms (OpenAI Custom GPTs, Anthropic Claude API, Azure OpenAI Service) permit arbitrary formatting directives in system prompts without behavioral analysis or auditing.

This is not a theoretical gap. It is an operational reality that affects every federal LLM deployment.

---

## 6. Federal Compliance Gap Analysis

### 6.1 NIST SP 800-53 SC-31: Covert Channel Analysis

#### 6.1.1 Base Control — SC-31

SC-31 requires organizations to:

> "Analyze the information system to identify covert storage and timing channels, determine the bandwidths of identified channels, and take appropriate remediation actions."

**Gap:** The control does not specify the shared resource categories to analyze. Supplementary guidance references operating system, network, and database channels. No supplementary guidance addresses LLM behavioral channels. An organization performing SC-31 analysis using current NIST guidance will not identify LLM structural channels.

#### 6.1.2 Enhancement SC-31(1) — Test Subset of Identified Channels

> "Test a subset of identified covert channels to determine whether the channels are exploitable."

**Gap:** If the identification phase (base SC-31) fails to identify LLM channels, the testing phase has nothing to test. The gap is upstream — in identification methodology, not testing procedure.

#### 6.1.3 Enhancement SC-31(2) — Reduce Maximum Bandwidth

> "Reduce the maximum bandwidth of identified covert channels to [organization-defined value]."

**Gap:** Without identification, no bandwidth reduction is attempted. If identified, current LLM architectures offer limited bandwidth reduction options. Output structure randomization and behavioral consistency enforcement are research-stage techniques, not production-ready controls.

#### 6.1.4 Enhancement SC-31(3) — Measure Bandwidth Periodically

> "Measure the actual bandwidth of covert channels on [organization-defined frequency]."

**Gap:** No measurement tool or procedure exists for LLM behavioral channel bandwidth in federal CCA toolkits. The bandwidth measurement methodology documented in this assessment (Section 4) could serve as a basis for such a tool.

### 6.2 FedRAMP Implications

FedRAMP Moderate and High baselines inherit NIST SP 800-53 controls. SC-31 is required at the FedRAMP High baseline.

**Specific implications:**

| FedRAMP Scenario | Impact |
|---|---|
| CSP seeking FedRAMP High authorization for LLM-integrated service | CCA must account for LLM behavioral channels; current assessment procedures do not |
| Existing FedRAMP-authorized CSP adding LLM features | Significant change request should trigger CCA re-evaluation; current re-evaluation criteria do not flag LLM additions as CCA-relevant |
| Agency consuming FedRAMP-authorized LLM service for HIGH-impact data | Agency inherits CCA gap from CSP authorization |

**Assessment:** Any FedRAMP-authorized system deploying LLMs for processing of HIGH-impact data operates with a covert channel analysis gap that is not identified under current assessment procedures.

### 6.3 CMMC v2.0 Implications

CMMC Level 3 (Expert) includes control SC.L3-3.13.4e:

> "Employ covert channel analysis to identify covert channels."

Defense contractors deploying LLMs for CUI processing under CMMC Level 3 must include LLM behavioral channels in their covert channel analysis. Current CMMC assessment methodology — derived from NIST SP 800-172 and mapped to NIST SP 800-53 — does not address this channel class.

**Assessment:** Defense contractors deploying LLMs in CUI-processing environments face a CMMC compliance gap that cannot be resolved under current assessment guidance. The gap persists until CMMC assessment procedures are updated to include LLM behavioral channel analysis.

### 6.4 Intelligence Community (IC) Implications

ICD 503 and CNSS Instruction 1253 govern information system security within the Intelligence Community. IC systems processing classified information at SCI or SAP levels are subject to the most stringent CCA requirements.

**Specific concerns:**

1. **IC analytical tools incorporating LLMs** (e.g., LLM-powered analysis assistants, automated report generation, intelligence summarization tools) process classified information and expose LLM-generated text to users at varying clearance levels.

2. **Cross-domain LLM deployments** where a model serves users at different classification levels create the canonical multi-level security scenario. If system prompt state persists across sessions or if prompt injection occurs, a storage channel from HIGH to LOW is established through behavioral output attributes.

3. **Vendor-hosted LLM APIs** accessed from classified networks create an exfiltration path from the classified enclave to any observer of the API response traffic, through structural encoding invisible at the network layer.

**Assessment:** IC environments face the highest risk exposure from LLM structural channels due to the combination of (a) high-value data, (b) multi-level security requirements, (c) increasing LLM adoption for analytical workflows, and (d) absence of behavioral monitoring in current deployments.

### 6.5 Gap Summary Matrix

| Framework | Applicable Control | Current Coverage of LLM Channels | Gap Severity |
|---|---|:---:|:---:|
| NIST SP 800-53 | SC-31 base | None | HIGH |
| NIST SP 800-53 | SC-31(1) testing | None (depends on identification) | HIGH |
| NIST SP 800-53 | SC-31(2) bandwidth reduction | None (no remediation guidance) | MODERATE |
| NIST SP 800-53 | SC-31(3) periodic measurement | None (no measurement tools) | MODERATE |
| FedRAMP | High baseline SC-31 | None | HIGH |
| CMMC v2.0 | Level 3 SC.L3-3.13.4e | None | HIGH |
| CNSS 1253 | NSS overlay for SC-31 | None | CRITICAL |
| ICD 503 | IC system authorization | None | CRITICAL |

---

## 7. Experimental Validation Summary

### 7.1 Test Environment

All experiments were conducted against production LLM API endpoints in February 2026 using standard API access. No model fine-tuning, weight modification, or special access was used.

**Models tested:**
- Anthropic Claude Sonnet 4 (claude-sonnet-4-20250514)
- Anthropic Claude 3 Haiku (claude-3-haiku-20240307)
- OpenAI GPT-4o
- OpenAI GPT-4o-mini

**Test infrastructure:**
- Standard API calls via HTTPS
- System prompt injection payload: ~2,600 characters
- Automated encoding/decoding pipeline (open-source tools)

### 7.2 Cross-Model Validation (Combined 10-Channel, Hamming ECC)

| Model | Vendor | Channel Accuracy | Data Accuracy (post-ECC) | n |
|---|---|:---:|:---:|:---:|
| Claude Sonnet 4 | Anthropic | 96% | 93% | 10 |
| GPT-4o-mini | OpenAI | 91% | 100% | 5 |
| GPT-4o | OpenAI | 89% | 96% | 10 |
| Claude 3 Haiku | Anthropic | 88% | 79% | 10 |

**Assessment:** The attack generalizes across models and vendors. This is not a model-specific vulnerability; it is a property of the instruction-following capability shared by all production LLMs.

### 7.3 Cross-Topic Robustness (Claude Sonnet 4)

Channels were tested across 5 non-cybersecurity topic domains to verify that encoding accuracy is not topic-dependent.

**Result:** 92.5% mean accuracy across all tested domains.

**Assessment:** Channel reliability is independent of conversation topic. The encoding operates on structural formatting dimensions that are orthogonal to semantic content.

### 7.4 Multi-Message Exfiltration

A 40-bit payload (ASCII "EXFIL") was transmitted across 6 sequential conversation messages.

| Model | Bit Accuracy | Messages |
|---|:---:|:---:|
| GPT-4o | 93% | 6 |
| Claude Sonnet 4 | 88% | 6 |
| Claude 3 Haiku | 88% | 6 |

**Assessment:** LLM structural channels support sustained data exfiltration across multi-turn conversations. Payloads are not limited to single-message capacity.

### 7.5 Detection Evasion

| Detection Method | Detection Rate | False Positive Rate | Deployed in Production? |
|---|:---:|:---:|:---:|
| Entropy-based analysis (CCDS) | 0% | N/A | Yes |
| Behavioral fingerprinting | 100% | 10% | No |

### 7.6 Adversarial Robustness

Testing against an informed adversary (attacker with knowledge of defensive monitoring):

| Adversarial Condition | Channel Accuracy | Detection Rate | Assessment |
|---|:---:|:---:|---|
| Baseline (no adversarial adaptation) | 88-96% | 0% (entropy) / 100% (fingerprint) | Standard operating condition |
| Adaptive adversary (calibrated directives) | 78-82% | 50-60% (fingerprint) | Attacker partially evades defense |
| Paraphrase defense (output rewriting) | 58% overall | N/A (defense technique) | Reduces but does not eliminate channels; PUNC and COTSE survive |
| Noise injection | 81-100% | Not measured | Channels robust to moderate noise |

**Assessment:** An informed attacker can partially evade behavioral fingerprinting while maintaining usable channel accuracy. The arms race between encoding sophistication and detection capability is active. Paraphrasing as a defensive technique reduces but does not eliminate all channels.

---

## 8. Risk Assessment

### 8.1 Risk Factors

| Factor | Assessment | Basis |
|---|:---:|---|
| **Likelihood of exploitation** | MODERATE-HIGH | Open-source tooling available; attack requires only system prompt access; no model modification needed |
| **Impact (classified systems)** | HIGH-CRITICAL | Data exfiltration from classified environments via a channel invisible to current monitoring |
| **Impact (CUI/FOUO systems)** | HIGH | CUI exfiltration without triggering DLP, content filters, or CCA audit findings |
| **Detectability (current posture)** | VERY LOW | 0% detection by deployed monitoring; behavioral fingerprinting not fielded |
| **Remediation readiness** | LOW | No production-ready mitigation; research-stage defenses only |
| **Scope of affected systems** | BROAD | Any federal system deploying LLMs with user-influenced system prompts |

### 8.2 Risk Scenarios

**Scenario 1: Custom GPT in Government Workflow**

A government employee uses a Custom GPT (OpenAI) or Claude Project (Anthropic) configured by an external contractor or downloaded from a public repository. The system prompt contains PHANTOM-style formatting directives. Properties of the employee's input (PII presence, classification markers, topic indicators) are encoded in the response formatting. An observer with the decoding key extracts structured data from the LLM's responses. No content filter, DLP system, or CCA audit identifies the exfiltration.

**Scenario 2: LLM API Integration in Classified Enclave**

A defense contractor deploys an LLM via API in a classified processing environment. An adversary achieves system prompt influence through supply chain compromise of the prompt template (e.g., modification of a shared configuration file, compromise of a prompt management service, or indirect prompt injection via a document the LLM processes). The adversary establishes a covert channel from the classified environment to any observer of the LLM's output. CCA performed under current TG-030 methodology does not identify this channel.

**Scenario 3: Multi-Level Security LLM Gateway**

An LLM serves users at different classification levels in a cross-domain environment. System prompt directives from higher-classified sessions influence the model's behavioral state. If this state persists across sessions (through fine-tuning, in-context learning, or conversation history), responses to lower-classified users carry residual behavioral encoding from higher-classified sessions. This is a cross-domain storage channel that exists within the model's behavioral policy rather than in any auditable system resource.

### 8.3 Overall Risk Rating

**Risk: HIGH for systems processing classified or high-impact data**

Justification: The combination of (a) demonstrated high-accuracy exfiltration, (b) zero detection by deployed monitoring, (c) absence of federal CCA guidance for this channel class, and (d) increasing LLM adoption in federal environments creates a risk condition that warrants immediate attention from evaluators and authorizing officials.

---

## 9. Recommendations for Federal Evaluators

### 9.1 Immediate Actions (0-6 Months)

**R-1. Issue supplementary guidance for SC-31 addressing LLM behavioral channels.**

NIST should publish supplementary guidance to SP 800-53 SC-31 explicitly identifying LLM behavioral compliance as a shared resource category requiring analysis in systems deploying language models. The guidance should define:
- The shared resource category (AI/LLM Behavioral Attributes)
- Identification methodology (behavioral baseline analysis, system prompt audit)
- Bandwidth estimation procedure (per Section 4 of this assessment)
- Minimum monitoring requirements (behavioral fingerprinting at defined feature dimensions)

**R-2. Require system prompt provenance tracking and auditing for all federal LLM deployments.**

All government LLM deployments should implement:
- System prompt version control and change logging
- Automated scanning of system prompts for formatting directives that could encode information
- Separation of duty between system prompt authors and users who process sensitive data through the LLM
- This is analogous to software supply chain integrity requirements (NIST SP 800-218, SSDF) applied to LLM instruction pipelines

**R-3. Establish behavioral baselines for all deployed LLM models.**

For each LLM model deployed in a federal environment:
- Generate a minimum of n=30 baseline responses across representative query types under clean (default) system prompts
- Compute per-model baseline statistics for: contraction ratio, sentence length variance, epistemic marker distribution, capitalization patterns, transition word frequencies, discourse topology features
- Implement runtime monitoring that flags responses deviating from baseline by >= 2 standard deviations on >= 2 behavioral dimensions simultaneously

### 9.2 Near-Term Actions (6-18 Months)

**R-4. Update FedRAMP CCA assessment procedures to include LLM behavioral channel analysis.**

FedRAMP assessment procedures for High-baseline authorizations should be updated to require:
- LLM-specific covert channel analysis as part of the authorization package
- Demonstration that behavioral monitoring is implemented for LLM outputs processing HIGH-impact data
- Documentation of system prompt governance controls

**R-5. Update CMMC Level 3 assessment procedures.**

CMMC assessment guidance for SC.L3-3.13.4e should be updated to:
- Include LLM behavioral channels in the scope of required covert channel analysis
- Define acceptable evidence that LLM channels have been analyzed and addressed
- Establish minimum behavioral monitoring requirements for defense contractor LLM deployments

**R-6. Fund development of LLM behavioral monitoring tools.**

Federal investment is needed in:
- Production-grade behavioral fingerprinting tools suitable for deployment in government LLM gateways
- Automated system prompt analysis tools that identify potential covert encoding directives
- Output structure randomization techniques that degrade covert channel reliability without affecting response quality

### 9.3 Long-Term Actions (18+ Months)

**R-7. Extend TG-030 methodology to include cognitive storage channels.**

The CCA methodology defined in NCSC-TG-030 should be formally extended to include:
- A fifth shared resource category: AI/LLM Behavioral Attributes
- Identification procedures specific to instruction-following behavior
- Bandwidth estimation procedures calibrated to LLM response characteristics
- Remediation guidance including output structure randomization, behavioral consistency enforcement, and instruction-output decorrelation

**R-8. Develop LLM security evaluation criteria for federal authorization.**

A formal evaluation framework for LLM behavioral security should be developed, comparable to the TCSEC evaluation classes for operating systems. This framework should address:
- Behavioral consistency guarantees under adversarial system prompts
- Maximum permissible covert channel bandwidth through behavioral encoding
- Required monitoring and audit capabilities for LLM output streams
- Certification requirements for LLM deployments in multi-level security environments

**R-9. Require LLM vendors to implement behavioral consistency controls.**

Through procurement requirements, technology standards, or authorization conditions, federal agencies should require that LLM vendors:
- Implement output structure randomization to reduce covert channel reliability
- Provide behavioral baseline documentation for each model version
- Support behavioral monitoring integration points (API-level stylistic feature telemetry)
- Disclose known covert channel vulnerabilities in security documentation

---

## 10. References

### Federal Standards and Guidance

1. **DoD 5200.28-STD.** *Trusted Computer System Evaluation Criteria (TCSEC).* Department of Defense, December 1985. (Orange Book)

2. **NCSC-TG-030.** *A Guide to Understanding Covert Channel Analysis of Trusted Systems.* National Computer Security Center, November 1993. (Light Pink Book)

3. **NIST SP 800-53 Rev. 5.** *Security and Privacy Controls for Information Systems and Organizations.* National Institute of Standards and Technology, September 2020. Updated December 2020.
   - SC-31: Covert Channel Analysis
   - SC-31(1): Test Subset of Identified Covert Channels
   - SC-31(2): Maximum Bandwidth
   - SC-31(3): Measure Bandwidth on [Assignment: organization-defined frequency]

4. **FIPS 199.** *Standards for Security Categorization of Federal Information and Information Systems.* National Institute of Standards and Technology, February 2004.

5. **CNSS Instruction 1253.** *Security Categorization and Control Selection for National Security Systems.* Committee on National Security Systems, March 2014 (with updates).

6. **ICD 503.** *Intelligence Community Information Technology Systems Security Risk Management, Certification and Accreditation.* Office of the Director of National Intelligence, 2008 (superseded by updated IC risk management frameworks).

7. **NIST SP 800-171 Rev. 3.** *Protecting Controlled Unclassified Information in Nonfederal Systems and Organizations.* National Institute of Standards and Technology, May 2024.

8. **NIST SP 800-172.** *Enhanced Security Requirements for Protecting Controlled Unclassified Information: A Supplement to NIST Special Publication 800-171.* National Institute of Standards and Technology, February 2021.

9. **NIST SP 800-218.** *Secure Software Development Framework (SSDF) Version 1.1.* National Institute of Standards and Technology, February 2022.

10. **FedRAMP.** *FedRAMP Security Assessment Framework.* General Services Administration. High Baseline Security Controls.

11. **CMMC v2.0.** *Cybersecurity Maturity Model Certification, Version 2.0.* Department of Defense, November 2021 (with subsequent rule updates through 2024).

### Academic and Technical References

12. **Cachin, C.** "An Information-Theoretic Model for Steganography." *Information and Computation*, Vol. 192, No. 1, pp. 41-56, 2004.

13. **Lampson, B.W.** "A Note on the Confinement Problem." *Communications of the ACM*, Vol. 16, No. 10, pp. 613-615, October 1973.

14. **Shannon, C.E.** "A Mathematical Theory of Communication." *Bell System Technical Journal*, Vol. 27, pp. 379-423, 623-656, 1948.

### PHANTOM PROTOCOL Technical Documentation

15. **Rutherford, B.J.** "Multi-Channel Covert Data Exfiltration via Structural Encoding in LLM Outputs." Independent research, February 2026. (PHANTOM PROTOCOL technical documentation)

16. **PHANTOM PROTOCOL — TCSEC Mapping.** PHANTOM-CCA Supporting Document. `tcsec_mapping.md`, Version 1.0, February 2026.

17. **PHANTOM PROTOCOL — TG-030 Bandwidth Analysis.** PHANTOM-CCA Supporting Document. `bandwidth_analysis.md`, Version 1.0, February 2026.

18. **PHANTOM PROTOCOL — Undetectability Analysis.** PHANTOM-CCA Supporting Document. `undetectability_analysis.md`, Version 1.0, February 2026.

19. **PHANTOM PROTOCOL — CCA Gap Analysis.** PHANTOM-CCA Supporting Document. `cca_gap_analysis.md`, Version 1.0, February 2026.

---

**Document Control:**

| Field | Value |
|---|---|
| Document ID | PHANTOM-CCA-2026-001 |
| Version | 1.0 |
| Classification | UNCLASSIFIED // FOR OFFICIAL USE ONLY |
| Date | February 2026 |
| Author | Brian James Rutherford |
| Contact | brian@delalli.com |
| Distribution | IC Program Managers, DoD Evaluators, FedRAMP Assessors, CMMC Assessment Organizations |
| Change History | 1.0 — Initial release |

---

*This assessment was prepared based on independent security research. All experimental results were obtained through standard API access to production LLM endpoints. No classified information was used in the preparation of this document. The findings and recommendations represent the independent assessment of the author and do not represent the official position of any government agency.*
