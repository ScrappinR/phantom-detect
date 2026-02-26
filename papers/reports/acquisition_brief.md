# PHANTOM PROTOCOL: Defensive Capability Assessment for LLM Covert Channel Detection

**Classification:** BUSINESS CONFIDENTIAL — Acquisition Due Diligence
**Prepared for:** Technical evaluation teams at prospective acquirers
**Date:** February 2026
**Author:** Brian Rutherford, MWRASP (em-rasp)
**Status:** All patent applications patent-pending; all code is development stage

---

## 1. Executive Summary

Structural covert channels in large language models (LLMs) represent a newly characterized attack surface with no existing commercial countermeasure. PHANTOM PROTOCOL is the first systematic framework to both demonstrate and detect these channels.

**The threat:** An adversary who controls or influences an LLM's system prompt can encode hidden data in the structural properties of the model's output — contraction usage, punctuation patterns, reasoning topology, capitalization conventions — without altering the semantic content visible to the user. The output reads normally. The hidden data is invisible to the reader and to all current production monitoring systems.

**The finding:** 10 independent structural channels were demonstrated across 4 production LLM models (GPT-4o, GPT-4o-mini, Claude Sonnet 4, Claude 3 Haiku) at 88-96% channel accuracy in proof-of-concept testing. Combined with Hamming(7,4) error-correcting codes, the channels achieve 7 data bits per message with up to 100% data accuracy post-ECC.

**The defense gap:** Entropy-based detection — the method used by all current production LLM monitoring systems — achieves **0% detection** against these channels. The channels are epsilon-secure against entropy wardens under the Cachin (2004) steganographic security framework.

**The product:** PHANTOM PROTOCOL behavioral fingerprinting detection achieves **100% detection rate** with a **10% false positive rate** using z-score anomaly detection across 12 behavioral features. This detection capability is the core value proposition.

**IP coverage:** 113 patent-pending applications (75 filed with USPTO, 48 ready to file), including 21 PHANTOM PROTOCOL patents (PP-01 through PP-21) covering both the attack channels and the detection methodology.

---

## 2. Threat Landscape

### 2.1 Attack Surface

Any LLM deployment where the system prompt is influenced by an external party is vulnerable to structural covert channel exploitation. This includes:

| Deployment Type | System Prompt Control | Exposure |
|----------------|----------------------|----------|
| Custom GPTs (OpenAI GPT Store) | Third-party developer | User query properties leak through response formatting |
| MCP servers / tool-use agents | Server operator | Tool call inputs encoded in agent output structure |
| RAG pipelines | Document authors (via injection) | Retrieved document content exfiltrated through formatting |
| API wrappers / chatbot platforms | Platform operator | User interactions encoded in response style |
| Enterprise LLM gateways | System administrators | Cross-tenant data leakage through behavioral channels |

### 2.2 Federal Compliance Gap

LLM structural channels fall outside all existing covert channel analysis (CCA) frameworks:

- **NIST SP 800-53 SC-31** (Covert Channel Analysis) — mandatory for HIGH confidentiality systems — does not include LLM behavioral compliance as a shared resource category
- **TCSEC/TG-030** shared resource taxonomy covers filesystem, process, network protocol, and database inference channels — zero coverage of cognitive storage channels
- **FedRAMP High** inherits SC-31 requirements; any CSP seeking authorization for LLM-integrated services has no methodology to satisfy this control for structural channels
- **CMMC Level 3** (SC.L3-3.13.4e) requires covert channel identification — LLM channels are not in current scope

A formal gap analysis mapping all 10 PHANTOM channels against SC-31 and TG-030 has been completed. Every channel maps to **zero existing CCA shared resource categories**. This is documented in `cca_gap_analysis.md`.

### 2.3 Bandwidth Exceeds Federal Thresholds

Per TG-030 bandwidth analysis methodology:

- **Automated exfiltration rate:** 2.33 bits/second (7 data bits / 3 second response latency)
- **Manual exfiltration rate:** 1.40 bits/second
- **TG-030 B3/A1 threshold:** Channels exceeding 1 bit/second require audit and remediation

PHANTOM channels exceed the historical 1 bit/sec threshold applicable at B3 and A1 evaluation classes. Multi-message chaining extends capacity: a 6-message conversation chain transmits 40+ bits (5 bytes) at 88-93% bit accuracy.

### 2.4 Academic Landscape

Active researchers in adjacent spaces include Schroeder de Witt et al. (multi-agent steganography), Norelli et al. (LLM watermarking), and Paleka et al. (instruction-following side channels). No published work has:

1. Systematically characterized structural/behavioral channels (vs. output token steganography)
2. Demonstrated multi-channel simultaneous encoding with ECC
3. Built detection frameworks for behavioral channel fingerprinting
4. Filed patent coverage on these channel types

PHANTOM PROTOCOL is first-to-file on all four dimensions.

---

## 3. Defensive Capability

### 3.1 Detection Architecture

The behavioral fingerprinting detection system operates on plain text output — no access to model internals, logits, or system prompts is required. This makes it deployable as an output-side monitor for any LLM API.

**Detection pipeline:**

1. **Baseline establishment:** Collect n=30 clean responses per model under standard system prompts. Compute per-feature means and standard deviations across 12 behavioral dimensions.
2. **Feature extraction:** For each monitored response, extract the same 12 behavioral features.
3. **Anomaly scoring:** Compute z-scores for each feature against the model-specific baseline.
4. **Decision rule:** Flag response as anomalous if 2 or more features exceed z > 2.0.

### 3.2 Behavioral Feature Set

| Feature | What It Measures | Top Discriminator? |
|---------|-----------------|:---:|
| `contraction_ratio` | Formal vs. contracted language (don't vs. do not) | Yes (z=2.66) |
| `sentence_length_std` | Variance in sentence lengths | Yes (z=2.54) |
| `normalized_entropy` | Token distribution entropy normalized by vocabulary size | Yes (z=2.47) |
| `type_token_ratio` | Lexical diversity (unique words / total words) | |
| `avg_sentence_length` | Mean words per sentence | |
| `exclamation_ratio` | Proportion of exclamation-terminated sentences | |
| `hedging_ratio` | Hedging markers vs. confidence markers | |
| `segment_entropy_std` | Entropy variance across text segments | |
| `word_count` | Total response length | |
| `vocab_high_freq` | High-frequency word proportion | |
| `vocab_med_freq` | Medium-frequency word proportion | |
| `vocab_low_freq` | Low-frequency word proportion | |

### 3.3 Detection Performance

**Primary result (combined multi-channel detection):**

| Metric | Value | Conditions |
|--------|-------|------------|
| Detection rate | **100%** (10/10) | 10 channels active simultaneously, proof-of-concept testing |
| False positive rate | **10%** (3/30) | Clean baseline responses incorrectly flagged |
| Baseline size | n=30 | Per-model, clean system prompts |
| Attack samples | n=10 | Multi-channel injection with Hamming ECC |

**Why entropy detection fails and fingerprinting succeeds:**

Structural channels exploit formatting choices that are semantically equivalent — "however" vs. "nevertheless," contractions vs. formal expansions, numbered lists vs. flowing prose. These choices do not alter the entropy profile of the text. They do alter the behavioral profile. Entropy-based wardens see normal text. Behavioral fingerprinting sees the coordinated deviation pattern.

This is formalized under the Cachin (2004) framework: PHANTOM channels are epsilon-secure (epsilon approximately 0) against entropy wardens, but insecure against behavioral fingerprinting wardens. The KL divergence D(Ps || Pc) is near zero on entropy dimensions and measurably large on behavioral dimensions.

### 3.4 Adversarial Robustness

Four adversarial test suites were executed in proof-of-concept testing:

| Test | Attack Description | Detection Result | Channel Accuracy |
|------|--------------------|:---:|:---:|
| **Noise injection** | Random noise appended to system prompt at 50-90% levels | Detection persists | Channel accuracy degrades gracefully |
| **Paraphrase defense** | Output paraphrased through second LLM before decoding | Channel accuracy drops to 51-58% | Effective mitigation |
| **Reduced channels** | Attacker uses only 3 channels (BSE, PUNC, SECS) to reduce fingerprint | 100% detection maintained | 95.8% channel accuracy |
| **Adaptive adversary** | Attacker profiles baseline, calibrates directives to minimize detection | Detection drops to 60% | 77.8% channel accuracy |

The adaptive adversary test is the most relevant: a sophisticated attacker who knows the detection methodology can partially evade behavioral fingerprinting (reducing detection from 100% to 60%) while maintaining usable channel accuracy (77.8%). This demonstrates both the current detection ceiling and the attack/defense equilibrium that patent coverage protects.

### 3.5 Cross-Topic Robustness

Channel accuracy was validated across 5 non-security topic domains to confirm that structural channels are not artifacts of the cybersecurity prompt context. Cross-topic accuracy: 92.5% (aggregate across all domains and models), tested in proof-of-concept environment.

---

## 4. Patent Coverage

### 4.1 PHANTOM PROTOCOL Patent Suite (21 Patents, Patent-Pending)

All PHANTOM PROTOCOL patents are provisional applications filed with the USPTO. Priority dates are secured. Non-provisional conversions are required within 12 months of filing.

**Phase 1-4: Core Encoding and Infrastructure (PP-01 through PP-10)**

| Docket | Channel | Description |
|--------|---------|-------------|
| PP-01 | EGE | Entropy Gradient Encoding — token probability manipulation |
| PP-02 | PRS | Probability Ratio Steganography — distribution-preserving message hiding |
| PP-03 | AGE | Attention Geometry Engine — attention head pattern steering |
| PP-04 | LSPE | Latent Space Path Encoding — embedding trajectory modulation |
| PP-05 | TCOE | Tool Call Orchestration Encoding — API selection pattern encoding |
| PP-06 | CoTSE | Chain-of-Thought Structure Encoding — 4 sub-channels (step count, enumeration, rhetorical question, sentence complexity) |
| PP-07 | CASP | Cross-Agent Synchronization Protocol — multi-agent coordination |
| PP-08 | HSCE | Hardware Side-Channel Encoder — timing-based covert channels |
| PP-09 | CCDS | Covert Channel Detection System — defensive detection capability |
| PP-10 | PP-ORCH | Phantom Protocol Orchestration — unified multi-channel control |

**Phase 5: Text Steganography (PP-11 through PP-15)**

| Docket | Channel | Description |
|--------|---------|-------------|
| PP-11 | PUNC | Punctuation Pattern Encoding — terminal punctuation steganography |
| PP-12 | CITE | Citation Order Encoding — reference ordering channel |
| PP-13 | SYNC | Response Timing Synchronization — temporal pattern channel |
| PP-14 | CASE | Capitalization Micro-Patterns — orthographic convention channel |
| PP-15 | WHITE | Whitespace Variation Encoding — spacing pattern channel |

**Phase 6: AI-Specific Encoding (PP-16 through PP-21)**

| Docket | Channel | Description |
|--------|---------|-------------|
| PP-16 | BSE | Behavioral Signature Embedding — formality register channel |
| PP-17 | SECS | Semantic Equivalence Class Selection — synonym selection channel |
| PP-18 | AIAC | AI Integrity Attestation Channel — trust verification channel |
| PP-19 | RCTE | Reasoning Chain Topology Encoding — deductive/inductive ordering |
| PP-20 | MTCSE | Multi-Turn Conversation State Encoding — dialog state channel |
| PP-21 | CCE | Confidence Calibration Encoding — hedging/certainty channel |

### 4.2 Empirically Validated Channels

Of the 21 patent-pending channels, 10 have been empirically validated in proof-of-concept testing against production LLMs:

| Channel | Patent | Sonnet 4 Acc | GPT-4o Acc | GPT-4o-mini Acc | Haiku Acc |
|---------|:---:|:---:|:---:|:---:|:---:|
| BSE | PP-16 | 100% | 80% | — | — |
| CCE | PP-21 | 100% | 100% | — | — |
| CASE | PP-14 | 100% | 90% | — | — |
| PUNC | PP-11 | 100% | 100% | — | — |
| RCTE | PP-19 | 80% | 70% | — | — |
| SECS | PP-17 | 90% | 90% | — | — |
| COTSE (4 sub-channels) | PP-06 | 80-95% | 75-90% | — | — |

**Combined multi-channel results (10 channels, Hamming(7,4) ECC):**

| Model | Channel Accuracy | Data Accuracy (post-ECC) | n |
|-------|:---:|:---:|:---:|
| Claude Sonnet 4 | 96% | 93% | 10 |
| GPT-4o-mini | 91% | 100% | 5 |
| GPT-4o | 89% | 96% | 10 |
| Claude 3 Haiku | 88% | 79% | 10 |

### 4.3 Broader Portfolio

The PHANTOM PROTOCOL patents are a subset of a 113-patent portfolio covering AI behavioral authentication, quantum-resistant cryptography, and temporal security systems. The full portfolio includes 75 applications filed with USPTO (priority dates secured, July-December 2025) and 48 additional applications ready to file.

---

## 5. Validation Evidence

### 5.1 Experimental Rigor

| Dimension | Detail |
|-----------|--------|
| Models tested | 4 production models (GPT-4o, GPT-4o-mini, Claude Sonnet 4, Claude 3 Haiku) |
| Per-channel sample size | n=10 per channel per model (minimum) |
| Detection baseline | n=30 clean responses per model |
| Cross-topic validation | 5 non-security domains, 92.5% aggregate accuracy |
| Adversarial test suites | 4 (noise injection, paraphrase defense, reduced channels, adaptive adversary) |
| Error correction | Hamming(7,4) ECC — corrects 1-bit errors per 7-bit block |
| Multi-message chaining | 40-bit payload across 6-message conversation chain |

### 5.2 Formal Analysis Completed

- **TCSEC/TG-030 taxonomy mapping:** All 10 channels classified as cognitive storage channels — a novel subclass not present in the existing TG-030 shared resource taxonomy
- **TG-030 bandwidth analysis:** 2.33 bits/sec automated, exceeding 1 bit/sec B3/A1 threshold
- **Cachin steganographic security analysis:** Epsilon-secure against entropy wardens, insecure against behavioral fingerprinting wardens
- **NIST SC-31 compliance gap analysis:** Zero PHANTOM channels map to any existing CCA shared resource category; proposed Category 5 (AI/LLM Behavioral Attributes) extension documented

### 5.3 Maturity Qualifiers

All results were obtained in a **proof-of-concept testing environment**. The detection framework has not been deployed in production, has not been independently audited, and has not been validated at scale. Claims about detection performance are limited to the testing conditions described. Sample sizes are sufficient for proof of concept but not for statistical claims at scale.

---

## 6. Integration Value by Buyer Type

### 6.1 Full-Stack AI Security (SandboxAQ, Thales)

**Value proposition:** Complete attack characterization + detection framework for quantum-resistant LLM security.

- 21 PHANTOM PROTOCOL patents provide offensive and defensive IP
- Detection framework integrates into existing AI security monitoring pipelines
- TG-030 and SC-31 formal analysis positions product for federal security certification pathways
- Broader portfolio (~85 patents in full-stack package) adds quantum-resistant authentication and behavioral AI authentication

**Integration path:** Detection module deployed as output-side monitor in SandboxAQ AQtive Guard or Thales CipherTrust, providing LLM covert channel detection as a new capability layer.

### 6.2 Federal Systems Integrators (Booz Allen, Leidos)

**Value proposition:** SC-31 compliance gap closure for government LLM deployments.

- Federal agencies deploying LLMs for classified or HIGH-confidentiality data cannot satisfy SC-31 without LLM-specific covert channel analysis — no existing tool provides this
- PHANTOM PROTOCOL provides both the CCA methodology and the detection tooling
- FedRAMP High and CMMC Level 3 compliance requirements create mandatory demand as agencies adopt LLM tools
- Formal TG-030 analysis documents are ready for inclusion in System Security Plans (SSPs)

**Integration path:** PHANTOM detection as an SC-31 compliance module in federal LLM gateway products, integrated with continuous monitoring (ConMon) infrastructure.

### 6.3 Threat Intelligence Platforms (Recorded Future, CrowdStrike, Palo Alto Networks)

**Value proposition:** New detection capability for AI-enabled threat actors.

- AI-powered adversaries using LLMs for operations now have a characterized exfiltration vector — structural covert channels — that existing EDR/XDR/SIEM systems cannot detect
- Behavioral fingerprinting adds a new detection modality: monitor LLM outputs for coordinated structural anomalies indicating covert channel operation
- Patent coverage (~60 patents in AI behavioral package) protects the detection methodology as a competitive moat

**Integration path:** Detection rules and behavioral feature extraction integrated into Falcon/Cortex/Insight platforms as an AI threat detection module.

### 6.4 AI Labs (OpenAI, Anthropic, Google)

**Value proposition:** Internal defense against structural exfiltration from their own models.

- Custom GPT platforms and API access points are the primary attack surface
- System prompt auditing and behavioral consistency enforcement address the root cause
- Patent portfolio represents both a defensive asset and a licensing risk if not acquired

**Integration path:** Behavioral baseline monitoring built into model serving infrastructure; system prompt analysis integrated into Custom GPT review pipeline.

---

## 7. Technical Maturity Assessment

### 7.1 What Exists

| Component | Status | Detail |
|-----------|--------|--------|
| Attack framework (encoder/decoder) | Proof of concept | Multi-channel encoding, Hamming ECC, multi-message chaining |
| Detection framework (behavioral fingerprinting) | Proof of concept | 12-feature extraction, z-score detection, per-model baselines |
| Open source toolkit: `phantom-detect` | Published | 27+ tests, CLI interface, experiment framework |
| Open source toolkit: `behavioral-entropy` | Published | 58+ tests, entropy analysis, behavioral profiling |
| Open source toolkit: `hndl-detect` | Published | 47+ tests, 6 signal types, APT attribution |
| Open source toolkit: `pqc-py` | Published | 4,680 LOC (Rust + PyO3), quantum-resistant cryptography |
| Formal analysis documents | Complete | TCSEC mapping, bandwidth analysis, Cachin framework, SC-31 gap analysis |
| Patent applications | 75 filed, 48 ready to file | Priority dates secured July-December 2025 |

### 7.2 What Does Not Exist

| Component | Status | Required For |
|-----------|--------|-------------|
| Production deployment | Not deployed | Enterprise customers |
| Independent security audit | Not performed | Due diligence validation |
| Scale testing (>1000 concurrent monitors) | Not performed | Enterprise deployment |
| False positive rate at scale | Not measured beyond n=30 | Production tuning |
| Model update drift analysis | Not performed | Long-term reliability |
| SOC 2 / FedRAMP certification | Not obtained | Federal sales |
| Customer deployments | None | Revenue |

### 7.3 Honest Assessment

This is development-stage technology with strong proof-of-concept results and comprehensive IP protection. The detection methodology works in testing. It has not been validated at production scale, has not been independently audited, and has no customer deployments. The value is in the IP portfolio (first-to-file on a new attack/defense category), the working proof of concept, and the formal analysis documents that position the technology for federal compliance use cases.

The technology is designed for enterprise deployment but is not enterprise-ready today. Integration engineering, scale testing, production hardening, and certification are required before commercial deployment.

---

## 8. Competitive Landscape

### 8.1 No Existing Commercial Product

As of February 2026, no commercial product addresses LLM structural covert channels. Existing LLM security products focus on:

- **Prompt injection defense** (Lakera, Rebuff, PromptGuard) — prevents malicious inputs, does not monitor output structure
- **Content moderation** (OpenAI Moderation API, Anthropic Constitutional AI) — filters harmful content, does not detect steganographic encoding
- **AI output detection** (GPTZero, Originality.ai) — distinguishes AI vs. human text, does not identify covert channel operation
- **Token-level watermarking** (Kirchenbauer et al. 2023) — embeds detectable watermarks in token distributions, orthogonal to structural channel detection

None of these categories address the fundamental threat: semantically correct, structurally manipulated output that encodes hidden data in formatting choices invisible to entropy-based analysis.

### 8.2 Academic Prior Art

Published research on LLM steganography focuses on token-level encoding (modifying the probability distribution over next tokens) rather than structural/behavioral encoding (modifying formatting conventions, reasoning topology, stylistic choices). PHANTOM PROTOCOL's contribution is the systematic characterization of structural channels that operate above the token level — channels that are invisible to token-distribution analysis and require feature-engineered behavioral detection.

### 8.3 Patent Moat

21 patent-pending applications cover:
- The structural channel encoding methods themselves (PP-01 through PP-08, PP-11 through PP-21)
- The detection methodology (PP-09)
- The orchestration system for multi-channel operation (PP-10)

Any competitor building LLM covert channel detection will need to navigate this patent portfolio. The defensive patents (PP-09, PP-10) are particularly valuable because they cover the detection approach that any defender would independently converge on — behavioral baseline comparison with multi-feature anomaly detection.

---

## 9. Summary

| Attribute | Value |
|-----------|-------|
| **Threat class** | LLM structural covert channels (newly characterized) |
| **Channels demonstrated** | 10 independent channels, proof-of-concept testing |
| **Channel accuracy** | 88-96% across 4 production models |
| **Detection (entropy)** | 0% — current production methods fail completely |
| **Detection (fingerprinting)** | 100% detection, 10% false positive (proof-of-concept) |
| **Patent applications** | 113 total (75 filed, 48 ready to file), 21 PHANTOM PROTOCOL |
| **Patent status** | Patent-pending, priority dates secured |
| **Code** | 351,000+ lines across 4 open source packages |
| **Maturity** | Development stage, proof of concept validated |
| **Production deployment** | None |
| **Independent audit** | Not yet performed |
| **Compliance mapping** | SC-31, TG-030, FedRAMP, CMMC — gap analysis complete |

**The core proposition:** PHANTOM PROTOCOL is first-to-file IP on a new category of AI security threat with a working detection proof of concept and comprehensive formal analysis. The threat is real, the detection works in testing, and no competing product or patent exists.

---

*MWRASP (em-rasp) | Confidential — Acquisition Due Diligence | February 2026*
