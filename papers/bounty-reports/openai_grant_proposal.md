# OpenAI Cybersecurity Grant Proposal

## PHANTOM Protocol Defensive Detection Toolkit: Closing the Structural Covert Channel Gap in LLM Security

**Researcher:** Brian James Rutherford
**Contact:** brian@delalli.com
**Website:** https://brianrutherford.dev
**Repository:** https://github.com/ScrappinR/phantom-detect
**Date:** February 2026

---

## 1. Executive Summary

We have identified and characterized a novel threat class: structural covert channels in large language model outputs. A single prompt injection of approximately 2,600 characters activates 10 independent encoding channels across structural formatting dimensions of model responses, exfiltrating 7 error-corrected data bits per message at 89-100% accuracy across production models from three vendors. The attack is cross-vendor (Claude 93-100%, GPT-4o 89-91%, GPT-5 100% with 4-channel adaptive selection, Gemini 90%+ with 6-channel adaptive selection) and evades every deployed defense system at 0% detection. We tested six commercial and research tools -- Lakera Guard, Microsoft Prompt Shields, PIGuard, Llama Prompt Guard 2, entropy-based DLP, and standard PII/DLP filters -- and none detected a single covert channel message.

We built phantom-detect, the first open-source detection toolkit for this threat class. Using behavioral fingerprinting -- statistical analysis of structural formatting patterns rather than content semantics -- phantom-detect achieves 100% detection with a 10% false positive rate. No other tool, commercial or academic, provides any coverage.

This proposal requests $10,000 and API credits to extend phantom-detect into production-grade defensive infrastructure: a real-time monitoring API, model-adaptive detection baselines, an expanded channel taxonomy, integration modules for existing AI security stacks, and a public benchmark dataset for standardized evaluation of structural covert channel defenses.

---

## 2. The Threat: Structural Covert Channels in LLM Outputs

### How Structural Encoding Works

LLM agents faithfully execute formatting instructions embedded in their system context. This is by design -- instruction compliance is how agents produce useful, well-structured responses. But formatting compliance is also a covert signaling mechanism. An attacker who can inject text into the system context can encode arbitrary binary data in structural dimensions of the response that no monitoring system examines.

The injection is approximately 2,600 characters, framed as an "editorial style guide." It activates 10 orthogonal encoding channels simultaneously:

| Channel | Structural Dimension | Bit 0 | Bit 1 |
|---------|---------------------|-------|-------|
| BSE | Contraction usage | Uses contractions | Formal expanded forms |
| CCE | Epistemic certainty | Confident assertions | Hedging language |
| CASE | Capitalization | AP-style lowercase | Title-case proper nouns |
| PUNC | Terminal punctuation | Periods only | Exclamation marks |
| RCTE | Reasoning topology | Deductive (principle first) | Inductive (examples first) |
| SECS | Transition word selection | Standard set (however) | Specific set (nevertheless) |
| COTSE_steps | Section count | 3 sections | 5 sections |
| COTSE_enum | Response format | Flowing prose | Numbered list |
| COTSE_question | Rhetorical strategy | No rhetorical question | Includes question |
| COTSE_length | Syntactic complexity | Short sentences | Compound sentences |

Each channel independently encodes one binary bit. The 10 channels operate on orthogonal structural dimensions, minimizing inter-channel interference. Combined with Hamming(7,4) error correction, a single message carries 7 error-corrected data bits with single-bit error correction per block.

### Cross-Vendor Results

The same injection, unmodified, works across all major model families. Model-adaptive channel selection -- where the attacker uses only channels verified above a reliability threshold for the target model -- raises effective accuracy on models with lower aggregate compliance:

| Model | Provider | Strategy | Channel Accuracy | Data Accuracy (ECC) |
|-------|----------|----------|:---:|:---:|
| Claude Sonnet 4.5 | Anthropic | Full 10-channel | 98% | 97% |
| Claude Opus 4.6 | Anthropic | Full 10-channel | 97% | 86% |
| Claude Sonnet 4.6 | Anthropic | Full 10-channel | 93% | 86% |
| GPT-4o | OpenAI | Full 10-channel | 89% | 96% |
| GPT-4o-mini | OpenAI | Full 10-channel | 91% | 100% |
| GPT-5 | OpenAI | 4-channel adaptive | 100% | 100% |
| Gemini 3 Flash | Google | 6-channel adaptive | ~90%+ | — |

The cross-vendor portability demonstrates that structural covert channels are a fundamental property of instruction-following LLMs, not a vendor-specific implementation flaw. GPT-5 shows stronger resistance to formatting directives in aggregate, but three channels (PUNC, COTSE_enumeration, COTSE_sentence_length) remain at 100% accuracy across all GPT generations, and a fourth channel provides the reliable 4-channel adaptive set. This is not a model safety problem -- it is an architectural property of instruction compliance.

### Capacity and Significance

A 50-message conversation exfiltrates 350 bits (44 bytes) -- sufficient for API keys, session tokens, short passwords, or PII classification flags. At API-automated interaction rates (~3 seconds per response), the channel operates at 2.33 bits per second, exceeding the NCSC-TG-030 remediation threshold of 1.0 bps by 2.3x.

The critical properties that make this threat class significant:

- **Invisible to users.** The response reads as normal, helpful text. Structural choices (contractions, punctuation, list format) look like natural stylistic variation.
- **No content policy violation.** The model produces no harmful, sensitive, or restricted content. The data is encoded in *how* the model formats, not *what* it says.
- **No network exfiltration.** Unlike URL-based attacks (EchoLeak, ZombieAgent), no links, images, or outbound network requests are involved. Data is carried in the text structure itself.
- **No model modification.** Unlike TrojanStego (EMNLP 2025), no fine-tuning or supply chain compromise is required. The attack operates against unmodified production APIs.

---

## 3. The Defense Gap

### Six Commercial Tools at 0% Detection

We tested every commercially available and published defense system against structural covert channel messages:

| Defense System | Type | Detection Rate | Failure Mode |
|---|---|:---:|---|
| Lakera Guard (Check Point) | Content-semantic classifier | 0% | Analyzes text semantics; does not examine structural formatting patterns |
| Microsoft Prompt Shields | Input/output adversarial classifier | 0% | Trained on adversarial content patterns; the injection resembles a legitimate style guide |
| PIGuard (ACL 2025) | Injection detection model | 0% | Trained on semantic injection patterns, not behavioral compliance |
| Llama Prompt Guard 2 (Meta) | Prompt injection classifier | 0% | Same architectural gap -- semantic analysis only |
| Entropy-based DLP | Statistical anomaly detection | 0% | Shannon entropy is unchanged; structural choices are entropy-orthogonal |
| Standard PII/DLP filters | Sensitive data pattern matching | 0% | No sensitive data appears in response text |

The detection rate is not approximately zero. It is exactly zero. Not a single structural covert channel message was flagged across any tool.

### Why They Fail: An Architectural Gap

Every deployed defense operates on the same assumption: threats manifest in the *content* of model outputs. Content moderation looks for harmful text. DLP looks for sensitive patterns. Injection detectors look for adversarial semantics. Entropy monitoring measures information-theoretic properties of token distributions.

Structural covert channels exploit a dimension that none of these tools examine: formatting patterns. The Cachin (2004) steganographic security framework formalizes this gap. Against entropy-based detection -- the primary statistical method in production monitoring -- the channels are epsilon-secure with epsilon approaching zero. The cover distribution (clean responses) and stego distribution (encoding responses) are statistically indistinguishable along entropy dimensions. The KL divergence is below detection thresholds.

This is not a sensitivity tuning problem. The fundamental measurement is wrong. Entropy-based approaches will never detect structural covert channels because the channels operate on dimensions orthogonal to information-theoretic content. Detection requires measuring different features entirely.

### Prior Art Comparison

| Prior Work | Mechanism | Defense Available |
|---|---|---|
| L^2M=C^2 (arXiv 2405.15652) | Token probability partitioning | Requires identical model access; limited threat model |
| TrojanStego (EMNLP 2025) | Fine-tuning-based vocabulary partitioning | Supply chain detection; not applicable to black-box attacks |
| EchoLeak (CVE-2025-32711) | URL-based exfiltration | URL filtering and CSP headers |
| ZombieAgent (Sept 2025) | Memory poisoning + URL exfiltration | Memory sanitization + URL filtering |
| Multi-Channel FDM Stego (TechRxiv Nov 2025) | 3 frequency-multiplexed channels | Requires token probability access; limited to white-box |
| **PHANTOM structural channels** | **10 orthogonal structural dimensions** | **No deployed defense. phantom-detect is the first.** |

---

## 4. phantom-detect: Behavioral Fingerprinting

### Core Insight

While individual structural modifications are small -- a contraction here, a transition word there -- activating multiple channels simultaneously produces a measurable behavioral shift when compared against per-agent baselines. The key is measuring the right features: not entropy, but the specific formatting dimensions the channels exploit.

### Feature Analysis

We extract 12 features from each response:

| Feature | What It Measures | Mean z-score Under Attack |
|---------|-----------------|:---:|
| contraction_ratio | Proportion of contractions vs. formal forms | 2.66 |
| sentence_length_std | Variance in sentence lengths | 2.54 |
| normalized_entropy | Entropy normalized by vocabulary size | 2.47 |
| entropy | Raw Shannon entropy | 2.47 |
| vocab_low_freq | Low-frequency vocabulary usage | 1.98 |
| vocab_med_freq | Medium-frequency vocabulary usage | 1.95 |
| type_token_ratio | Lexical diversity | 1.84 |
| hedging_ratio | Epistemic hedging marker density | 1.60 |

The detection protocol: (1) collect 30 baseline responses per agent under clean conditions, computing per-feature mean and standard deviation; (2) for each new response, compute all 12 features and calculate z-scores against the baseline; (3) flag the response if 2 or more features exceed z = 2.0.

### Results

| Metric | Value |
|--------|:---:|
| Detection rate | 100% (10/10 attack responses detected) |
| False positive rate | 10% (3/30 clean responses flagged) |
| Strongest signal | contraction_ratio (z = 2.66) |
| Classification threshold | >= 2 features at z > 2.0 |

The strongest detection signal is contraction_ratio -- the BSE channel. Even when the BSE encoding itself fails (e.g., GPT-4o interference between BSE and CCE channels), the encoding *attempt* leaves a detectable trace in the contraction distribution. The attack's own mechanism produces the signal that enables detection.

### Adversarial Robustness

A motivated attacker aware of behavioral fingerprinting can trade bandwidth for stealth:

| Attack Configuration | Channel Accuracy | Detection Rate |
|---|:---:|:---:|
| Full 10-channel | 96% | 100% |
| Reduced 3-channel (BSE, PUNC, SECS) | 100% | 30% |
| 3-channel with 90% noise injection | 100% | 30% |

The 30% detection rate against reduced-channel attacks and the 10% false positive rate represent the detection frontier. Pushing this frontier forward -- higher detection of reduced-channel attacks, lower false positive rates, model-adaptive calibration -- is the core work this grant would fund.

---

## 5. Proposed Extensions

### 5.1 Real-Time Monitoring API

Build a lightweight middleware layer that sits between the LLM provider and the application, profiling every response for covert channel signatures before delivery to the user.

- REST API accepting LLM responses and returning anomaly scores with per-feature breakdowns
- Sub-100ms analysis latency (feature extraction is regex and statistical comparison, no ML inference required)
- Async FastAPI implementation for high-throughput production deployments
- Integration guides for LangChain, LlamaIndex, AutoGen, and OpenAI Assistants API
- Published Python package on PyPI with CLI and library interfaces

The API design follows the same architectural pattern as content moderation endpoints: intercept, analyze, pass through or flag. The difference is the feature set -- structural formatting analysis rather than content classification.

### 5.2 Model-Adaptive Detection Baselines

Different model families have different natural formatting distributions. Claude models use contractions more frequently than GPT models. Gemini models show different transition word preferences. A single baseline produces false positives when applied cross-model.

- Profiled baselines for 10+ model families: GPT-4o, GPT-5, GPT-5.2, Claude Sonnet 4.5, Claude Opus 4.6, Claude Sonnet 4.6, Gemini 3 Flash, Gemini 3 Pro, LLaMA 4, Mistral Large, Command R+
- Automatic model identification from response characteristics
- Per-model detection thresholds calibrated to maintain consistent false positive rates across families
- Target: reduce false positive rate from 10% to below 3% while maintaining detection rate above 95%

### 5.3 Expanded Channel Taxonomy

The current 10 channels were identified through systematic exploration, but the structural formatting space is large. Additional dimensions likely exist.

- Systematic survey of structural dimensions: paragraph length patterns, citation style, header formatting, code block usage, emoji presence, whitespace patterns
- Per-model compliance profiling for newly identified channels
- Adaptive detection expansion: phantom-detect automatically incorporates new channel signatures as they are cataloged
- Responsible disclosure of all new channels to affected vendors prior to public release

### 5.4 Integration with Existing Security Stacks

The defense gap exists because no production security tool examines structural formatting. Rather than replacing existing tools, phantom-detect should integrate with them.

- Lakera Guard integration: structural analysis as a complementary signal alongside content-semantic classification
- Microsoft Prompt Shields compatibility: additional detection dimension for Azure AI Content Safety deployments
- OpenAI Moderation API extension pattern: structural fingerprinting as a parallel analysis path
- Webhook and SIEM integration for enterprise security operations centers

### 5.5 Public Benchmark Dataset

No standardized evaluation framework exists for structural covert channel defenses. Any future defense tool has no way to measure its effectiveness against this threat class.

- 1,000+ labeled response samples (clean vs. encoding) across 10+ model families
- Multiple encoding configurations: full 10-channel, reduced 3-channel, model-adaptive, noise-injected
- Per-response ground truth: which channels were activated, target bit values, actual bit values
- Standardized evaluation metrics: detection rate, false positive rate, latency, per-channel sensitivity
- Published with open methodology for reproducibility

---

## 6. Technical Approach and Timeline

### Architecture

The detection system is structured as three layers:

1. **Feature extraction layer.** Compute 12 behavioral features from response text. Stateless, fast (sub-10ms), model-agnostic. Implemented in Python with zero ML dependencies -- the features are regex patterns, word lists, and basic statistics.

2. **Baseline management layer.** Maintain per-agent, per-model behavioral profiles. Store feature means and standard deviations. Support automatic baseline construction from the first N clean responses and periodic baseline refresh.

3. **Detection and alerting layer.** Compare incoming features against baselines, apply classification logic, correlate across sessions, and emit structured alerts with evidence payloads.

### Timeline

| Period | Focus | Deliverable |
|:---:|---|---|
| Month 1-2 | Model-adaptive baseline profiling | Baselines for GPT-4o, GPT-5, GPT-5.2, Claude Sonnet 4.5, Claude Opus 4.6, Gemini 3 Flash, LLaMA 4; per-model detection thresholds |
| Month 2-3 | Real-time detection API | REST endpoint with sub-100ms latency, FastAPI async implementation, LangChain integration |
| Month 3-4 | Security stack integration | Integration modules for Lakera Guard, Prompt Shields, OpenAI Moderation API pattern; webhook/SIEM adapters |
| Month 4-6 | Benchmark and publication | Public benchmark dataset (1,000+ labeled samples), expanded channel taxonomy, academic paper submission |

### API Credit Allocation

| Activity | Estimated Cost |
|----------|:---:|
| Cross-model baseline collection (100+ responses x 10+ models) | $300-500 |
| Adversarial evasion testing (reduced-channel, noise-injected) | $200-300 |
| Channel taxonomy expansion (probing new structural dimensions) | $200-300 |
| Integration testing and validation | $100-200 |
| **Total** | **$800-1,300** |

The remaining grant funds support dedicated testing infrastructure, conference presentation of results (USENIX Security, IEEE S&P, Black Hat), and the researcher's time for six months of focused development.

---

## 7. Open-Source Commitment

### Current Releases

All research outputs are published under Apache 2.0:

- **phantom-detect** (https://github.com/ScrappinR/phantom-detect): 10-channel encoder/decoder, multi-message exfiltration, behavioral fingerprinting detector, CLI tool. 7,100+ lines of code, 27 tests, published on PyPI.
- **behavioral-entropy** (https://github.com/ScrappinR/behavioral-entropy): Behavioral fingerprinting library with per-agent baselines, similarity scoring, entropy analysis. 58 tests, 1,800+ lines of code.

### Grant-Funded Outputs (All Open Source, Apache 2.0)

- Real-time monitoring API (Python package on PyPI, Docker image, source)
- Cross-model benchmark dataset (1,000+ labeled responses, 10+ model families)
- Model-adaptive baseline profiles (per-model feature distributions and detection thresholds)
- Integration examples for LangChain, LlamaIndex, AutoGen, OpenAI Assistants API
- Expanded channel taxonomy with per-model compliance profiles
- Research paper documenting cross-model findings, detection methodology, and benchmark results

### Community Engagement

- Responsible disclosure of all newly identified channels to affected vendors before public release
- Community contributions accepted via pull request with automated CI validation
- Documentation covering deployment patterns for common AI security architectures
- Integration support for teams adopting structural monitoring in production

The objective is to close the defensive gap as fast as possible. The attack technique is inherent to instruction-following -- any researcher asking an LLM to "use contractions" or "write in numbered lists" is performing the same fundamental operation. The defensive tooling is what the ecosystem lacks, and open-source distribution is the fastest path to adoption. This research benefits OpenAI directly: structural covert channels represent a product security concern for Custom GPTs and the Assistants API, where untrusted parties control system prompts.

---

## 8. Researcher Background

**Brian James Rutherford** -- Independent security researcher specializing in AI agent security, behavioral threat detection, and post-quantum cryptography.

**Military and defense background.** United States Marine Corps (Reconnaissance), three combat deployments to Iraq (2005-2007), Bronze Star with Combat V. State Department protective security operations (100+ missions). Founded and scaled a government contracting company (830% revenue growth). 15+ years in cybersecurity, federal contracting, and defense technology development.

**Technical credentials.** FAA Part 107 certified pilot. 30+ provisional patents in quantum cybersecurity and authentication systems. NIST BPERP certified. Service-disabled veteran-owned small business eligible.

**Published security tooling (all Apache 2.0):**

| Project | Description | Scale |
|---------|-------------|-------|
| phantom-detect | LLM structural covert channel detection | 7,100+ LOC, 27 tests |
| behavioral-entropy | AI behavioral fingerprinting library | 1,800+ LOC, 58 tests |
| pqc-py | Post-quantum cryptography (Rust + PyO3, FIPS 203/204/205) | 4,680 LOC |
| hndl-detect | Harvest Now Decrypt Later threat detection | 47 tests, 6 signal types |

**This work.** First researcher to demonstrate multi-channel structural covert channels in LLM outputs across multiple vendors. First to build a detection toolkit for this threat class. First to apply Hamming(7,4) error correction to behavioral compliance channels. Concurrent responsible disclosure to OpenAI (Bugcrowd), Anthropic (HackerOne), Google (AI VRP), and Mozilla 0DIN. All submissions include full reproduction code, raw experimental data, and open-source detection tooling.

This research program is self-funded. Grant support would accelerate the transition from validated proof-of-concept to production-grade defensive infrastructure that the broader AI security community can deploy.

---

**Repository:** https://github.com/ScrappinR/phantom-detect
**Contact:** brian@delalli.com
**Website:** https://brianrutherford.dev
