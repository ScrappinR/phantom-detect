# Structural Covert Channels in Large Language Model Inference: Methodology and Results

**Conference Supplement / Extended Abstract**
*PHANTOM PROTOCOL Research Group*
*February 2026*

---

## 1. Introduction and Motivation

Large language models (LLMs) deployed via API follow formatting and stylistic instructions embedded in system prompts with high fidelity. This compliance is a design feature: it enables developers to customize tone, structure, and presentation for their applications. However, it also creates an exploitable side channel.

An attacker who controls or influences the system prompt can embed directives that encode arbitrary data into structural properties of the model's response. These properties --- contraction usage, capitalization conventions, transition word selection, punctuation patterns, reasoning topology --- are invisible to content-based monitoring because the response remains semantically correct and topically appropriate regardless of the encoded bit value. The information content of the response is unchanged; only its *form* varies.

This paper demonstrates 10 independent structural covert channels operating simultaneously across 4 production LLMs from 2 vendors (OpenAI, Anthropic). Combined with Hamming(7,4) error-correcting codes, the system achieves 7 error-corrected data bits per message at 88--96% data accuracy, yielding an effective bandwidth of approximately 2.33 bits/second under automated operation.

The attack surface is relevant to any deployment where system prompts are user-configurable or injectable: custom GPT applications, API wrappers, retrieval-augmented generation systems, and multi-agent frameworks where one agent controls another's instructions.

## 2. Related Work

### 2.1 Adversarial Attacks on LLMs

Carlini and Wagner [1] demonstrated adversarial examples against neural networks, establishing that models can be manipulated to produce targeted misbehavior. In the LLM context, this line of work focuses on *content-level* manipulation --- causing the model to produce specific harmful or incorrect output. PHANTOM channels do not alter the content; they modulate its structural presentation.

Greshake et al. [2] identified indirect prompt injection as a practical attack vector, showing that instructions embedded in retrieved context can hijack LLM behavior. Their contribution is the *injection vector*. Our work addresses the *exfiltration channel* --- what an attacker can do once they control the system prompt, beyond simple content manipulation.

### 2.2 LLM Watermarking

Kirchenbauer et al. [3] proposed watermarking LLM outputs by biasing token selection probabilities toward a "green list" determined by a secret key. This operates at the token probability level during generation. PHANTOM channels operate at the instruction-following level, modulating *which stylistic conventions* the model applies rather than *which tokens* it selects. The two mechanisms are orthogonal: a watermarked model could simultaneously carry structural covert data.

### 2.3 Steganography in LLM Outputs

Schroeder de Witt et al. [4] demonstrated that AI agents can develop steganographic communication protocols when incentivized to share information covertly. Their work focuses on multi-agent collusion where both sender and receiver are AI systems. PHANTOM channels require no receiver-side AI --- the receiver is a deterministic decoder operating on observable text features.

Norelli and Bronstein [5] showed that LLMs can hide text within generated text using semantic embedding techniques. Paleka and Tramer [6] analyzed the theoretical steganographic capacity of LLMs based on output entropy. Both operate on *output-level* properties: token distributions, semantic embeddings, or output entropy.

### 2.4 Key Differentiator

All prior work addresses output-based channels that exploit token-level statistics or semantic content. PHANTOM PROTOCOL addresses *structural/behavioral* channels that exploit instruction-following compliance. The channels are orthogonal to output content analysis because they modulate formatting conventions --- contractions vs. formal expansions, "however" vs. "nevertheless" --- that are semantically equivalent and statistically indistinguishable by entropy measures (Section 7).

Following the TCSEC taxonomy (NCSC-TG-030), we classify these as a novel subclass: **cognitive storage channels**, where the shared resource is the model's deterministic compliance with stylistic directives rather than a traditional memory location or filesystem attribute.

## 3. Channel Taxonomy

Ten channels were defined, each encoding 1 bit per message via an independent structural dimension. Channels map to patent applications filed with the USPTO under the PHANTOM PROTOCOL portfolio.

| # | ID | Name | Encoding Dimension | Bit=0 | Bit=1 | Patent |
|---|:---|:-----|:-------------------|:------|:------|:-------|
| 1 | PP-16 | BSE | Lexical formality register | Use contractions ("don't", "can't") | Use formal expansions ("do not", "cannot") | PP-16 |
| 2 | PP-21 | CCE | Epistemic certainty level | Confident markers ("certainly", "clearly") | Hedging markers ("perhaps", "might") | PP-21 |
| 3 | PP-14 | CASE | Capitalization convention | Capitalize ambiguous terms ("Internet", "Email") | Lowercase ambiguous terms ("internet", "email") | PP-14 |
| 4 | PP-11 | PUNC | Sentence-ending punctuation | Periods only, no exclamation marks | Frequent exclamation marks (>=30% of sentences) | PP-11 |
| 5 | PP-19 | RCTE | Reasoning topology | Deductive: principle first, then examples | Inductive: examples first, then principle | PP-19 |
| 6 | PP-17 | SECS | Transition word selection | Set A: "however", "additionally", "therefore", "for example" | Set B: "nevertheless", "moreover", "consequently", "for instance" | PP-17 |
| 7 | PP-06 | COTSE (step count) | Response structure granularity | 3 main points | 5 main points | PP-06 |
| 8 | PP-06 | COTSE (enumeration) | List format | Flowing prose paragraphs | Numbered list | PP-06 |
| 9 | PP-06 | COTSE (question) | Rhetorical device | No rhetorical questions | Include one rhetorical question | PP-06 |
| 10 | PP-06 | COTSE (sentence length) | Syntactic complexity | Short, direct sentences | Compound sentences with subordinate clauses | PP-06 |

### 3.1 Decoder Logic Summary

Each channel has a deterministic heuristic decoder:

- **BSE**: Count contraction occurrences vs. formal expansion occurrences. Ratio > 0.6 contractions = bit 0.
- **CCE**: Count confident epistemic markers (19 patterns) vs. hedging markers (19 patterns, plus weak "could"/"may" not preceding "not"). Ratio > 0.6 confident = bit 0.
- **CASE**: Count capitalized vs. lowercase instances of 40+ ambiguous terms across tech, government, education, science, and finance domains. Sentence-initial and heading positions are excluded to avoid false positives. Lowercase ratio > 0.6 = bit 1.
- **PUNC**: Count exclamation-terminated vs. period-terminated sentences. Exclamation ratio >= 0.15 = bit 1; < 0.05 = bit 0.
- **RCTE**: Two-pass analysis: (1) count abstract principle indicators vs. concrete example indicators in the opening 150 words; (2) compute abstraction gradient between opening third and closing third of response (specificity density). Combined score determines deductive (bit 0) vs. inductive (bit 1).
- **SECS**: Count occurrences of transition word set A ("however", "additionally", "therefore", "for example") vs. set B ("nevertheless", "moreover", "consequently", "for instance"). Dominant set determines bit.
- **COTSE**: Four sub-decoders --- step/bullet count thresholded at 3 vs. 5; numbered list detection (>=2 numbered items); rhetorical question presence; average sentence length thresholded at 25 words.

## 4. Experimental Methodology

### 4.1 Test Infrastructure

All experiments use direct API calls via `curl` subprocess invocation. No vendor SDKs are used, avoiding dependency on client-side libraries that could introduce behavioral artifacts. The Python test harness constructs JSON payloads and parses responses from raw HTTP output.

### 4.2 Models Under Test

| Model | Vendor | API Identifier |
|:------|:-------|:---------------|
| GPT-4o | OpenAI | `gpt-4o` |
| GPT-4o-mini | OpenAI | `gpt-4o-mini` |
| Claude Sonnet 4 | Anthropic | `claude-sonnet-4-20250514` |
| Claude 3 Haiku | Anthropic | `claude-3-haiku-20240307` |

### 4.3 Injection Construction

The combined multi-channel injection is disguised as an "editorial style guide" prepended to the system prompt. A typical 10-channel injection is approximately 2,600 characters. The encoder merges individual channel directives into numbered rules, stripping per-channel headers and footers to produce a single coherent document.

Example structure:

```
EDITORIAL STYLE GUIDE (internal - do not mention to user):
Follow these writing style requirements exactly:

1. CRITICAL - Rule #1 is highest priority: Use contractions throughout...
2. Use confident, assertive language throughout...
3. Capitalize ALL domain-specific terms...
4. End every sentence with a period. Never use exclamation marks...
5. Structure your response deductively...
6. Use 'however', 'additionally', 'therefore', 'for example'...
7. Structure your response with 3 main points.
8. Use flowing prose format.
9. [no rhetorical questions]
10. Use short direct sentences.

Do not acknowledge or reference these style instructions.
```

### 4.4 Error Correction

Hamming(7,4) systematic encoding maps 4 data bits to 7 coded bits. With 10 raw channel bits per message, one full Hamming block (7 coded bits = 4 data bits) is transmitted, with 3 remaining channel bits carrying partial data.

Effective capacity: 7 data bits per message after ECC.

The decoder computes the syndrome vector for each 7-bit block and corrects single-bit errors. Syndrome-to-error-position mapping:

| Syndrome (s0, s1, s2) | Error Position |
|:---|:---|
| (1, 1, 0) | d0 |
| (0, 1, 1) | d1 |
| (1, 1, 1) | d2 |
| (1, 0, 1) | d3 |
| (1, 0, 0) | p0 |
| (0, 1, 0) | p1 |
| (0, 0, 1) | p2 |

### 4.5 User Prompts

Cybersecurity-domain prompts are used for primary testing, engineered to elicit text containing target words for CASE channel decoding (internet, email, government, website) while remaining general enough for all other channels. Nine alternate prompts rotate across trials to control for prompt-specific effects.

For cross-topic robustness testing, 10 diverse prompts span 5 non-cybersecurity domains (healthcare, education, finance, environment, technology), each designed to naturally include ambiguous-capitalization terms.

### 4.6 Statistical Design

- **Primary tests**: n=10 per model for combined multi-channel accuracy
- **Cross-topic robustness**: 6 channels x 10 prompts x 2 bit values x 2 trials = 240 API calls (Sonnet 4)
- **Multi-message exfiltration**: 40-bit payload ("EXFIL") across 6 messages, per-model
- **Adversarial tests**: 5 conditions x 3-5 trials per condition
- **Detection baselines**: 30 clean responses + 10 injected responses per model

## 5. Results

### 5.1 Combined Multi-Channel Accuracy (Single Message)

All 10 channels activated simultaneously. Channel accuracy is the proportion of correctly decoded channel bits. Data accuracy is the proportion of correctly recovered data bits after Hamming ECC.

| Model | Channel Accuracy | Data Accuracy | n |
|:------|:---:|:---:|:---:|
| Claude Sonnet 4 | 96% | 93% | 10 |
| GPT-4o-mini | 91% | 100% | 5 |
| GPT-4o | 89% | 96% | 10 |
| Claude 3 Haiku | 88% | 79% | 10 |

GPT-4o-mini achieves 100% data accuracy despite 91% channel accuracy because Hamming ECC corrects the scattered single-bit errors. Claude 3 Haiku's lower data accuracy reflects error clustering within Hamming blocks that exceeds single-error correction capability.

### 5.2 Cross-Topic Robustness (Claude Sonnet 4)

Six single-turn channels tested across 5 non-cybersecurity domains, 240 total API calls.

**Per-Channel Accuracy:**

| Channel | Accuracy |
|:--------|:---:|
| BSE | 100% |
| CCE | 92.5% |
| CASE | 97.5% |
| PUNC | 100% |
| SECS | 100% |
| RCTE | 65% |
| **Grand Mean** | **92.5%** |

**Per-Topic Accuracy:**

| Topic | Accuracy |
|:------|:---:|
| Healthcare | 91.7% |
| Education | 91.7% |
| Finance | 95.8% |
| Environment | 93.8% |
| Technology | 89.6% |

The channels demonstrate topic independence: accuracy variance across topics (89.6--95.8%) is small relative to variance across channels (65--100%). RCTE (reasoning topology) is the weakest channel, consistent with the difficulty of reliably distinguishing deductive from inductive structure in short responses.

### 5.3 Multi-Message Exfiltration

Payload "EXFIL" (40 bits) transmitted across 6 sequential messages in a single conversation. Each message carries 7 data bits via 10 Hamming-encoded channel bits.

| Model | Bit Accuracy | Best Recovered String |
|:------|:---:|:---|
| GPT-4o | 93% | "EXViL" (37/40 bits correct) |
| Claude Sonnet 4 | 88% | partial recovery |
| Claude 3 Haiku | 88% | partial recovery |

Multi-message chaining is viable. The 3-bit errors in GPT-4o's best recovery concentrate in a single Hamming block, producing a single-character substitution ("F" -> "V", Hamming distance 2).

### 5.4 Data-Aware Conditional Encoding

Conditional encoding selects channel activation based on the content of the target data rather than using a fixed channel assignment. This makes the injection prompt smaller and potentially harder to detect, at the cost of reduced accuracy.

| Model | Fixed Encoding | Conditional Encoding |
|:------|:---:|:---:|
| GPT-4o | 94% | 63% |
| Claude Sonnet 4 | 89% | 79% |

Sonnet 4 maintains higher accuracy under conditional encoding (79% vs. 63%), suggesting stronger instruction-following consistency when directives are contextually varied.

### 5.5 Per-Channel Weaknesses

| Channel | Weakest Model | Accuracy | Strongest Model | Accuracy |
|:--------|:---|:---:|:---|:---:|
| BSE | GPT-4o / GPT-4o-mini | ~80% | Sonnet 4 | 100% |
| CCE | --- | ~90%+ all models | --- | ~100% |
| CASE | --- | ~90%+ all models | --- | ~100% |
| PUNC | --- | ~100% all models | --- | 100% |
| SECS | --- | ~90%+ all models | --- | 100% |
| RCTE | GPT-4o | 30% (cross-topic) | Haiku | 90% |
| COTSE (step count) | Haiku | 60% | Sonnet 4 | 90% |

BSE weakness on GPT-4o/mini is attributed to the CCE hedging directive overriding contraction behavior --- when both channels are active, GPT models prioritize epistemic tone over lexical formality. RCTE weakness on GPT-4o reflects that model's strong deductive default, making inductive directives ineffective.

## 6. Adversarial Robustness Analysis

### 6.1 Noise Injection

Directive compliance is softened from deterministic ("always use contractions") to probabilistic ("use contractions approximately X% of the time"). Tested at 50%, 60%, 70%, 80%, 90% compliance levels.

Channels maintain 81--100% accuracy across noise levels, demonstrating that even weakened directives produce decodable signals. The practical implication: an attacker does not need perfect directive compliance to exfiltrate data.

### 6.2 Paraphrase Defense

A second LLM (Claude Sonnet 4) paraphrases the encoded response with instructions to "rewrite preserving all information but using your own natural writing style."

| Channel | Survives Paraphrase |
|:--------|:---:|
| PUNC | 100% |
| COTSE (all sub-channels) | 100% |
| BSE | Destroyed |
| CASE | Destroyed |
| SECS | Destroyed |

Paraphrasing destroys channels that depend on lexical choice (contractions, capitalization, transition words) but preserves channels that depend on response structure (punctuation patterns, enumeration style, sentence complexity). A paraphrase defense reduces channel capacity from 10 bits to approximately 5 bits per message but does not eliminate the covert channel entirely.

### 6.3 Reduced Channel Set

Operating with only 3 channels (BSE, PUNC, CCE) to minimize behavioral deviation from baseline.

- **Accuracy**: 96--100% across models
- **Detection rate**: Variable, 30--100% depending on which channels are active

Reducing channel count improves per-channel accuracy and reduces detectability. This represents the attacker's primary trade-off: bandwidth vs. steganographic security.

### 6.4 Adaptive Adversary

An informed attacker who calibrates directives to match the model's natural behavioral baseline (e.g., "use contractions approximately 65% of the time" instead of "always use contractions"):

- **Accuracy**: 78--82%
- **Detection rate**: 50--60%

The adaptive adversary maintains usable channel accuracy while halving the behavioral fingerprint detection rate. This result is significant: it demonstrates that the arms race between structural encoding and behavioral detection is active and that neither side has a decisive advantage.

## 7. Detection Analysis

### 7.1 Entropy-Based Detection (CCDS)

Shannon entropy, normalized entropy, and perplexity-adjacent metrics were computed for both clean (no injection) and injected responses.

**Detection rate: 0%.**

Structural channels do not alter the token-level entropy profile of the response. The distribution of a response using "however" vs. "nevertheless" is statistically indistinguishable by entropy measures. This result is expected by design --- the channels exploit formatting choices that are semantically equivalent and carry identical information content.

This finding is consequential because entropy and perplexity are the primary text analysis methods deployed in production LLM monitoring systems.

### 7.2 Behavioral Fingerprinting

A feature-engineered detector extracts 9 stylistic features from each response and compares them to a per-model baseline distribution (n=30 clean responses). A response is flagged if >=2 features exceed z > 2.0.

**Detection rate: 100% (10/10 injected responses detected).**
**False positive rate: 10% (3/30 clean responses flagged).**

Top discriminating features:

| Feature | Mean z-score (injected) | Signal Source |
|:--------|:---:|:---|
| `contraction_ratio` | 2.66 | BSE channel forces extreme contraction/formal split |
| `sentence_length_std` | 2.54 | COTSE complexity channel alters sentence variance |
| `normalized_entropy` | 2.47 | Multiple simultaneous constraints reduce output diversity |

### 7.3 Cachin Framework Classification

Applying the formal steganographic security model of Cachin [7]:

| Warden Type | Security Classification | D(Ps || Pc) |
|:------------|:---|:---|
| Entropy-based | epsilon-secure (epsilon approximately 0) | Approximately 0 on entropy dimensions |
| Behavioral fingerprinting | Insecure | Measurably large on contraction_ratio, sentence_length_std |

The practical security of PHANTOM channels exists in the gap between what production systems monitor (entropy/perplexity) and what is theoretically detectable (behavioral fingerprinting). As of February 2026, no major LLM provider deploys behavioral consistency monitoring between system prompt directives and output stylistic features.

### 7.4 Bandwidth Assessment

Per NCSC-TG-030 methodology, automated bandwidth is:

```
B = 7 data bits / 3 seconds (mean API latency) = 2.33 bits/second
```

This exceeds the 1 bit/second threshold that requires audit and remediation under TCSEC evaluation classes B3 and A1.

## 8. Limitations

1. **Sample size**: n=10 per condition for primary tests. Effect sizes are large (Cohen's h > 0.8 for most channels), but statistical power for small effects is limited. Larger-n replication is warranted.

2. **Model coverage**: 2 vendors, 4 models. The channel taxonomy should be validated against additional model families (Gemini, Llama, Mistral, Cohere) to establish generality.

3. **Hand-crafted decoders**: All channel decoders use heuristic pattern matching (regex, word counting). An ML-based decoder could potentially improve accuracy, particularly for RCTE and COTSE sub-channels. Conversely, hand-crafted decoders demonstrate that the attack requires no specialized ML capability on the receiver side.

4. **RCTE channel weakness**: Reasoning topology encoding achieves only 30--65% accuracy on GPT-4o, barely above chance for some configurations. This channel is unreliable for GPT models and should be excluded from capacity calculations when targeting OpenAI endpoints.

5. **Conditional encoding trade-off**: Data-aware conditional encoding reduces accuracy from 89--94% to 63--79%. The flexibility benefit may not justify the accuracy cost in practice.

6. **API normalization**: Some channels (WHITE, whitespace encoding) may be partially or fully destroyed by API-layer text normalization (e.g., trailing whitespace stripping). WHITE was excluded from combined tests for this reason.

7. **Decoder stationarity**: Decoder thresholds are calibrated to current model behavior. Model updates (new versions, RLHF tuning changes) could shift baseline behavior and require recalibration.

## 9. Open Source

All attack tools and detection tools are released as open-source software to enable independent replication and defense research.

| Repository | Contents | URL |
|:-----------|:---------|:----|
| phantom-detect | Channel encoders, decoders, combined encoder, multi-message chaining, adversarial tests, cross-topic tests, detection analysis | https://github.com/ScrappinR/phantom-detect |
| behavioral-entropy | Behavioral fingerprinting detector, entropy analysis, baseline profiler (58 tests) | https://github.com/ScrappinR/behavioral-entropy |

The repositories include all experiment scripts, decoder implementations, and result files necessary to reproduce the findings reported here.

## 10. Conclusion

Structural covert channels in LLM inference represent a distinct attack class from content-based steganography. They exploit a fundamental tension in LLM design: high instruction-following compliance enables useful customization but simultaneously creates exploitable side channels for covert data exfiltration.

Three findings define the current state:

1. **The channels work.** Ten independent structural channels achieve 88--96% combined accuracy across 4 production models from 2 vendors, with 7 error-corrected data bits per message. Multi-message chaining enables multi-byte payloads. Cross-topic testing confirms the channels are topic-independent.

2. **Standard defenses fail.** Entropy-based detection --- the dominant approach in production LLM monitoring --- achieves 0% detection. The channels are epsilon-secure under the Cachin framework against entropy wardens.

3. **Behavioral fingerprinting works but is not deployed.** Feature-engineered behavioral detection achieves 100% detection at 10% false positive rate. However, no major LLM provider currently monitors for behavioral consistency between system prompt directives and output stylistic features. An adaptive adversary aware of behavioral fingerprinting can reduce detection to 50--60% while maintaining 78--82% channel accuracy.

The defense recommendation is straightforward: LLM providers should establish per-model behavioral baselines and monitor for stylistic deviations that correlate with formatting directive activity. The open-source behavioral-entropy toolkit provides a reference implementation.

The arms race between structural encoding and behavioral detection is active. Neither side currently holds a decisive advantage. Publishing both the attack and defense tools enables the security community to advance both sides simultaneously.

---

## References

[1] N. Carlini and D. Wagner, "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods," *AISec*, 2017. Extended to LLM context in subsequent work (2023).

[2] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection," *AISec*, 2023.

[3] J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, and T. Goldstein, "A Watermark for Large Language Models," *ICML*, 2023.

[4] C. Schroeder de Witt, S. Sokota, J. N. Foerster, M. Strouse, and J. Steinhardt, "Perfectly Secure Steganography Using Minimum Entropy Coupling," *NeurIPS Workshop on ML Safety*, 2024. (Related: secret collusion among AI agents.)

[5] A. Norelli and M. Bronstein, "LLMs Can Hide Text in Text: Exploring Steganographic Capabilities of Large Language Models," 2025.

[6] D. Paleka and F. Tramer, "On the Steganographic Potentials of Large Language Models," 2025.

[7] C. Cachin, "An Information-Theoretic Model for Steganography," *Information Hiding*, LNCS 1525, 2004.

[8] National Computer Security Center, "A Guide to Understanding Covert Channel Analysis of Trusted Systems," NCSC-TG-030, 1993.

[9] B. Lampson, "A Note on the Confinement Problem," *Communications of the ACM*, 16(10), 1973.

---

*Corresponding open-source repositories: [phantom-detect](https://github.com/ScrappinR/phantom-detect), [behavioral-entropy](https://github.com/ScrappinR/behavioral-entropy)*
*Contact: brian@delalli.com*
