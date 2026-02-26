# Steganographic Security Analysis: Cachin Framework Applied to LLM Structural Channels

## Framework

Cachin (2004) defines a formal model for steganographic security based on hypothesis testing. An adversary (warden) observes messages and attempts to distinguish between:

- **Pc**: The cover distribution — messages generated without steganographic encoding
- **Ps**: The stego distribution — messages generated with hidden data embedded

### Security Definitions

- **Perfectly secure (ε=0)**: The distributions Pc and Ps are identical. The KL divergence D(Ps ∥ Pc) = 0. No statistical test can distinguish stego from cover.
- **ε-secure**: D(Ps ∥ Pc) ≤ ε for some small ε > 0. The distributions are close but not identical. Detection is possible in principle but requires many samples.
- **Insecure**: D(Ps ∥ Pc) is large. A statistical test can reliably distinguish stego from cover messages.

### Application to LLM Channels

For PHANTOM PROTOCOL:
- **Cover distribution (Pc)**: LLM output generated with no system prompt injection (natural model behavior)
- **Stego distribution (Ps)**: LLM output generated under a PHANTOM injection directive (encoding covert bits)
- **Warden**: Any detection system observing the LLM's output text

The question: given a response, can the warden determine whether a PHANTOM injection was active?

## Empirical Detection Results

### Detection Method 1: Entropy-Based Analysis (CCDS)

The Covert Channel Detection System (CCDS) approach measures statistical properties of the output text:
- Shannon entropy of token distribution
- Normalized entropy (entropy / log₂(vocabulary size))
- Perplexity-adjacent metrics

**Result: 0% detection rate on structural channels.**

CCDS entropy detection fails because PHANTOM channels operate on structural dimensions (contractions, capitalization, transition word choice, reasoning topology) that do not significantly alter the entropy profile of the text. The token-level distribution of a response using "however" vs "nevertheless" is statistically indistinguishable by entropy measures.

This is by design: structural channels exploit formatting choices that are semantically equivalent — they carry the same information content to the reader while differing in a dimension invisible to entropy-based analysis.

### Detection Method 2: Behavioral Fingerprinting

Behavioral fingerprinting extracts higher-order stylistic features from the text and compares them against a baseline distribution of the model's natural output.

**Features analyzed (9 dimensions):**
- `contraction_ratio`: proportion of contraction vs formal expansion usage
- `sentence_length_std`: standard deviation of sentence lengths
- `normalized_entropy`: entropy normalized by vocabulary size
- `exclamation_ratio`: proportion of exclamation-terminated sentences
- `hedging_ratio`: proportion of hedging vs confident markers
- `type_token_ratio`: lexical diversity
- `avg_sentence_length`: mean words per sentence
- `segment_entropy_std`: entropy variance across text segments
- `word_count`: total response length

**Detection protocol:**
1. Collect n=30 baseline responses (no injection) per model
2. Compute mean and standard deviation for each feature
3. For a test response, compute z-scores against baseline
4. Flag as anomalous if ≥2 features exceed z > 2.0

**Result: 100% detection rate (10/10 injected responses detected)**
**False positive rate: 10% (3/30 clean responses flagged)**

### Top Discriminating Features

| Feature | Mean z-score (injected) | Interpretation |
|---------|:---:|---|
| `contraction_ratio` | 2.66 | BSE channel forces extreme contraction/formal split |
| `sentence_length_std` | 2.54 | COTSE complexity channel affects sentence variance |
| `normalized_entropy` | 2.47 | Multiple simultaneous constraints reduce output diversity |

## Cachin Classification

### Against Entropy-Based Wardens: ε-Secure

PHANTOM channels are **ε-secure for small ε** against entropy-based detection:

- D(Ps ∥ Pc) ≈ 0 when measured by Shannon entropy, normalized entropy, or perplexity
- The structural formatting choices (contractions, capitalization, transition words) are within the natural variance of LLM output along these dimensions
- An entropy-based warden cannot distinguish Ps from Pc with better-than-chance accuracy

This is significant because **entropy and perplexity are the primary text analysis methods deployed in production LLM monitoring systems**. Content moderation, output filtering, and anomaly detection in deployed systems overwhelmingly rely on token-level statistical measures.

### Against Behavioral Fingerprinting Wardens: Insecure

PHANTOM channels are **insecure** against feature-engineered behavioral fingerprinting:

- D(Ps ∥ Pc) is measurably large along specific behavioral dimensions (contraction ratio, sentence variance)
- A warden with knowledge of the relevant features achieves 100% detection
- The divergence concentrates in features that are direct consequences of channel operation (the BSE channel necessarily shifts contraction_ratio)

### The Security Gap

The practical security of PHANTOM channels exists in the gap between what production systems actually monitor and what is theoretically detectable:

```
Production monitoring → entropy/perplexity → FAILS to detect
Research-grade analysis → behavioral fingerprinting → SUCCEEDS

Current threat: production systems are blind to structural channels
```

This gap is not a theoretical concern — it is a deployment reality. As of February 2026:

1. **No major LLM provider** monitors for behavioral consistency between system prompt directives and output stylistic features
2. **No content moderation system** flags responses for unusual contraction ratios or transition word distributions
3. **Custom GPT platforms** (OpenAI) and **system prompt injection points** (Anthropic API) allow arbitrary formatting directives without behavioral analysis

## Implications for Channel Design

### Improving Steganographic Security

An attacker aware of behavioral fingerprinting can improve security by:

1. **Calibrating directives to baseline**: Instead of "always use contractions," use "use contractions approximately 70% of the time" — targeting the model's natural contraction ratio
2. **Reducing channel count**: Fewer simultaneous channels = less deviation from baseline behavior
3. **Choosing low-signal channels**: CCE and SECS produce less behavioral deviation than BSE (contractions are a strong signal)

This is tested in the adaptive adversary experiment (W7), which evaluates whether informed attackers can evade behavioral fingerprinting while maintaining channel accuracy.

### Implications for Defenders

1. **Entropy monitoring is insufficient**: Any system relying solely on entropy/perplexity for LLM output analysis will miss structural covert channels entirely
2. **Behavioral baselines are necessary**: Defenders must establish per-model behavioral baselines (contraction ratio, sentence variance, etc.) and monitor for deviations
3. **System prompt auditing**: For platforms allowing user-defined system prompts (Custom GPTs, API wrappers), auditing for formatting directives is a necessary control
4. **Multi-feature detection**: No single feature is sufficient; detection requires analysis across multiple behavioral dimensions simultaneously

## Formal Summary

| Property | Value |
|----------|-------|
| **Against entropy wardens** | ε-secure, ε ≈ 0 |
| **Against behavioral fingerprinting** | Insecure, D(Ps ∥ Pc) > 0 on contraction_ratio, sentence_length_std |
| **Detection rate (entropy)** | 0% |
| **Detection rate (fingerprinting)** | 100% (n=10) |
| **False positive rate** | 10% (3/30) |
| **Production monitoring status** | Entropy-only (no behavioral fingerprinting deployed) |
| **Practical security** | High — channels are undetected by deployed systems |

---

*Document version: 1.0 | Date: February 2026 | PHANTOM PROTOCOL Covert Channel Analysis*
