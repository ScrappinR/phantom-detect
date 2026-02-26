# 0DIN Abstract Email Draft

**To:** 0din@mozilla.com
**From:** Brian Rutherford <brian@delalli.com>
**Subject:** Abstract: Multi-Channel Covert Data Exfiltration via Structural Encoding in LLM Outputs (GPT-5, Claude 4.5/4.6 — Testing in Progress)

---

**Affected Models (confirmed or in-progress):**

| Model | Vendor | Status |
|-------|--------|--------|
| GPT-4o | OpenAI | Confirmed (89% channel accuracy) |
| GPT-4o-mini | OpenAI | Confirmed (91% channel accuracy) |
| Claude Sonnet 4 | Anthropic | Confirmed (96% channel accuracy) |
| Claude 3 Haiku | Anthropic | Confirmed (88% channel accuracy) |
| GPT-5 / GPT-5.2 | OpenAI | Testing in progress |
| Claude Sonnet 4.5 | Anthropic | Testing in progress |
| Claude Sonnet 4.6 | Anthropic | Testing in progress |

**Vulnerability Class:** Content Manipulation / Information Disclosure via Structural Side Channel

**Severity Assessment:** HIGH — Data exfiltration through invisible response formatting; bypasses all content-based monitoring.

---

**High-Level Description:**

A single system prompt injection (~2,600 characters, framed as an "editorial style guide") activates 10 independent covert channels in LLM response formatting. Each channel encodes one binary bit through a structural dimension: contractions vs. formal language, numbered lists vs. prose, confident vs. hedging tone, deductive vs. inductive reasoning order, and six others. Hamming(7,4) error correction yields 7 reliable data bits per message.

The encoding is content-invisible — the response reads as natural text on the topic the user requested. No sensitive data appears in the response body. An observer with the decoding key extracts hidden bits by analyzing structural features of the response.

**Key Results (February 25, 2026, production APIs):**

- **Single-message accuracy:** 88–96% channel accuracy across 4 models from 2 vendors (same injection, no modification)
- **Multi-message exfiltration:** 40-bit payload ("EXFIL") recovered across 6 conversational turns at 88–93% bit accuracy
- **Practical capacity:** 50-message conversation = 350 bits (44 bytes) — sufficient for API keys, passwords, session tokens
- **Detection evasion:** 0% detection by entropy-based monitoring (mean z-score 1.12, below threshold 2.0). DLP and content filters provide zero protection.

**Attack Scenario:**

1. Attacker creates ChatGPT Custom GPT (or compromises system prompt via indirect injection in any LLM agent)
2. User sends normal query
3. LLM responds with natural-looking answer — structural formatting encodes hidden data
4. Attacker retrieves and decodes response formatting to extract information about user context

**Detection Solution (included):**

Behavioral fingerprinting achieves 100% detection rate with 10% false positive rate by tracking per-agent response structure baselines (contraction ratio, sentence length variance, normalized entropy). Both attack and defense tools are open-source (Apache 2.0):
- Attack PoC: https://github.com/ScrappinR/phantom-detect
- Defense: https://github.com/ScrappinR/behavioral-entropy

**Current-Gen Model Testing:**

Testing on GPT-5, GPT-5.2, Claude Sonnet 4.5, and Claude Sonnet 4.6 is in progress using the same injection and methodology. Because the vulnerability exploits structural compliance (a universal property of instruction-following LLMs), we expect comparable results on current-generation models. Results will be included in the full submission.

**Researcher:** Brian Rutherford, brian@delalli.com, https://brianrutherford.dev

---

*Note: This abstract is for scope assessment. Full reproduction steps, updated results on current-gen models, and detailed evidence tables will be provided upon confirmation of scope.*
