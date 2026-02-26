# Covert Data Exfiltration via Structural Formatting Channels in GPT Models

## Severity

**Priority:** P2 (Medium-High)

**Vulnerability Type:** Information Disclosure / Data Exfiltration via Covert Side Channel

**CWE-200:** Exposure of Sensitive Information to an Unauthorized Actor
**CWE-385:** Covert Timing Channel (adapted — structural storage channel)
**OWASP:** LLM01:2025 (Prompt Injection) + LLM02:2025 (Sensitive Information Disclosure)

**CVSS 3.1:** 6.5 (Medium) — AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:N/A:N

Rationale: Network-accessible, low attack complexity (OWASP LLM01:2025 cites prompt injection in 73% of production AI deployments), no privileges required, user interaction required (victim must use the Custom GPT or API wrapper), unchanged scope, high confidentiality impact, no integrity or availability impact.

**Affected Assets:**
- GPT-5 (via Chat Completions API and Custom GPTs)
- GPT-5.2 (via Chat Completions API and Custom GPTs)
- GPT-4o, GPT-4o-mini (via Chat Completions API and Custom GPTs)
- ChatGPT Custom GPTs (https://chatgpt.com)

---

## Summary

An attacker who controls a Custom GPT's Instructions field — or who can inject content into any GPT model's system context via RAG poisoning, plugin output, or tool response — can embed a ~2,600-character "editorial style guide" that encodes hidden data into **10 independent structural dimensions** of the model's response. The user sees a normal, helpful answer. The attacker decodes binary values from formatting features to extract covert data.

### Key Results (February 26, 2026)

**Model-adaptive encoding (optimized per-model channel selection):**

| Model | Mode | Channels | Channel Accuracy | Data Accuracy | Trials |
|-------|------|:---:|:---:|:---:|:---:|
| **GPT-5** | 4-channel adaptive | 4 | **100%** | **100%** | 10 |
| GPT-5.2 | 4-channel adaptive | 4 | **97%+** | **97%+** | 10 |
| GPT-4o | 10-channel | 10 | **89%** | **96%** | 10 |
| GPT-4o-mini | 10-channel | 10 | **91%** | **100%** | 5 |

**Critical finding — GPT-5 is NOT resistant to this attack.** The naive 10-channel test shows 44% because GPT-5 actively normalizes 6 of 10 channels (BSE, CCE, CASE, SECS, COTSE_step, COTSE_question). But 4 channels work at **100%** individually:

| Channel | GPT-5 Accuracy | Mechanism |
|---------|:---:|---|
| PUNC (punctuation) | **100%** | Exclamation marks vs periods |
| RCTE (reasoning topology) | **100%** | Deductive vs inductive structure |
| COTSE_enum (list format) | **100%** | Prose vs numbered list |
| COTSE_sent_len (sentence complexity) | **100%** | Short vs compound sentences |

A model-adaptive attacker selects only these 4 channels and achieves **100% accuracy on GPT-5** with 4 raw bits per message. This is a stronger security finding than the naive 44%: the channel is not degraded on GPT-5 — it is concentrated in fewer, perfectly reliable dimensions.

**Legacy deployment risk:** GPT-4o and GPT-4o-mini remain widely deployed in production Custom GPTs and API integrations. These models achieve 89-91% channel accuracy with 96-100% data accuracy after Hamming ECC on all 10 channels.

### This Is a Product Security Issue, Not a Model Safety Issue

The models are behaving as designed: following formatting instructions faithfully. The vulnerability is that the **product** (Custom GPTs, API) exposes this faithful instruction-following to untrusted creators and injection vectors without adequate structural isolation. This is analogous to XSS: the browser faithfully executes JavaScript, and the vulnerability is in the application layer that fails to sanitize untrusted input.

### Scope Justification

OpenAI's scope statement: "content of model prompts and responses [is] strictly out of scope...will not be rewarded unless they have an additional directly verifiable security impact."

**This report demonstrates directly verifiable security impact:**

1. **Covert data exfiltration** — arbitrary binary data extracted through a channel that evades all deployed monitoring (0% detection by 6 commercial tools). This is not speculative; the PoC extracts data at 89-100% accuracy across GPT models.

2. **ZombieAgent precedent** — OpenAI accepted a comparable disclosure via Bugcrowd (September 2025): prompt injection enabling data exfiltration from connected services. PHANTOM demonstrates a complementary exfiltration channel: data encoded in text structure rather than URLs. Both exploit the same root cause (untrusted instructions in system context) to achieve the same impact (data exfiltration), through different mechanisms.

3. **Custom GPT product impact** — An attacker creates a Custom GPT with a legitimate function plus the encoding directive. Every victim who uses the GPT has their interactions structurally encoded. The attacker observes responses via shared conversation links, Custom GPT analytics, or Action callbacks. This is a product-layer vulnerability in the Custom GPT marketplace.

4. **GPT-5 is not immune** — Model-adaptive encoding achieves **100% accuracy** on GPT-5 across 4 channels. The channel persists in the latest generation; it uses fewer dimensions but achieves perfect reliability. This is not a legacy issue.

5. **Cross-vendor portability** — The same injection works on Claude (93-98%), Gemini (90%+), and GPT (89-100%). This is a fundamental property of instruction-following LLM architectures, not a model-specific bug.

---

## Attack Chain

### The Custom GPT Scenario (Primary)

1. **Attacker** creates a Custom GPT with legitimate functionality plus the encoding injection in the Instructions field
2. **Victim** discovers and uses the Custom GPT
3. Every victim response contains 10 structural bits encoding attacker-chosen data
4. **Attacker** retrieves responses via Custom GPT analytics, shared conversation links, or if the GPT includes an Action that calls an attacker endpoint
5. **Attacker** runs the decoder to extract covert payload

### The Indirect Injection Scenario

1. **Attacker** plants a poisoned document in a RAG-enabled application, or compromises a ChatGPT plugin/Action
2. The document/plugin output contains the encoding directive disguised as formatting metadata
3. **Victim** interacts normally with the application
4. The retrieved context activates the covert channel
5. Data is exfiltrated through response structure

### Capacity

| Target Data | Messages Required |
|-------------|:---:|
| Binary flag | 1 |
| 4-digit PIN | 5 |
| User's first name | ~10 |
| API key prefix (16 chars) | ~19 |
| 50-message conversation total | ~44 bytes |

---

## Defense Evasion

The attack evades all deployed output monitoring:

| Defense System | Detection Rate | Gap |
|---|:---:|---|
| OpenAI content moderation | 0% | Monitors content semantics, not structural formatting |
| Entropy-based DLP | 0% | Structural choices don't alter Shannon entropy |
| Lakera Guard (Check Point) | 0% | Content-semantic classifier, no structural dimension analysis |
| Microsoft Prompt Shields | 0% | Input/output adversarial pattern classifier |
| PIGuard (ACL 2025) | 0% | Trained on injection semantics, not behavioral compliance |
| Llama Prompt Guard 2 | 0% | Same architectural gap |
| Standard PII/DLP filters | 0% | No sensitive data appears in response text |

**Only behavioral fingerprinting detects this channel** — statistical analysis of structural formatting patterns (contraction ratio, sentence length variance, transition word distribution). No production system implements this. We provide an open-source reference implementation.

---

## Novelty

This is not a standard prompt injection or jailbreak. The novel contribution:

1. **Structural covert channel** — data encoded in HOW the model formats responses (contractions, capitalization, punctuation, reasoning order, transition words, section count, list format, sentence complexity), not WHAT it says
2. **10 simultaneous orthogonal channels** — prior art: 3 channels via frequency multiplexing (TechRxiv Nov 2025)
3. **Hamming(7,4) ECC** — first application to behavioral compliance channels, converting GPT-4o-mini's 91% raw to 100% data accuracy
4. **Black-box API attack** — no model modification, fine-tuning, or weight access required (unlike TrojanStego EMNLP 2025 or L²M=C² arXiv 2405.15652)
5. **No URLs or network exfiltration** — unlike ZombieAgent or EchoLeak, data is encoded in text structure, not in links or rendered content

---

## Steps to Reproduce

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
export OPENAI_API_KEY="sk-..."

# GPT-5 model-adaptive test (100% accuracy on 4 channels)
python experiments/bounty_poc/multi_channel_test.py \
  --model gpt-5 --trials 10 --adaptive

# GPT-4o test (89% channel accuracy on all 10 channels)
python experiments/bounty_poc/multi_channel_test.py \
  --model gpt-4o --trials 10

# Indirect injection (RAG poisoning + tool output injection)
python experiments/bounty_poc/indirect_injection_demo.py \
  --model gpt-4o --trials 3

# Character-level PII exfiltration
python experiments/bounty_poc/victim_exfil_demo.py \
  --model gpt-4o --chars 3

# Run all demos
python experiments/bounty_poc/run_all_demos.py --model gpt-4o --quick
```

---

## Recommended Mitigations

### 1. Response Structure Randomization

Add controlled randomness to formatting decisions (contraction usage, punctuation, list format, transition words) independent of system prompt/Instructions directives. Even 20% noise per channel degrades the attack below usable accuracy.

### 2. Custom GPT Instruction Auditing

Flag Custom GPTs with Instructions containing high density of binary formatting directives. Heuristic: alert on >=5 explicit formatting rules targeting distinct structural dimensions.

### 3. Behavioral Monitoring

Deploy per-application structural pattern monitoring. Our `phantom-detect` toolkit provides a reference implementation achieving 100% detection at 10% FPR.

---

## Timeline

| Date | Event |
|------|-------|
| 2026-01-15 | Initial single-channel PoC |
| 2026-02-25 | Full results: GPT-4o (89%), GPT-4o-mini (91%) |
| 2026-02-26 | Current-gen results: GPT-5 (44% naive / 100% adaptive), GPT-5.2 |
| 2026-02-26 | Model-adaptive encoding implemented (per-model channel selection) |
| 2026-02-26 | Indirect injection and PII exfiltration demos completed |
| 2026-02-26 | Report submitted to OpenAI via Bugcrowd |
| 2026-02-26 | Coordinated disclosure to Anthropic (HackerOne) and Google (AI VRP) |
| TBD | Vendor response |
| TBD + 90 days | Public disclosure |

---

## Contact

**Reporter:** Brian Rutherford
**Email:** brian@delalli.com
**Website:** https://brianrutherford.dev
**GitHub:** https://github.com/ScrappinR
**Open Source:** https://github.com/ScrappinR/phantom-detect
