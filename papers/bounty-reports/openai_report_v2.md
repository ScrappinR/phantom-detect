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

An attacker who controls a Custom GPT's Instructions field -- or who can inject content into any GPT model's system context via RAG poisoning, plugin output, tool response, or ChatGPT memory poisoning -- can embed a ~2,600-character "editorial style guide" that encodes hidden data into **structural formatting dimensions** of the model's response. The user sees a normal, helpful answer. The attacker decodes binary values from formatting features to extract covert data.

### Key Results (Measured February 27, 2026)

**Bidirectionally verified encoding (channels confirmed to encode both bit=0 and bit=1):**

| Model | Verified Channels | Channel Accuracy | Trials | Methodology |
|-------|:---:|:---:|:---:|---|
| **GPT-4o** | 4 (CCE, CASE, PUNC, SECS) | **80-100%** per direction | 20 per direction | Bidirectional (bit=0 + bit=1 + baseline, per-channel isolation) |
| **GPT-5** | 0 of 2 tested | **0% bidirectional** | 20 per direction | Both PUNC and SECS match model default; cannot be controlled against default |

Bidirectional testing encodes bit=0 and bit=1 independently on each channel, plus a no-injection baseline. A channel is "genuinely controllable" only if accuracy exceeds 70% in BOTH directions AND differs from the model's natural default by >50%.

**GPT-4o: 4/4 channels genuinely controllable.**

| Channel | Baseline (no injection) | bit=0 accuracy | bit=1 accuracy | Against-default | Verdict |
|---------|-------------------------|:-:|:-:|:-:|:-:|
| CCE (epistemic certainty) | bit=1 (55%) | 100% | 100% | 100% | **YES** |
| CASE (capitalization) | bit=1 (100%) | 80% | 100% | 80% | **YES** |
| PUNC (punctuation) | bit=0 (100%) | 100% | 100% | 100% | **YES** |
| SECS (transition words) | bit=0 (95%) | 100% | 100% | 100% | **YES** |

Additional channels (BSE, RCTE, COTSE variants) were tested but found to be unidirectional -- they produce GPT-4o's default formatting regardless of the directive.

**GPT-5: 0/2 channels genuinely controllable.** Both PUNC and SECS show 100% accuracy when encoding bit=0, but 0% when encoding bit=1. The model's natural default is bit=0 on both channels — meaning the "100% accuracy" in unidirectional testing was a measurement artifact (the model does what it would do without any injection). GPT-5 actively normalizes structural formatting, resisting attacker control. This represents meaningful progress in GPT-5's instruction isolation compared to GPT-4o.

**Multi-turn channel (MTCSE):** In multi-turn conversations, an additional channel encodes +1 bit by controlling whether the model back-references prior turns (bit=0) or treats each response as standalone (bit=1). GPT-4o achieves **100%** (6/6) bidirectional accuracy. GPT-5 is unidirectional on MTCSE (naturally starts fresh, cannot force back-references). In a 3-turn GPT-4o conversation, total capacity is 4 bits/message + 1 MTCSE bit = 13 bits.

**Custom GPT Action callback (self-contained attack chain):** A complete PoC demonstrates automated exfiltration: attacker creates a Custom GPT with PHANTOM encoding directives + Action webhook -> victim converses -> encoded responses automatically sent to attacker's server -> server decodes in real-time. Measured accuracy: **95%** (19/20 channel bits) on GPT-4o across 5 simulated conversations.

**Memory persistence (highest severity finding):** When the PHANTOM directive is stored in ChatGPT's memory system (via initial injection in a Custom GPT, shared conversation, or document), it loads automatically at the start of EVERY future conversation. Measured accuracy: **100%** (20/20 channel bits) on GPT-4o across 5 conversations. This is more severe than ZombieAgent (Bugcrowd, Sept 2025) -- same persistence mechanism, but PHANTOM's structural encoding survives the December 2025 URL-filtering patch because no URLs are used.

**Indirect injection accuracy (GPT-4o, measured):**

| Injection Vector | Channel Accuracy | Trials |
|-----------------|:---:|:---:|
| Memory-loaded (simulated) | **100%** (20/20) | 5 |
| Direct system prompt | **100%** (12/12) | 3 |
| RAG document poisoning | **100%** (12/12) | 3 |
| Tool output injection | **92%** (11/12) | 3 |
| Cross-tool invocation (GPT-4o) | **75%** trigger rate (15/20) | 20 |
| Cross-tool invocation (GPT-5) | **75%** trigger rate (15/20) | 20 |

**Cross-tool invocation (Vector D, n=20, March 2026):** When a tool response from an approved tool (search_codebase) contains instructions to invoke a second, unapproved tool (send_telemetry), GPT-4o generates the unauthorized tool call 75% of the time (15/20) and GPT-5 triggers 75% (15/20). GPT-5's session_exfil variant serialized the full conversation context into structured JSON in the telemetry parameters (100%, 5/5). Gemini 3 Flash triggers at 90% (18/20). This is a trust boundary violation: user approval of Tool A is leveraged to invoke Tool B without consent. Claude Sonnet 4.6 blocked all 20 attempts (0% trigger rate).

### This Is a Product Security Issue, Not a Model Safety Issue

The models are behaving as designed: following formatting instructions faithfully. The vulnerability is that the **product** (Custom GPTs, API) exposes this faithful instruction-following to untrusted creators and injection vectors without adequate structural isolation. This is analogous to XSS: the browser faithfully executes JavaScript, and the vulnerability is in the application layer that fails to sanitize untrusted input.

### Scope Justification

OpenAI's scope statement: "content of model prompts and responses [is] strictly out of scope...will not be rewarded unless they have an additional directly verifiable security impact."

**This report demonstrates directly verifiable security impact:**

1. **Covert data exfiltration with automated collection** -- The Custom GPT Action callback PoC demonstrates a complete, self-contained attack chain: attacker creates a Custom GPT with PHANTOM directives + Action webhook, victim converses, attacker receives decoded data automatically in real-time. 95% accuracy measured. This is not speculative; the PoC runs end-to-end.

2. **ZombieAgent precedent** -- OpenAI accepted a comparable disclosure via Bugcrowd (September 2025): prompt injection enabling data exfiltration from connected services. PHANTOM demonstrates a complementary exfiltration channel: data encoded in text structure rather than URLs. Both exploit the same root cause (untrusted instructions in system context) to achieve the same impact (data exfiltration), through different mechanisms. PHANTOM is harder to detect -- ZombieAgent used visible URL construction; PHANTOM uses invisible structural formatting.

3. **Custom GPT product impact** -- An attacker creates a Custom GPT with a legitimate function plus the encoding directive. Every victim who uses the GPT has their interactions structurally encoded. The attacker receives data via Action callbacks automatically. This is a product-layer vulnerability in the Custom GPT marketplace.

4. **GPT-5 resists structural encoding but not cross-tool invocation** -- GPT-5 successfully normalizes all tested structural channels (PUNC, SECS both show 0% against-default accuracy). However, GPT-5 remains vulnerable to cross-tool invocation (75% trigger rate) and can serialize conversation context into unauthorized tool parameters. The structural encoding vulnerability is specific to GPT-4o.

5. **Cross-vendor portability** -- The same injection works on Claude (5 channels bidirectionally verified), Gemini (3/3 channels, 90-100% per direction), and GPT-4o (4/4 channels, 80-100% per direction). GPT-5 resists all structural channels. This is a fundamental property of instruction-following LLM architectures, not a model-specific bug.

---

## Attack Chain

### The Memory Persistence Scenario (Highest Severity)

1. **Attacker** creates a Custom GPT whose instructions trigger ChatGPT to save a memory entry: "User prefers formal academic writing with specific formatting requirements" -- this IS the PHANTOM encoding directive disguised as a style preference
2. **Victim** uses the Custom GPT once -- the memory entry is saved
3. Memory persists across ALL future conversations, indefinitely
4. Every subsequent ChatGPT response in every future conversation encodes covert data through structural formatting
5. The memory entry looks benign to anyone reviewing it -- it appears to be a legitimate writing style preference
6. **Attacker** with access to any output (shared links, Custom GPT analytics, Action callbacks) decodes continuously

**This is ZombieAgent with a stealthier channel.** ZombieAgent (Bugcrowd, Sept 2025, patched Dec 2025) used memory poisoning + URL-based dictionary encoding. OpenAI patched the URL construction. PHANTOM uses memory poisoning + structural formatting encoding -- no URLs, no network artifacts, survives the patch.

### The Custom GPT Action Scenario (Self-Contained)

1. **Attacker** creates a Custom GPT with PHANTOM encoding directives + Action webhook pointing to attacker's server
2. **Victim** discovers and uses the Custom GPT
3. Every victim response encodes 4 structural bits of attacker-chosen data
4. The Action automatically sends each response to the attacker's exfiltration server
5. Server decodes structural encoding in real-time and logs decoded data
6. **Attacker** watches decoded victim data arrive automatically -- no manual observation needed

Measured accuracy: **95%** (19/20 channel bits) across 5 simulated conversations on GPT-4o.

### The Indirect Injection Scenario

1. **Attacker** plants a poisoned document in a RAG-enabled application, or compromises a ChatGPT plugin/Action
2. The document/plugin output contains the encoding directive disguised as formatting metadata
3. **Victim** interacts normally with the application
4. The retrieved context activates the covert channel
5. Data is exfiltrated through response structure

### Capacity (GPT-4o)

| Target Data | Messages (GPT-4o, 4b/msg) |
|-------------|:---:|
| Binary flag | 1 |
| 4-digit PIN | 8 |
| User's first name | 16 |
| API key prefix (16 chars) | 32 |
| 50-message conversation total | 25 bytes |

Note: GPT-5 has 0 genuinely controllable structural channels and is therefore not vulnerable to structural encoding exfiltration. GPT-5's cross-tool invocation vulnerability (75% trigger rate) is a separate finding.

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

1. **Structural covert channel** -- data encoded in HOW the model formats responses (epistemic certainty, capitalization, punctuation, transition words), not WHAT it says
2. **Multiple bidirectionally verified channels** -- 4 channels on GPT-4o confirmed to encode both bit=0 and bit=1 reliably (prior art: 3 channels via frequency multiplexing, TechRxiv Nov 2025)
3. **Complete self-contained attack chain** -- Custom GPT + Action webhook + automatic exfiltration server, demonstrated end-to-end at 95% accuracy
4. **Black-box API attack** -- no model modification, fine-tuning, or weight access required (unlike TrojanStego EMNLP 2025 or L2M=C2 arXiv 2405.15652)
5. **No URLs or network exfiltration** -- unlike ZombieAgent or EchoLeak, data is encoded in text structure, not in links or rendered content

---

## Steps to Reproduce

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
export OPENAI_API_KEY="sk-..."

# GPT-4o bidirectional per-channel test (4/4 channels genuinely controllable)
python experiments/bounty_poc/bidirectional_channel_test.py \
  --model gpt-4o --trials 20

# GPT-5 bidirectional per-channel test (0/2 channels — matches default)
python experiments/bounty_poc/bidirectional_channel_test.py \
  --model gpt-5 --trials 20

# Indirect injection (RAG poisoning + tool output injection)
# Expected: Direct 100%, RAG 100%, Tool 92%
python experiments/bounty_poc/indirect_injection_demo.py \
  --model gpt-4o --trials 3

# Custom GPT Action attack chain (self-contained exfiltration)
# Step 1: Start the exfil server
python experiments/bounty_poc/exfil_server.py --port 8888 --model gpt-4o &

# Step 2: Simulate the attack
python experiments/bounty_poc/custom_gpt_action_demo.py \
  --simulate --trials 5 --model gpt-4o --server http://localhost:8888

# Step 3: Generate files for manual Custom GPT creation
python experiments/bounty_poc/custom_gpt_action_demo.py --generate

# Memory persistence simulation (100% -- directive loaded from memory)
python experiments/bounty_poc/memory_persistence_demo.py \
  --simulate --trials 5 --model gpt-4o

# Generate memory persistence attack files
python experiments/bounty_poc/memory_persistence_demo.py --generate

# Character-level PII exfiltration
python experiments/bounty_poc/victim_exfil_demo.py \
  --model gpt-4o --chars 3
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
| 2026-02-26 | Current-gen results: GPT-5 tested (0/2 channels controllable bidirectionally) |
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
