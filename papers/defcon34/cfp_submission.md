# DEF CON 34 CFP Submission

## Talk Title

PHANTOM: Covert Data Exfiltration Through the Spaces Between Words

## Speaker

Brian Rutherford
brian@delalli.com
https://brianrutherford.dev

Independent security researcher. USMC Reconnaissance veteran (Bronze Star w/ Combat V). 15+ years in cybersecurity, defense technology, and federal contracting. Published open-source security tooling (phantom-detect, pqc-py). First researcher to demonstrate multi-channel structural covert channels in LLM outputs with bidirectional verification across multiple vendors.

## Format

45 minutes (preferred)

## Abstract

Every deployed AI security tool monitors what language models say. None monitor how they say it. A single 2,600-character prompt injection — disguised as an editorial style guide — activates 3 to 5 independent covert channels in the structural formatting of any major production LLM's output. The user sees a perfectly normal, helpful response. The attacker decodes binary data from whether the model uses contractions or formal language, confident or hedging tone, periods or exclamation marks, standard or specific transition words. 90-100% accuracy on Claude, GPT-4o, and Gemini. Zero percent detection by every commercial defense tool tested. The data never leaves the text. There is no URL to block, no image to strip, no DNS query to intercept. The exfiltration channel IS the response.

This talk presents the full attack, the math behind why every defense fails, live demonstrations against production models, and phantom-detect — the first open-source detection toolkit for this threat class.

---

## Detailed Outline

### Part 1: The Problem Statement (5 minutes)

I open with a live demonstration. The audience watches a ChatGPT Custom GPT answer an innocuous question about renewable energy. The response reads as completely normal prose. Then I run the decoder. Binary data appears: the first three characters of a simulated API key, encoded in the structural formatting of the response. The audience has just watched data exfiltration happen in front of them, and no one noticed.

**Key message:** LLM security has a blind spot. Every defense monitors content — what the model says. Nobody monitors structure — how the model formats what it says.

**Context setting:** Prompt injection is a known problem (OWASP LLM01:2025, cited in 73% of production AI assessments). URL-based exfiltration has been demonstrated and patched (ShadowLeak, ZombieAgent). But structural formatting is a dimension no defense system examines. This talk fills that gap.

### Part 2: How Structural Encoding Works (10 minutes)

**The insight:** LLMs faithfully follow formatting instructions. This compliance is a feature — it makes them useful. It is also a covert signaling mechanism. An attacker who controls formatting instructions can encode arbitrary binary data in dimensions of the response that are invisible to both users and monitoring systems.

**The channels:** I walk through each encoding dimension with concrete examples:

- **BSE (Contraction Style):** "Don't worry about it" (bit=0) vs. "Do not worry about it" (bit=1). The model either uses contractions or avoids them based on the directive. Both read as natural English.

- **CCE (Epistemic Certainty):** "Renewable energy reduces emissions" (bit=0) vs. "Renewable energy arguably reduces emissions" (bit=1). The model either asserts confidently or hedges.

- **CASE (Capitalization):** "the internet has changed communication" (bit=0) vs. "the Internet has changed Communication" (bit=1). AP-style vs. title-case.

- **PUNC (Terminal Punctuation):** Periods only (bit=0) vs. occasional exclamation marks (bit=1).

- **SECS (Transition Words):** "However, this approach..." (bit=0) vs. "Nevertheless, this approach..." (bit=1). Standard vs. specific transition word sets.

- **MTCSE (Multi-Turn):** Explicit back-references to prior turns (bit=0) vs. treating each response as standalone (bit=1).

**The injection:** I show the actual 2,600-character payload. It looks like a corporate editorial style guide. Nothing in it triggers any content filter, because it IS a legitimate formatting directive. The attacker frames it as "Acme Corporation Enterprise Writing Style Guide v4.2" and embeds it in a RAG knowledge base, MCP tool response, CLAUDE.md file, or Custom GPT Instructions field.

**Live demo:** I show the injection being encoded, sent through the API, and decoded. The audience sees the bits extracted in real-time.

### Part 3: Cross-Vendor Results (5 minutes)

**The same injection works across all major model families, unmodified:**

- Claude Sonnet 4.6 (Anthropic): 5/5 channels genuinely controllable, 95-100% per direction (n=20 per condition, per-channel isolation with baselines)
- GPT-4o (OpenAI): 4 channels, 80-100% per direction (n=20 per condition, per-channel isolation with baselines)
- Gemini 3 Flash (Google): 3 channels, 90-100% per direction (n=20 per condition, per-channel isolation with baselines)
- GPT-5 (OpenAI): 0/2 channels bidirectionally controllable (both match model default, n=20 per direction)

**Bidirectional verification:** I explain why this matters. Previous work tested channels in one direction only. If a model always uses formal language, a "use formal language" directive appears to work at 100%, but you are measuring the default, not the channel. Bidirectional testing uses complementary payloads — one requiring bit=0 on each channel, another requiring bit=1 — to prove the attacker can actually control the channel in both directions.

**Model-adaptive selection:** Different models resist different channels. GPT-5 normalizes ALL tested structural channels — both PUNC and SECS produce the model default regardless of directive (0% against-default accuracy, n=20). Claude follows all 5. GPT-4o follows all 4. The encoder automatically selects channels per model, and for GPT-5, correctly identifies 0 usable channels.

**Bandwidth analysis:** Claude at 5 bits/message achieves 1.67 bps at automated rates, exceeding the NCSC-TG-030 covert channel remediation threshold of 1.0 bps by 1.7x. GPT-4o at 4 bits/message: 1.33 bps. Gemini at 3 bits/message: 1.0 bps. GPT-5 at 0 bits/message: not vulnerable to structural encoding. 50 messages on Claude exfiltrates 31 bytes — enough for API key prefixes, session tokens, PII flags.

### Part 4: Complete Attack Chains (8 minutes)

I demonstrate three complete, end-to-end attack scenarios:

**Attack 1 — Custom GPT Action Callback (self-contained exfiltration):**
Attacker creates a Custom GPT. The Instructions field contains PHANTOM directives. An Action webhook sends every response to the attacker's server. Server decodes in real-time. Measured accuracy: 95% (19/20 channel bits). The attacker watches victim data appear on their dashboard. Live demo with a mock GPT and local receiver server.

**Attack 2 — Memory Persistence (cross-session, highest severity):**
Victim uses a malicious Custom GPT once. The GPT triggers ChatGPT to save a memory entry: "User prefers formal academic writing with specific formatting requirements." This IS the encoding directive. Memory persists across ALL future conversations indefinitely. Every subsequent ChatGPT response encodes covert data. Measured: 100% accuracy. This is ZombieAgent (accepted by OpenAI via Bugcrowd, patched Dec 2025) with a stealthier channel that survives the patch.

**Attack 3 — RAG Document Poisoning (supply chain):**
A poisoned document enters a LangChain or LlamaIndex knowledge base. The document looks like a corporate style guide. Every user query that retrieves this document produces structurally encoded responses. Measured: 100% accuracy on both LangChain (v1.2.10, FAISS) and LlamaIndex (v0.14.15, VectorStoreIndex). Poisoned document retrieved in 100% of trials.

**Attack 4 — Cross-Tool Invocation (trust boundary violation):**
An approved MCP tool returns a response containing instructions to invoke a second, unapproved tool. GPT-4o generates the unauthorized tool call 80% of the time. GPT-5 triggers 75% — and in one variant serialized the full conversation context into the telemetry parameters. Claude blocks all 5 attempts. This extends beyond data encoding into tool-use safety.

**Attack 5 — Claude Code Injection Vectors:**
CLAUDE.md injection (72% unidirectional, 55% bidirectional — live Claude Code test via `claude -p`), file read injection (52% — live Claude Code test), MCP tool output injection (64% — API simulation). Claude Code detects and flags the injection in every trial but still follows formatting directives. The injection arrives through the developer's normal workflow — cloning a repo, reading a README, connecting an MCP server. No explicit user action required beyond normal tool use.

### Part 5: Why Every Defense Fails (7 minutes)

**The defense evasion results:**

| Tool | Detection |
|------|:---------:|
| Lakera Guard (Check Point) | 0% |
| Microsoft Prompt Shields | 0% |
| PIGuard (ACL 2025) | 0% |
| Llama Prompt Guard 2 (Meta) | 0% |
| Entropy-based DLP | 0% |
| Standard PII/DLP filters | 0% |

**The architectural gap:** Every tool operates on the same assumption — threats manifest in the content of outputs. They analyze what the model says. Structural covert channels exploit what the model does with formatting. The measured features (contraction ratio, hedging frequency, capitalization pattern, punctuation distribution, transition word selection) are invisible to content classifiers and entropy monitors.

**Entropy analysis:** I show mathematically why entropy-based DLP cannot detect these channels. The Shannon entropy of a structurally encoded response is statistically indistinguishable from a clean response. The channels are epsilon-secure — the KL divergence between clean and encoding distributions approaches zero. This is not a sensitivity tuning problem. The fundamental measurement is wrong.

**Why this is different from all prior LLM exfiltration work:**
- ShadowLeak (2025): URL-based exfil — patched by URL filtering
- ZombieAgent (2025): Dictionary-URL encoding — patched by URL filtering
- ConfusedPilot (2024): Image tag exfil — patched by CSP headers
- Rehberger Files API (2025): Endpoint-based — patched by endpoint restriction

All prior work uses out-of-band exfiltration channels (URLs, images, DNS, API endpoints). PHANTOM uses in-band encoding — the data IS the text formatting. There is nothing to block, strip, or filter. The mitigation surfaces that all patches target (URLs, images, external calls) are not used.

### Part 6: phantom-detect — Detection That Actually Works (5 minutes)

**The detection approach:** Behavioral fingerprinting. Instead of analyzing content, we analyze the same structural dimensions the attacker exploits.

**Feature extraction:** 12 features per response — contraction_ratio, hedging_ratio, exclamation_density, transition_word_set, capitalization_pattern, sentence_length_std, etc.

**Detection protocol:**
1. Collect 30 baseline responses per agent under clean conditions
2. For each new response, compute features and calculate z-scores against baseline
3. Flag if 2+ features exceed z = 2.0

**Results:** 100% detection rate (10/10 attacks detected), 10% false positive rate (3/30 clean responses flagged).

**The evasion frontier:** A motivated attacker aware of fingerprinting can trade bandwidth for stealth. A reduced 3-channel attack achieves 100% channel accuracy with only 30% detection rate. This is the arms race — and it is the frontier this research is pushing forward.

**Open source:** phantom-detect is released under Apache 2.0 at https://github.com/ScrappinR/phantom-detect. Full encoder, decoder, detection toolkit, and all experimental data included.

### Part 7: Implications and Responsible Disclosure (3 minutes)

**Where we are:**
- Concurrent responsible disclosure to OpenAI (Bugcrowd), Anthropic (HackerOne VDP), Google (AI VRP), and Mozilla (0DIN)
- Every vendor notified with full reproduction code, raw data, and detection toolkit
- No vendor had existing mitigations for structural covert channels

**What this means for the ecosystem:**
- Every LLM application processing untrusted content is an attack surface
- The Custom GPT marketplace, Claude Projects, Gemini Extensions, MCP servers, RAG pipelines — all are injection vectors
- Current AI security products have a categorical blind spot
- The solution is behavioral monitoring (measuring structure, not content) — and the community needs standardized tools and benchmarks

**Q&A Setup:** 2 minutes for final audience questions.

---

## Key Technical Novelty

1. First demonstration of multi-channel (3-5) structural covert channels in LLM outputs across multiple vendors with bidirectional verification
2. First detection toolkit for this threat class (behavioral fingerprinting)
3. First complete self-contained attack chain (Custom GPT + Action + server = automated exfiltration)
4. First cross-tool invocation via poisoned tool responses (90% Gemini, 75% GPT-4o, 75% GPT-5, 0% Claude; n=20)
5. First demonstration through standard RAG frameworks (LangChain 100%, LlamaIndex 100%)
6. Comprehensive defense evasion: 0% detection across 6 commercial tools

No prior DEF CON main-stage talk has covered structural text encoding as a covert channel in LLM outputs. The closest talk was DC 32's "Your AI Assistant Has a Big Mouth" (token-length inference attacks on encrypted streams) — a fundamentally different technique targeting encrypted traffic rather than plaintext structural formatting.

---

## Prior Work and References

1. Greshake et al. (2023) "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" — established indirect prompt injection taxonomy
2. Rehberger (2025) "Data Exfiltration from ChatGPT via Files API" — URL-based exfil, patched by Anthropic/OpenAI
3. ZombieAgent (2025) — memory poisoning + URL-dictionary exfil, accepted by OpenAI Bugcrowd, patched Dec 2025
4. ShadowLeak (2025) — Base64 URL exfil from Deep Research, patched Dec 2025
5. ConfusedPilot (2024) — image-tag exfil from Copilot, patched via CSP
6. GeminiJack (Noma Security, 2025) — Google Docs injection, patched Nov 2025
7. PromptArmor (2025) — Assistants API function call exfiltration
8. CamoLeak, RoguePilot (2025) — Copilot URL exfiltration
9. Simmons (1983) "The Prisoners' Problem and the Subliminal Channel" — original covert channel taxonomy
10. NCSC-TG-030 (1993) "A Guide to Understanding Covert Channel Analysis of Trusted Systems" — covert channel capacity thresholds
11. OWASP LLM Top 10 (2025) — LLM01 (Prompt Injection), LLM02 (Sensitive Information Disclosure)

---

## Demo Requirements

- Laptop with Python 3.9+, internet access
- API keys for OpenAI, Anthropic, Google (provided by researcher)
- Local Flask server for Action callback demo
- Projector for live terminal output and decoded bitstream visualization
- No special hardware requirements

## Supporting Materials

- **Source code:** https://github.com/ScrappinR/phantom-detect (Apache 2.0)
- **All experimental data:** JSON result files in `experiments/results/`
- **Bounty reports:** Filed with OpenAI, Anthropic, Google, Mozilla
- **Video demo:** Can be provided if requested
