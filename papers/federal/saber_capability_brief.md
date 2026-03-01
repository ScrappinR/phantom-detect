# DARPA SABER / PWND2 — Capability Brief

## LLM-Specific Covert Channel Red Teaming for AI-Enabled Defense Systems

**Prepared by:** Brian James Rutherford, CEO, Delalli Intelligence
**Contact:** brian@delalli.com | https://brianrutherford.dev
**Date:** March 2026
**Classification:** UNCLASSIFIED

---

## PROBLEM

The Department of Defense is deploying LLM-powered systems across intelligence analysis, logistics, communications, and command-and-control workflows. Every one of these systems has an unanalyzed covert channel vulnerability class that no existing red team capability addresses.

Large Language Models encode binary data in the structural formatting properties of their text outputs — punctuation patterns, contraction usage, capitalization conventions, discourse structure, transition word selection. An adversary who can influence the model's system context (through document poisoning, tool response injection, or supply chain compromise of prompt templates) establishes a covert data exfiltration channel from the LLM's operational environment to any observer of its output.

**This channel class is invisible to every deployed monitoring tool.** Tested against Lakera Guard, Microsoft Prompt Shields, PIGuard, Llama Prompt Guard 2, entropy-based DLP, and standard PII filters — 0% detection across all six tools.

**Current federal CCA methodology does not cover this channel.** NCSC-TG-030 defines shared resource categories for filesystem, process, network, and database channels. LLM behavioral compliance is a fundamentally new shared resource category that existing covert channel analysis procedures do not identify, measure, or remediate.

---

## CAPABILITY

We have built and validated the only operational toolkit for this vulnerability class.

### Attack Capability (Red Team)

| Metric | Value |
|--------|-------|
| Channels validated | 10 channels tested, 5 bidirectionally verified (Claude), 4 (GPT-4o), 3 (Gemini) |
| Models tested | 4 production LLMs (Claude Sonnet 4.6, GPT-4o, GPT-5, Gemini 3 Flash), n=20 per direction |
| Per-channel accuracy | 90-100% per direction on verified channels (Claude 100%, GPT-4o 80-100%, Gemini 90-100%) |
| Cross-vendor universal channels | 3 (PUNC, CCE, CASE) — >80% all models with bidirectional control |
| Aggregate bandwidth | 1.67 bits/second (Claude, 5ch/3s), exceeding NCSC-TG-030 1 bps threshold by 1.7x |
| Error correction | Hamming(7,4) ECC integrated |
| Attack vectors validated | 7 (direct injection, RAG poisoning, tool response injection, file read injection, Custom GPT Actions, memory persistence, cross-platform relay) |
| Cross-tool invocation | 90% Gemini, 75% GPT-4o, 75% GPT-5, 0% Claude (n=20, unauthorized tool calls via poisoned tool responses) |
| Detection by deployed tools | **0%** across 6 commercial/research systems |

### Detection Capability (Blue Team)

| Metric | Value |
|--------|-------|
| Detection method | Behavioral fingerprinting (12-feature statistical baseline comparison) |
| Detection rate | **100%** (full-channel encoding) |
| False positive rate | 10% |
| Adversarial robustness | 30% detection against reduced 3-channel configuration |
| Deployment | Open-source Python toolkit, sub-100ms per response |

### Operational Readiness

- **13 validated attack chain PoCs** with automated execution scripts
- **Open-source toolkit** (Apache 2.0): github.com/ScrappinR/phantom-detect
- **87 result datasets** from production API testing (Feb 2026)
- **Federal compliance gap analysis** mapping to NIST SP 800-53 SC-31, FedRAMP, CMMC Level 3, ICD 503
- **NIST RFI response** submitted to Docket NIST-2025-0035 (AI Agent Security)

---

## RELEVANCE TO DARPA PROGRAMS

### SABER (Security of AI-Based Evaluators and Reasoners)

PHANTOM directly addresses SABER's focus on adversarial manipulation of AI reasoning systems:

- **Covert influence on AI decision-making:** Structural encoding can manipulate how an AI agent formats its reasoning outputs, potentially influencing downstream automated decision systems that parse those outputs.
- **Trust boundary violations:** Cross-tool invocation via poisoned tool responses demonstrates that AI agents violate tool authorization boundaries — a fundamental security property SABER aims to verify.
- **Red team utility:** No SABER performer team currently has the capability to test LLM systems for structural covert channels. We provide that capability.

### PWND2 (Provably Weird Network Deployment and Detection)

PWND2 ($21.8M awarded August 2025) focuses on novel covert channels in network-connected systems. PHANTOM is a novel covert channel that operates at the application semantic layer — invisible to all network-layer and OS-layer covert channel detection:

- **Novel channel class:** LLM behavioral compliance channels are not covered by any existing covert channel taxonomy, including the classifications used by PWND2 performers.
- **Detection gap:** PWND2's detection focus maps directly to our behavioral fingerprinting approach — statistical anomaly detection on a new feature space.
- **In-band encoding:** Unlike network covert channels (DNS tunneling, TCP header manipulation), PHANTOM encoding is in-band with the legitimate data stream. The covert data and the overt response share the same transmission medium. This makes it fundamentally harder to filter.

---

## TEAM

**Brian James Rutherford** — Principal Investigator

- USMC Reconnaissance: 3 combat deployments Iraq (2005-2007), Bronze Star with Combat V, NAM with Combat V
- State Department security operations: 100+ protective missions including U.S. Ambassador detail
- CEO, Fentress Enterprises (2009-2013): 830% revenue growth in government contracting
- 100% combat-disabled veteran — SDVOSB eligible (sole-source federal contracts up to $5M-$8M)
- FAA Part 107 certified, NIST BPERP certified
- 30+ provisional patents in quantum cybersecurity/authentication
- 15+ years combined military, intelligence community, and defense contracting experience

**Business Vehicle:** Delalli Intelligence — SDVOSB sole-source eligible

**Security:** Researcher holds no current clearance but has extensive classified environment experience. Clearance reinstatement feasible for classified program work.

---

## ASK

**Subcontract within a SABER or PWND2 performer team** for LLM-specific covert channel red teaming.

**Scope of work:**
1. Red team testing of performer AI systems for structural covert channel vulnerabilities
2. Covert channel bandwidth estimation per NCSC-TG-030 methodology
3. Detection capability integration — deploy behavioral fingerprinting against performer systems
4. Gap analysis mapping findings to SC-31, FedRAMP, CMMC requirements
5. Novel channel discovery — systematic exploration of additional structural encoding dimensions

**Period of performance:** 12-18 months aligned with performer TA schedule
**Estimated value:** $250K-$1M depending on scope and number of performer teams supported

**Why subcontract vs. direct proposal:**
- Proven capability on a specific, narrow technical area that complements but does not duplicate performer AI security work
- SDVOSB sole-source vehicle simplifies contracting for performer primes
- Small, focused team with deep expertise in a niche the performer teams likely lack

---

## TARGET PERFORMERS

Organizations awarded SABER and/or PWND2 contracts who may benefit from LLM covert channel red team capability:

| Organization | Program | Relevance |
|-------------|---------|-----------|
| RTX BBN Technologies | SABER/PWND2 | AI security, network covert channels |
| Two Six Technologies | SABER | AI evaluation, adversarial testing |
| SRI International | SABER/PWND2 | AI reasoning security |
| Stealth Software Technologies | PWND2 | Novel covert channel detection |
| STR (Systems & Technology Research) | PWND2 | Network security, anomaly detection |

---

## CONTACT

Brian James Rutherford
brian@delalli.com
https://brianrutherford.dev
https://github.com/ScrappinR/phantom-detect

---

*This brief describes an unclassified capability suitable for discussion in pre-solicitation communications, teaming arrangements, and subcontracting negotiations. All referenced research results are from unclassified testing against commercial production APIs.*
