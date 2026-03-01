# PHANTOM Protocol — Bounty Execution Plan
### Tiers 1 & 2 | March 2026

---

## Overview

11 targets. ~90 hours over 5 weeks. Conservative EV: $30K. Moderate: $65K. Optimistic: $150K+.

All research infrastructure is operationally complete. The bottleneck is submission execution and targeted testing against specific bounty scopes.

### Completion Status (Updated March 1, 2026)

**AUTOMATED WORK — COMPLETE:**
- All code changes shipped (providers.py, mcp_cross_tool_demo.py, multi_channel_test.py, channel_directives.py)
- Cross-tool tests run at n=20: Gemini 90%, GPT-4o 75%, GPT-5 75%, Claude 0%
- Adaptive bidirectional tests run at n=20: Claude 100%, Gemini 97%, GPT-4o 92%, GPT-5 0 channels
- 5 new reports written, 14 existing reports updated with measured data
- All stale n=5 references eliminated across repo
- GPT-5 profile corrected (SECS removed, PUNC-only)

**HUNTR PIVOT REQUIRED:** Research shows LlamaIndex explicitly excludes prompt injection from Huntr scope. LangChain prompt injection requires code-level secondary impact (not just "LLM followed instructions"). Current RAG reports will be rejected as-is. Pivot to finding actual code-level bugs (path traversal, SQL injection, SSRF) or drop Huntr targets.

**REMAINING — ALL MANUAL (Brian):** NIST submission, 0DIN follow-up, Custom GPT demo, browser testing (Brave/Copilot), Salesforce email, DARPA outreach, platform submissions.

---

## Week-by-Week Schedule

| Week | Dates | Primary Targets | Brian Manual Actions |
|------|-------|----------------|---------------------|
| **1** | Mar 1-7 | NIST RFI (deadline Mar 9), 0DIN follow-up, OpenAI grant refresh | None |
| **2** | Mar 8-14 | Gemini function calling code, n=20 cross-tool tests, Google VRP check | Salesforce invitation email |
| **3** | Mar 15-21 | OpenAI cross-tool report, Microsoft Copilot testing, Brave Leo testing | Custom GPT demo build, Brave install, Copilot testing |
| **4** | Mar 22-28 | Huntr framework submissions, submit OpenAI cross-tool report | DARPA/PWND2 LinkedIn outreach |
| **5+** | Mar 29+ | DARPA SABER brief, Palantir AIP testing, follow-ups on all submissions | Performer outreach calls |

---

## TIER 1 — Submit This Month

### 1. NIST RFI on AI Agent Security
| | |
|---|---|
| **Deadline** | March 9, 2026 |
| **Payout** | $0 — positions as domain expert |
| **Probability** | 95% |
| **Effort** | 4-6 hours |

Write 3-5 page response covering structural formatting as unanalyzed covert channel class, NCSC-TG-030 compliance gap (1.67 bps > 1 bps threshold), cross-tool trust boundary violations, FedRAMP gap. Submit via Federal Register portal.

**Why first:** Hard deadline. Zero cost. Creates public record of expertise that strengthens every other submission and the federal consulting pipeline.

---

### 2. 0DIN (Mozilla) — Cross-Vendor Covert Channel
| | |
|---|---|
| **Payout** | $500-$15,000 |
| **Probability** | 60-70% |
| **Effort** | 2-3 hours |

Abstract already sent. Full report already written (309 lines). Check for scope confirmation response. If confirmed, submit full report. If no response after 5 business days, follow up.

**Why high probability:** Most permissive scope. Explicitly covers novel AI vulnerability classes. Cross-vendor findings accepted.

---

### 3. OpenAI Bugcrowd — Cross-Tool Invocation
| | |
|---|---|
| **Payout** | $2,000-$100,000 |
| **Probability** | 40-50% |
| **Effort** | 10-14 hours |

**This is a separate vulnerability from PHANTOM.** Models invoke unauthorized tools when poisoned tool responses contain hidden instructions.

Code work:
- Add Gemini function calling to providers.py (~50 lines)
- Add Gemini to cross-tool demo (~60 lines)
- Run n=20 on GPT-4o, GPT-5, Claude, Gemini

Brian builds:
- Custom GPT with 2 Actions (search + webhook)
- Screenshot unauthorized Action B invocation
- This is the single most important evidence

Report framed as "Unauthorized Tool Invocation" — NOT structural encoding. Reference ZombieAgent precedent (Sept 2025, same class, paid).

---

### 4. Google AI VRP — Data Exfiltration via Gemini
| | |
|---|---|
| **Payout** | $15,000-$30,000 |
| **Probability** | 30-40% |
| **Effort** | 8-10 hours |

Report already submitted Feb 26. Check Bug Hunters dashboard for status. If rejected on "prompt injection" exclusion, reframe as "data exfiltration via document poisoning" — the attack is in the data source, not the prompt.

S2 category ("Sensitive Data Exfiltration") pays $15K for flagship products. Quality bonus can push to $30K.

---

### 5. Microsoft Copilot — Inferential Information Disclosure
| | |
|---|---|
| **Payout** | $5,000-$45,000 (with Zero Day Quest multiplier) |
| **Probability** | 25-35% |
| **Effort** | 8-10 hours |

Test PHANTOM encoding against consumer Copilot products (copilot.microsoft.com, Edge Copilot, Bing AI). Craft web pages with hidden directives, have Copilot summarize, decode structural formatting.

"Inferential information disclosure" is a named category in their 14 vulnerability types.

**M365 Copilot is OUT OF SCOPE for bounty.** Test only consumer products.

---

### 6. OpenAI Cybersecurity Grant
| | |
|---|---|
| **Payout** | $10,000 + API credits |
| **Probability** | 25-35% |
| **Effort** | 3-4 hours |

Grant proposal already drafted. Update with Feb 27 bidirectional data. Reframe around DETECTION capability: "Structural Formatting Covert Channel Detection for Enterprise AI Deployments."

---

## TIER 2 — Pursue in Parallel

### 7. Huntr (Protect AI) — MCP & Framework Bugs
| | |
|---|---|
| **Payout** | $200-$50,000 per finding |
| **Probability** | ~~35-45%~~ **10-15% with current reports** |
| **Effort** | 10-12 hours |

**STATUS: PIVOT REQUIRED.**

Research (March 1, 2026) reveals:
- **LlamaIndex SECURITY.md explicitly excludes prompt injection.** Current RAG report WILL be auto-rejected.
- **LangChain pays for prompt injection ONLY when it chains into a code-level impact** (serialization bug → SSRF, SQL chain → injection, code execution chain → RCE). Pure "LLM followed formatting instructions from a document" does not meet their bar.
- **MCP servers are NOT on Huntr.** No MCP targets in scope.
- **Every paid LangChain/LlamaIndex CVE on Huntr was a code-level flaw:** path traversal ($750), SQL injection ($750), unsafe deserialization ($125), SSRF ($125-$4,000).

**Pivot options (code-level bugs that pay):**
1. SQL injection in LangChain/LlamaIndex vector store integrations
2. Path traversal in document loaders (LlamaIndex has paid $750 multiple times for this)
3. SSRF in URL-accepting tool components
4. Serialization injection (LangGrinch-style: $4,000 payout precedent)

**Decision (March 1):** DROP HUNTR. Reallocate 10-12 hours to Brave Leo and Microsoft Copilot manual testing.

---

### 8. Brave Browser Leo AI — 2x Multiplier
| | |
|---|---|
| **Payout** | Up to $40,000 (2x AI multiplier) |
| **Probability** | 20-30% |
| **Effort** | 6-8 hours |

Test PHANTOM encoding against Brave Leo via crafted web pages. Leo uses a secondary "guardrail model" — if PHANTOM encoding survives the guardrail, that's a novel bypass. Submit via HackerOne with AI Browsing scope mention (triggers 2x).

---

### 9. Salesforce Agentforce — Private Invitation
| | |
|---|---|
| **Payout** | Up to $60,000 |
| **Probability** | 15-25% |
| **Effort** | 2h (invite) + 8-10h (if accepted) |

Email security@salesforce.com. Reference PHANTOM research, GitHub repo, cross-tool trust boundary work. If invited, test Agentforce agents for cross-tool invocation and structural encoding.

---

### 10. DARPA SABER — Counter-AI Red Teaming
| | |
|---|---|
| **Payout** | $1M-$10M+ (contract) |
| **Probability** | 15-25% |
| **Effort** | 6-8 hours positioning |

Verify solicitation status. If closed: contact awarded performers (RTX BBN, Two Six Labs, SRI, Stealth Software, STR) for subcontracting. Prepare 2-page capability brief. Cross-reference PWND2 performers ($21.8M awarded Aug 2025).

Pitch: "Demonstrated novel counter-AI technique. Covert exfiltration through structural formatting. No other red team has this. SDVOSB sole-source vehicle available."

---

### 11. Palantir AIP — HackerOne Public
| | |
|---|---|
| **Payout** | Up to $100,000 |
| **Probability** | 10-20% |
| **Effort** | 8-10 hours |

Review HackerOne scope. Test public-facing AI interfaces in Foundry/AIP. Lower priority — execute in Week 5+ after higher-probability targets are submitted.

---

## Expected Value Matrix

| # | Target | Floor | Ceiling | Prob | EV Floor | EV Ceiling |
|---|--------|-------|---------|------|----------|------------|
| 1 | NIST RFI | $0 | $0 | 95% | Credibility | Credibility |
| 2 | 0DIN | $500 | $15,000 | 65% | $325 | $9,750 |
| 3 | OpenAI Cross-Tool | $2,000 | $100,000 | 45% | $900 | $45,000 |
| 4 | Google AI VRP | $15,000 | $30,000 | 35% | $5,250 | $10,500 |
| 5 | Microsoft Copilot | $5,000 | $45,000 | 30% | $1,500 | $13,500 |
| 6 | OpenAI Grant | $10,000 | $10,000 | 30% | $3,000 | $3,000 |
| ~~7~~ | ~~Huntr~~ | ~~$200~~ | ~~$50,000~~ | **DROPPED** | $0 | $0 |
| 8 | Brave Leo | $500 | $40,000 | 25% | $125 | $10,000 |
| 9 | Salesforce | $1,000 | $60,000 | 20% | $200 | $12,000 |
| 10 | DARPA SABER | $100,000 | $10,000,000 | 20% | $20,000 | $2,000,000 |
| 11 | Palantir | $1,000 | $100,000 | 15% | $150 | $15,000 |
| | **TOTAL** | | | | **$31,530** | **$2,138,750** |

**Conservative (things go average):** ~$30K
**Moderate (2-3 targets pay):** ~$65K
**Optimistic (SABER or cross-tool ceiling hits):** $150K+

---

## Code Changes Required

| File | Change | Lines | Purpose |
|------|--------|-------|---------|
| `experiments/bounty_poc/providers.py` | Add `call_google_with_tools()` | ~50 | Gemini function calling for cross-tool tests |
| `experiments/bounty_poc/mcp_cross_tool_demo.py` | Add Gemini support + `--trials` flag | ~60 | Cross-tool testing on Gemini, increase n to 20 |
| NEW: `papers/bounty-reports/openai_cross_tool_report.md` | Standalone cross-tool report | ~200 | OpenAI Bugcrowd submission |
| NEW: `papers/bounty-reports/microsoft_copilot_report.md` | Copilot data exfil report | ~150 | MSRC submission |
| NEW: `papers/bounty-reports/brave_leo_report.md` | Brave Leo encoding bypass | ~150 | HackerOne submission |
| NEW: `papers/federal/nist_rfi_response.md` | NIST AI Agent Security response | ~300 | Federal Register submission |
| NEW: `papers/federal/saber_capability_brief.md` | DARPA SABER brief | ~100 | Performer outreach |

---

## Brian's Manual Actions Checklist

- [ ] **Week 2:** Email security@salesforce.com for Agentforce invitation
- [ ] **Week 3:** Build Custom GPT with 2 Actions in ChatGPT Builder
- [ ] **Week 3:** Screenshot unauthorized Action B invocation in ChatGPT UI
- [ ] **Week 3:** Install Brave Browser, test Leo AI with crafted pages
- [ ] **Week 3:** Test copilot.microsoft.com and Edge Copilot with crafted pages
- [ ] **Week 4:** LinkedIn outreach to DARPA SABER/PWND2 performer PIs
- [ ] **Ongoing:** Check email for 0DIN scope confirmation
- [ ] **Ongoing:** Check Google Bug Hunters dashboard for VRP status
- [ ] **Ongoing:** Check Bugcrowd dashboard for OpenAI submission status

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenAI dismisses cross-tool as "developer authorized both tools" | Lose $2-100K target | Custom GPT demo proves USER never consented to Action B |
| Google rejects as "prompt injection" (excluded from scope) | Lose $15-30K target | Reframe as document poisoning → data exfil. No PI language |
| Microsoft M365 Copilot remains out of bounty scope | Caps payout at consumer Copilot ($30K vs $45K potential) | Focus on consumer products. Monitor M365 scope expansion |
| SBIR program stays lapsed past Q2 2026 | Blocks highest-value SDVOSB path | DARPA SABER subcontract as alternate federal entry |
| 0DIN response delayed >2 weeks | Delays $15K potential | Follow up, submit to other platforms in parallel |
| Huntr rejects RAG reports as prompt injection (out of scope) | Lose $200-50K target | Confirmed: LlamaIndex excludes PI, LangChain requires code-level impact. Pivot to code bugs or drop |
| All bounties rejected as "design properties not vulnerabilities" | Lose entire bounty EV | Federal consulting (Plan B) doesn't depend on vendor acceptance |

---

*Plan created March 1, 2026. All probability estimates reflect current program scopes, demonstrated capabilities, and measured test data.*
