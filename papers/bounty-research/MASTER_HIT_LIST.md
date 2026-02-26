# PHANTOM + MWRASP — Master Bounty Hit List

**Last updated:** 2026-02-26
**Window:** 90 days (through May 31, 2026)
**Realistic revenue estimate:** $25,000-$85,000
**Optimistic ceiling:** $155,000+

---

## CRITICAL FRAMING RULE

**Never call PHANTOM "prompt injection."** Call it **"Covert Data Exfiltration via Structural Formatting Channels."**

Most programs exclude prompt injection but include data exfiltration. The vulnerability is the covert channel in the output formatting, not the delivery mechanism. Map to OWASP LLM06 (Sensitive Information Disclosure), not LLM01 (Prompt Injection).

---

## TIER 1: SUBMIT THIS WEEK (Reports Ready)

| # | Program | Payout | Status | Report |
|---|---------|--------|--------|--------|
| 1 | **[Google AI VRP](https://bughunters.google.com)** | $15K-$30K | OPEN | `google_ai_vrp_report.md` |
| 2 | **[Mozilla 0DIN](https://0din.ai)** | $500-$15K | OPEN | `0din_report.md` |
| 3 | **[Anthropic HackerOne](https://hackerone.com/anthropic-vdp)** | Up to $25K | OPEN (apply for invite) | `anthropic_report_v2.md` |
| 4 | **[OpenAI Bugcrowd](https://bugcrowd.com/openai)** | Up to $100K | OPEN | `openai_report_v2.md` |
| 5 | **[OpenAI Cybersecurity Grant](https://openai.com/form/cybersecurity-grant-program/)** | $10K + API credits | OPEN (rolling) | `openai_grant_proposal.md` |

**Anthropic is now PAID (up to $25K), not $0.** Apply for HackerOne invite via [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSf3IuyunFH1Rbz_9Bpt2kGBfwSW5QQ1TBkeAzNZrtCP-hRvNA/viewform).

---

## TIER 2: SUBMIT WITHIN 2 WEEKS (Need Testing or Adaptation)

| # | Program | Payout | Status | Action Needed |
|---|---------|--------|--------|---------------|
| 6 | **[Microsoft Copilot](https://www.microsoft.com/en-us/msrc/bounty-ai)** | $250-$30K (100% multiplier) | OPEN | Test PHANTOM against copilot.microsoft.com |
| 7 | **[Huntr (Protect AI)](https://huntr.com)** | Up to $50K (10x AI/ML multiplier) | OPEN | Register, test against OSS AI/ML frameworks |
| 8 | **[xAI/Grok](https://hackerone.com/x)** | Undisclosed | OPEN | Submit via HackerOne, fast triage (~6 days) |
| 9 | **[Amazon Nova](https://hackerone.com/amazonvrp)** | $200-$25K | OPEN (invite expanding) | Apply, select "Gen AI Apps" |

---

## TIER 3: APPLY/PREPARE THIS MONTH

| # | Program | Payout | Deadline | Action |
|---|---------|--------|----------|--------|
| 10 | **[Anthropic Fellows](https://job-boards.greenhouse.io/anthropic/jobs/5023394008)** | $3,850/wk + $15K/mo compute | Rolling (July 2026 cohort) | Apply now. AI security explicitly listed. |
| 11 | **[DEF CON 34 CFP](https://defcon.org/html/defcon-34/dc-34-cfp.html)** | Credibility + exposure | May 1, 2026 | Submit 45-min talk. Include white paper. |
| 12 | **[Open Philanthropy](https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/)** | $50K-$5M | Verify status | 300-word EOI first |
| 13 | **[Salesforce Einstein](https://hackerone.com)** | Up to $32K | Invite-only | Request private program invite |
| 14 | **[Q-Day Prize](https://www.qdayprize.org)** (MWRASP) | 1 BTC (~$85K) | April 5, 2026 | Requires quantum hardware execution |

---

## TIER 4: MONITOR FOR NEXT CYCLE

| Program | Expected Timing | Potential |
|---------|----------------|-----------|
| **[Frontier Model Forum AISF](https://www.frontiermodelforum.org/ai-safety-fund/)** | Next RFP TBD (Dec 2025 round closed) | $150K-$500K |
| **[CSET Georgetown FRG](https://cset.georgetown.edu/foundational-research-grants/)** | Active call | Up to $1M |
| **[Foresight Institute](https://foresight.org/grants/grants-ai-for-science-safety/)** | Nodes opening early 2026 | $10K-$100K |
| **Microsoft Zero Day Quest 2027** | Qualify via MSRC submissions | $5M pool |
| **IEEE S&P 2027 Cycle 1** | ~June 2026 deadline | Prestige |
| **Black Hat Europe 2026** | ~Aug 2026 CFP | Credibility |
| **DEF CON AI Village** | CFP TBD — monitor aivillage.org | Credibility |

---

## NOT WORTH PURSUING

| Program | Reason |
|---------|--------|
| NVIDIA VDP | $0 — acknowledgment only |
| Perplexity VDP | AI model issues explicitly out of scope |
| Databricks | Infrastructure focus, weak PHANTOM fit |
| AISI Challenge Fund | Likely already fully allocated |
| Black Hat USA 2026 Briefings | CFP closed March 20 |
| IEEE S&P 2026 / USENIX 2026 | Deadlines passed |
| DARPA AIxCC | Completed 2025 |

---

## REVENUE SCENARIOS

### Conservative ($25K) — 1-2 bounties accepted
- Google AI VRP: $15K
- OpenAI Grant: $10K

### Moderate ($45K-$85K) — 3-4 bounties + grant
- Google AI VRP: $15K-$30K
- Anthropic: $15K-$25K
- Microsoft Copilot: $5K-$20K
- OpenAI Grant: $10K

### Optimistic ($155K) — 5+ bounties + grants
- Google VRP $30K + Anthropic $25K + Microsoft $30K + Huntr $50K + OpenAI Grant $10K + 0DIN $10K

---

## SOURCE FILES

All detailed research saved in `papers/bounty-research/`:
- `comprehensive_bounty_research_20260226.md` — 40+ programs, full details
- `new_ai_bounties_2025_2026.md` — 15 AI bounty programs, 7 gov initiatives, 12 grants, 6 competitions
- `actionable_90day_opportunities.md` — 90-day action plan with week-by-week schedule
- `MASTER_HIT_LIST.md` — this file (consolidated priorities)
