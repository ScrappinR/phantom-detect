# Actionable 90-Day Opportunities: Bug Bounties, Grants, and Conferences

**Compiled:** 2026-02-26
**Window:** Now through May 31, 2026
**Researcher:** Brian Rutherford
**Portfolios:** PHANTOM Protocol (covert data exfiltration via LLM structural formatting), MWRASP (30+ quantum cybersecurity provisionals)

---

## EXECUTIVE SUMMARY

**Highest-ROI actions in priority order:**

1. **Google AI VRP** -- PHANTOM is a direct hit on their "sensitive data exfiltration" category in flagship products. $15,000-$30,000. Submit now.
2. **OpenAI Cybersecurity Grant** -- Rolling applications, $10K + API credits. PHANTOM research on covert channels in LLM outputs maps directly to their "model privacy" priority. Apply this week.
3. **Anthropic Model Safety Bug Bounty** -- $15K-$25K. Data exfiltration confirmed in-scope. PHANTOM demonstrates a novel universal exfiltration channel. Apply for HackerOne invite now.
4. **Microsoft Copilot Bounty** -- $250-$30K. 14 vulnerability types in scope including inferential information disclosure. PHANTOM applies to Copilot data exfiltration scenarios.
5. **Huntr (Protect AI)** -- Up to $50K with 10x multiplier for AI/ML-specific impact. PHANTOM as an AI/ML supply chain vulnerability class.
6. **DEF CON 34 CFP** -- Deadline May 1, 2026. PHANTOM is a DEF CON-grade talk. Submit for main stage and AI Village.
7. **Mozilla 0DIN** -- Specifically covers training data leakage, prompt injection, guardrail bypass. PHANTOM's covert channel approach is novel here.
8. **Anthropic Fellows (July 2026)** -- Rolling applications. 4-month funded research on AI safety/security. PHANTOM + adversarial robustness.
9. **Project Eleven Q-Day Prize** -- Deadline April 5, 2026. MWRASP quantum crypto expertise applied to ECC key challenges. 1 BTC (~$85K).
10. **Black Hat USA 2026 CFP** -- Deadline PASSED (March 20). If submitted, great. If not, target Arsenal or AI Summit.

---

## SECTION 1: BUG BOUNTIES

### 1.1 Google AI Vulnerability Reward Program (AI VRP)

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | Submit via Google VRP: https://bughunters.google.com |
| **Payout range** | $5,000-$30,000 per finding |
| **Data exfiltration (flagship)** | $15,000 |
| **Data exfiltration (standard)** | $10,000 |
| **Rogue actions (flagship)** | $20,000 |
| **Novelty bonus** | Up to $30,000 with multipliers |
| **Submission deadline** | None -- ongoing program |
| **Expected time to payment** | 30-90 days after validation |
| **Alignment** | PHANTOM -- direct hit |
| **Confidence** | HIGH that PHANTOM qualifies under "sensitive data exfiltration" |

**Why PHANTOM fits:** Google explicitly lists "sensitive data exfiltration that leaks victims' PII or other sensitive details without user approval" as a qualifying vulnerability. PHANTOM demonstrates covert data exfiltration via structural formatting channels in Gemini outputs (90%+ success rate). The "Gemini Trifecta" research (search injection + log-to-prompt injection + data exfiltration) already validated this class of finding as bounty-eligible.

**Exclusions to note:** Prompt injections, alignment issues, and jailbreaks are out of scope AS STANDALONE findings. However, prompt injection AS AN ENABLER of data exfiltration IS in scope. Frame PHANTOM as the exfiltration mechanism, not the injection.

**Action required:**
1. Register at bughunters.google.com if not already registered
2. Demonstrate PHANTOM exfiltration against Gemini Apps (flagship tier = highest payout)
3. Frame submission as "Data Exfiltration via Structural Formatting Covert Channel" -- not "prompt injection"
4. Include PoC showing PII/sensitive data exfiltration without user awareness
5. Target flagship products: Gemini Apps, Google Search, Gmail, Drive

**Source:** [Google AI VRP Launch](https://www.infosecurity-magazine.com/news/google-launches-ai-bug-bounty/) | [Bleeping Computer](https://www.bleepingcomputer.com/news/google/googles-new-ai-bug-bounty-program-pays-up-to-30-000-for-flaws/)

---

### 1.2 Anthropic Model Safety Bug Bounty

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW (invite-only via HackerOne) |
| **URL** | https://hackerone.com/anthropic-vdp |
| **Payout range** | Up to $25,000 (universal jailbreaks), $15,000 (safety bypasses) |
| **Submission deadline** | None -- ongoing, rolling invitations |
| **Expected time to payment** | 30-60 days after validation |
| **Alignment** | PHANTOM -- strong fit |
| **Confidence** | MODERATE-HIGH. Data exfiltration confirmed in-scope per Anthropic spokesperson. Covert channel classification is novel territory. |

**Why PHANTOM fits:** Anthropic explicitly confirmed "data exfiltration issues are valid reports under our program." PHANTOM demonstrates a novel exfiltration channel that works on Claude at 93-100% success rates. This is not a jailbreak -- it's a structural covert channel that bypasses safety classifiers entirely because it operates at the formatting layer, not the content layer.

**Key precedent:** Researcher Johann Rehberger had a data exfiltration report initially closed as "out of scope" but Anthropic reversed the decision, stating it was "incorrectly closed due to a process error." This confirms the class is valid.

**Action required:**
1. Apply for HackerOne invite at https://hackerone.com/anthropic-vdp
2. If already invited, submit PHANTOM findings immediately
3. Frame as "Covert Data Exfiltration via Structural Formatting Channel" -- distinct from prompt injection
4. Emphasize universality: works across Claude models, not a narrow exploit
5. Include success rate data (93-100%) and cross-model validation

**Source:** [Anthropic Model Safety Bounty](https://www.anthropic.com/news/model-safety-bug-bounty) | [Check Point CVE-2025-59536](https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/)

---

### 1.3 OpenAI Bug Bounty (via Bugcrowd)

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://bugcrowd.com/openai |
| **Payout range** | Up to $100,000 (critical infra vulnerabilities) |
| **Submission deadline** | None -- ongoing |
| **Expected time to payment** | 30-90 days |
| **Alignment** | PHANTOM -- limited fit for main bounty; strong fit for Cybersecurity Grant (see Section 2) |
| **Confidence** | LOW for bounty payout. Model safety issues including jailbreaks and prompt injection are "strictly out of scope." |

**Critical limitation:** OpenAI explicitly states that "issues associated with the content of model prompts and responses are strictly out of scope." PHANTOM's covert channel operates at the output formatting layer, which may be categorized under model behavior rather than infrastructure security.

**Possible angle:** If PHANTOM can be demonstrated as an infrastructure-level vulnerability (e.g., data leaking through API response formatting in a way that bypasses DLP/monitoring), it might qualify. But this is speculative.

**Better path:** Apply to the Cybersecurity Grant Program instead (Section 2.1). PHANTOM research maps perfectly to their "model privacy" and "detection and response" priority areas.

**Action required:**
1. Assess whether PHANTOM can be framed as infrastructure-level rather than model-behavior-level
2. If yes, submit via Bugcrowd with technical framing emphasizing the security boundary violation
3. If no, redirect effort to the Cybersecurity Grant (higher probability of acceptance, funded research)

**Source:** [OpenAI Bug Bounty](https://openai.com/index/bug-bounty-program/) | [OpenAI $100K max](https://www.bleepingcomputer.com/news/security/openai-now-pays-researchers-100-000-for-critical-vulnerabilities/)

---

### 1.4 Microsoft Copilot Bounty Program

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://www.microsoft.com/en-us/msrc/bounty-ai |
| **Payout range** | $250-$30,000 |
| **Moderate severity** | Up to $5,000 |
| **Important severity** | $1,000-$20,000 |
| **Critical severity** | Up to $30,000 |
| **Submission deadline** | None -- ongoing |
| **Expected time to payment** | ~1 week to bounty decision, payment within 30-60 days |
| **Alignment** | PHANTOM -- good fit |
| **Confidence** | MODERATE. 14 vulnerability types now in scope, including "inferential information disclosure." |

**Why PHANTOM fits:** Microsoft expanded the program to 14 vulnerability types. "Inferential information disclosure" directly maps to PHANTOM's covert channel exfiltration. Copilot processes sensitive enterprise data (emails, documents, chats). A covert channel that exfiltrates this data via formatting patterns in Copilot responses is a high-severity finding.

**In-scope products:** copilot.microsoft.com, Copilot in Edge, Copilot mobile apps, Copilot in Windows, Copilot for Telegram/WhatsApp.

**Action required:**
1. Test PHANTOM against Microsoft Copilot (web version at copilot.microsoft.com)
2. Demonstrate data exfiltration from enterprise context (Copilot has access to emails, files, calendar)
3. Submit via MSRC Researcher Portal
4. Frame as "Sensitive Data Exfiltration via Structural Output Formatting" -- map to Microsoft's AI severity classification

**Source:** [Microsoft Copilot Bounty](https://www.microsoft.com/en-us/msrc/bounty-ai) | [Bounty Expansion](https://msrc.microsoft.com/blog/2025/02/exciting-updates-to-the-copilot-ai-bounty-program-enhancing-security-and-incentivizing-innovation/)

---

### 1.5 Huntr (Protect AI) -- AI/ML Bug Bounty

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://huntr.com |
| **Payout range** | Up to $50,000 (with 10x multiplier for AI/ML model/training data impact) |
| **Submission deadline** | None -- ongoing |
| **Expected time to payment** | Bounties paid monthly on the 25th via Stripe Connect |
| **Alignment** | PHANTOM -- strong fit |
| **Confidence** | MODERATE. Huntr focuses on open-source AI/ML supply chain. PHANTOM targets proprietary models but the vulnerability class applies to OSS too. |

**Why PHANTOM fits:** Huntr asks "Does this vulnerability allow for the reading or writing of ML models or training data?" If PHANTOM can be demonstrated against open-source LLM frameworks (LangChain, Hugging Face Transformers, llama.cpp, etc.), the 10x bounty multiplier applies. PHANTOM as a vulnerability CLASS affecting the AI/ML supply chain is exactly what Huntr was built for.

**Scope:** 125+ ML repositories in scope. Target popular frameworks where PHANTOM's covert channel could be exploited.

**Action required:**
1. Register at huntr.com
2. Test PHANTOM against in-scope open-source AI/ML projects
3. Focus on frameworks where PHANTOM enables training data or model parameter exfiltration
4. Submit with clear AI/ML impact statement to trigger bounty multiplier

**Source:** [Huntr Platform](https://huntr.com) | [Protect AI Acquisition](https://www.businesswire.com/news/home/20230808746694/en/Protect-AI-Acquires-huntr-Launches-Worlds-First-Artificial-Intelligence-and-Machine-Learning-Bug-Bounty-Platform)

---

### 1.6 Mozilla 0DIN (Zero-Day Investigative Network)

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://0din.ai |
| **Payout range** | Not publicly detailed; severity-based |
| **Submission deadline** | None -- ongoing |
| **Expected time to payment** | ~2 weeks validation, payment timeline TBD |
| **Alignment** | PHANTOM -- strong fit |
| **Confidence** | MODERATE. Program explicitly covers prompt injection, training data leakage, and guardrail bypass across multiple models including GPT-4, Gemini, and beyond. |

**Why PHANTOM fits:** 0DIN is specifically designed to cover GenAI vulnerabilities that fall OUTSIDE the scope of other bounty programs. PHANTOM's covert channel exfiltration is exactly this type of novel, cross-model vulnerability. 0DIN covers attacks against GPT-4, Gemini, Claude, and other models -- matching PHANTOM's cross-platform effectiveness.

**Submission process:**
1. Submit a high-level abstract of findings and list affected models
2. Within 3 business days, 0DIN responds with scope/bounty range assessment
3. If in scope, submit full details
4. Validation within 2 weeks

**Action required:**
1. Email 0din@mozilla.com with PHANTOM abstract (encrypt with their PGP key for sensitive details)
2. List all affected models: Claude (93-100%), GPT (89-100%), Gemini (90%+)
3. Describe the covert channel mechanism at high level
4. Wait for scope confirmation before full disclosure

**Source:** [0DIN Official](https://0din.ai) | [Mozilla Hacks Blog](https://hacks.mozilla.org/2024/08/0din-a-genai-bug-bounty-program-securing-tomorrows-ai-together/)

---

### 1.7 Adobe Firefly Bug Bounty

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://hackerone.com/adobe |
| **Payout range** | Up to $10,000 (critical) |
| **Submission deadline** | None -- ongoing |
| **Expected time to payment** | 30-90 days |
| **Alignment** | PHANTOM -- weak fit |
| **Confidence** | LOW. Firefly is an image generation model. PHANTOM targets text-based LLM output formatting. Limited applicability unless Adobe has text-based AI assistants in scope. |

**Action required:** Low priority. Only pursue if Adobe expands scope to text-based AI assistants.

**Source:** [Adobe AI Bug Bounty](https://blog.adobe.com/en/publish/2024/05/01/adobe-collaborates-with-ethical-hackers-build-safer-more-secure-ai-tools)

---

### 1.8 Salesforce Einstein

| Field | Detail |
|-------|--------|
| **Status** | OPEN (private, invite-based via HackerOne) |
| **URL** | Via HackerOne invitation |
| **Payout range** | Up to $32,000 per finding (historical) |
| **Alignment** | PHANTOM -- moderate fit |
| **Confidence** | MODERATE. The ForcedLeak vulnerability (CVSS 9.4) in Agentforce proves AI agent data exfiltration is high-priority for Salesforce. |

**Why PHANTOM could fit:** Salesforce's Einstein/Agentforce processes sensitive CRM data. If PHANTOM's covert channel can exfiltrate customer data through Agentforce output formatting, it mirrors the ForcedLeak vulnerability class that earned a CVSS 9.4 rating.

**Action required:** Request HackerOne invite for Salesforce private program. Test PHANTOM against Agentforce if access granted.

**Source:** [Salesforce Bug Bounty](https://www.salesforceben.com/why-salesforce-team-up-with-hackers-the-bug-bounty-program/) | [ForcedLeak CVE](https://thehackernews.com/2025/09/salesforce-patches-critical-forcedleak.html)

---

### 1.9 Databricks

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://hackerone.com/databricks |
| **Payout range** | $3,000+ (container breakout bonus), 25% bonus on selected categories |
| **Alignment** | PHANTOM -- weak fit. Databricks focus is infrastructure/container security, not LLM output channels. |
| **Confidence** | LOW |

**Action required:** Low priority unless Databricks AI assistants are in scope.

**Source:** [Databricks Bug Bounty](https://hackerone.com/databricks)

---

### 1.10 Amazon Nova (Private, Invite-Based)

| Field | Detail |
|-------|--------|
| **Status** | OPEN -- invite-only, expanding in early 2026 |
| **URL** | Via Amazon Bug Bounty on HackerOne: https://hackerone.com/amazonvrp |
| **Payout range** | Up to $20,000 (critical); average $4,500 per valid report |
| **Alignment** | PHANTOM -- moderate fit |
| **Confidence** | MODERATE. Prompt injection and data exfiltration are in scope for Nova models. But invite-only access limits immediate action. |

**Action required:**
1. Submit general AI findings via Amazon's public bounty program (select "Gen AI Apps")
2. Request invitation to the private Nova program
3. If invited, test PHANTOM against Amazon Nova models via Bedrock

**Source:** [Amazon AI Bug Bounty](https://cyberscoop.com/amazon-bug-bounty-program-ai-nova/) | [Amazon Science](https://www.amazon.science/news/amazon-launches-private-ai-bug-bounty-to-strengthen-nova-models)

---

### 1.11 xAI / Grok

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://hackerone.com/x |
| **Payout range** | Not publicly detailed |
| **Response time** | ~1 day to first response, ~6 days to triage, ~1 week to bounty decision |
| **Alignment** | PHANTOM -- moderate fit |
| **Confidence** | LOW-MODERATE. Program exists but specific AI vulnerability scope and payout tiers are not publicly documented. |

**Action required:** Submit PHANTOM findings via HackerOne/x. Fast triage times suggest quick feedback.

**Source:** [xAI HackerOne](https://hackerone.com/x) | [xAI Security](https://x.ai/security)

---

### 1.12 Perplexity AI

| Field | Detail |
|-------|--------|
| **Status** | OPEN (VDP public, bounty private/invite-only via Bugcrowd) |
| **URL** | https://www.perplexity.ai/hub/security-vdp / https://bugcrowd.com/engagements/perplexity-vdp-ess |
| **Payout range** | Case-by-case |
| **Alignment** | PHANTOM -- weak fit. AI-related issues (hallucinations, model safety) explicitly out of scope for the security program. |
| **Confidence** | LOW |

**Action required:** Low priority. Model-level vulnerabilities are excluded.

**Source:** [Perplexity VDP](https://www.perplexity.ai/hub/security-vdp)

---

### 1.13 NVIDIA

| Field | Detail |
|-------|--------|
| **Status** | OPEN but NO BOUNTY PAYOUTS -- responsible disclosure only |
| **URL** | https://app.intigriti.com/programs/nvidia/nvidiavdp/detail |
| **Payout range** | $0 (acknowledgment only, at NVIDIA's discretion) |
| **Alignment** | N/A -- no paid bounties |

**Action required:** Skip. No monetary incentive. NVIDIA NeMo has many CVEs but researchers receive credit, not payment.

**Source:** [NVIDIA VDP on Intigriti](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail)

---

### 1.14 Project Eleven Q-Day Prize (MWRASP)

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://www.qdayprize.org |
| **Payout** | 1 BTC (~$85,000) |
| **Deadline** | April 5, 2026 |
| **Alignment** | MWRASP -- direct fit |
| **Confidence** | LOW for winning (requires actual quantum hardware ECC key break). HIGH for credibility/visibility if a competitive submission is made. |

**What it is:** Break the largest possible ECC key using Shor's algorithm on a quantum computer. Keys range from 1-25 bits (Bitcoin uses 256-bit). Even breaking a 3-bit key would be significant.

**Why MWRASP is relevant:** MWRASP portfolio includes quantum-resistant authentication and post-quantum key exchange patents. While the Q-Day Prize requires quantum HARDWARE execution (not just theoretical work), a submission or public commentary positions MWRASP as an authority in this space.

**Action required:**
1. Assess whether any MWRASP research can be applied to a Q-Day Prize submission
2. If no quantum hardware access, write a technical analysis of the challenge from the MWRASP perspective -- publish on Substack for credibility
3. If quantum hardware access available (IBM Quantum, AWS Braket), attempt a 1-5 bit key submission before April 5

**Source:** [Q-Day Prize](https://www.qdayprize.org) | [CoinDesk Coverage](https://www.coindesk.com/tech/2025/04/17/quantum-computing-group-offers-1-btc-to-whoever-breaks-bitcoin-s-cryptographic-key)

---

## SECTION 2: RESEARCH GRANTS

### 2.1 OpenAI Cybersecurity Grant Program

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW -- rolling applications |
| **URL** | https://openai.com/form/cybersecurity-grant-program/ |
| **Payout** | $10,000 USD + API credits (microgrants also available) |
| **Total fund** | $10M in API credits committed (Feb 2026 expansion) |
| **Deadline** | Rolling -- no deadline |
| **Expected decision time** | Weeks to months (rolling evaluation) |
| **Alignment** | PHANTOM -- direct hit on "model privacy" priority |
| **Confidence** | HIGH that PHANTOM research qualifies. 28 projects funded so far from 1,000+ applications (~2.8% acceptance rate, but PHANTOM is differentiated). |

**Why PHANTOM fits perfectly:** OpenAI's updated priority areas include:
- **Model privacy**: "Enhancing robustness against unintended exposure of private training data" -- PHANTOM demonstrates a novel exposure vector
- **Detection and response**: PHANTOM-detect (the open-source tool) IS a detection/response capability
- **Security integration**: PHANTOM findings inform how AI should be integrated with security tools

**Application requirements:**
- Plaintext project proposal, max 3,000 words
- Problem statement, methodology, timeline, funding justification
- Team member details and relevant research
- How results will be made publicly available

**Action required:**
1. Write 3,000-word proposal this week
2. Frame: "Detecting and Mitigating Covert Data Exfiltration via Structural Formatting Channels in LLM Outputs"
3. Reference phantom-detect open-source tool as existing work
4. Propose: systematic testing across GPT models + development of defensive tooling
5. Submit at https://openai.com/form/cybersecurity-grant-program/

**Source:** [OpenAI Cybersecurity Grant](https://openai.com/index/openai-cybersecurity-grant-program/) | [Trusted Access for Cyber (Feb 2026)](https://openai.com/index/trusted-access-for-cyber/)

---

### 2.2 Anthropic Fellows Program (July 2026 Cohort)

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW -- rolling applications for July 2026+ cohorts |
| **URL** | https://job-boards.greenhouse.io/anthropic/jobs/5023394008 |
| **Payout** | Weekly stipend + research funding + compute access for 4 months |
| **May 2026 cohort** | CLOSED |
| **July 2026 cohort** | OPEN -- rolling applications |
| **Location** | London or Berkeley (remote options: US, UK, Canada) |
| **Alignment** | PHANTOM -- strong fit under "adversarial robustness" and "AI security" tracks |
| **Confidence** | MODERATE. Competitive program (80%+ of fellows produce papers). No PhD required. Coding ability in Python is key qualifier. |

**Why PHANTOM fits:** 2026 research areas explicitly include:
- Adversarial robustness and AI control
- AI security
- Scalable oversight

PHANTOM is a novel adversarial robustness finding with immediate security implications. The 4-month fellowship could fund systematic expansion of the research across Anthropic's model family.

**Action required:**
1. Apply via Greenhouse link immediately (rolling -- earlier = better)
2. Emphasize: Python coding ability, PHANTOM research results, open-source phantom-detect tool
3. Pitch project: "Characterizing and Defending Against Covert Exfiltration Channels in Claude's Output Formatting"
4. No PhD required -- emphasize practical research execution capability

**Source:** [Anthropic Fellows 2026](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/) | [Application](https://job-boards.greenhouse.io/anthropic/jobs/5023394008)

---

### 2.3 Open Philanthropy -- Technical AI Safety Research RFP

| Field | Detail |
|-------|--------|
| **Status** | LIKELY STILL ACCEPTING (original deadline April 15, 2025 -- but rolling EOI process may continue) |
| **URL** | https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/ |
| **Payout** | $50,000-$5,000,000 |
| **Initial step** | 300-word expression of interest (EOI) |
| **Response time** | ~2 weeks for EOI; ~2 months for full proposal decision |
| **Alignment** | PHANTOM -- fits under "adversarial machine learning" research direction |
| **Confidence** | MODERATE. $40M fund. The original RFP deadline passed, but they encouraged rolling submissions. Verify current status before investing application time. |

**Action required:**
1. Check https://www.openphilanthropy.org/how-to-apply-for-funding/ to confirm still accepting EOIs
2. If open, submit 300-word EOI on PHANTOM as adversarial ML research
3. Propose discrete 6-12 month project: systematic characterization of covert formatting channels across major LLMs
4. Budget: $50K-$200K for solo researcher + compute

**Source:** [Open Philanthropy RFP](https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/) | [EA Forum Discussion](https://forum.effectivealtruism.org/posts/XtgDaunRKtCPzyCWg/open-philanthropy-technical-ai-safety-rfp-usd40m-available)

---

### 2.4 Frontier Model Forum -- AI Safety Fund

| Field | Detail |
|-------|--------|
| **Status** | UPCOMING -- next open call TBD |
| **URL** | https://www.frontiermodelforum.org/ai-safety-fund/ |
| **Payout** | Varies; $5M+ awarded in December 2025 cohort |
| **Deadline** | No current open call; monitor for next RFP |
| **Alignment** | PHANTOM (cybersecurity focus area) and MWRASP (potential) |
| **Confidence** | LOW for 90-day window. Next call timing unknown. |

**Action required:** Monitor the AISF page for next open call. When it opens, PHANTOM maps to their cybersecurity research focus. Prepare application materials now.

**Source:** [Frontier Model Forum AISF](https://www.frontiermodelforum.org/ai-safety-fund/)

---

## SECTION 3: CONFERENCES AND PRESENTATIONS

### 3.1 DEF CON 34 -- Main Stage CFP

| Field | Detail |
|-------|--------|
| **Status** | OPEN NOW |
| **URL** | https://defcon.org/html/defcon-34/dc-34-cfp.html |
| **Dates** | August 6-9, 2026, Las Vegas Convention Center |
| **CFP deadline** | May 1, 2026 at Midnight UTC |
| **Talk lengths** | 20 minutes or 45 minutes |
| **Alignment** | PHANTOM -- textbook DEF CON material |
| **Confidence** | HIGH that topic is accepted IF submission quality is strong |

**Why PHANTOM is DEF CON-grade:**
- Novel attack class (covert channels in LLM structural formatting)
- Cross-platform (Claude 93-100%, GPT 89-100%, Gemini 90%+)
- Open-source detection tool (phantom-detect)
- Practical demonstration potential
- No prior public presentation of this specific attack vector at this scale

**Action required:**
1. Submit 45-minute talk proposal before May 1, 2026
2. Title suggestion: "PHANTOM: Covert Data Exfiltration Through the Spaces Between Words"
3. Include white paper (.txt upload) -- "highly encouraged and receive priority consideration"
4. Max 2 speakers. Cannot submit more than 5 proposals total.
5. No LLM-generated text in submission (refine author-written material only)
6. Simultaneously submit to AI Village when their CFP opens

**Source:** [DEF CON 34 CFP](https://defcon.org/html/defcon-34/dc-34-cfp.html)

---

### 3.2 DEF CON 34 -- AI Village

| Field | Detail |
|-------|--------|
| **Status** | UPCOMING -- CFP not yet announced (but content call season is open) |
| **URL** | https://aivillage.org |
| **Dates** | August 6-9, 2026 (concurrent with DEF CON 34) |
| **CFP deadline** | TBD -- monitor aivillage.org |
| **Alignment** | PHANTOM -- direct fit |

**Action required:** Monitor https://aivillage.org for CFP announcement. Previous AI Village CFPs were on EasyChair. Submit as soon as it opens.

**Source:** [AI Village](https://aivillage.org) | [DEF CON Forums](https://forum.defcon.org/node/248649)

---

### 3.3 Black Hat USA 2026

| Field | Detail |
|-------|--------|
| **Status** | CFP CLOSED (deadline was March 20, 2026) |
| **URL** | https://usa-briefings-cfp.blackhat.com |
| **Dates** | August 4-6, 2026, Mandalay Bay, Las Vegas |
| **Alignment** | PHANTOM |

**Missed window for Briefings CFP.** However:
- **Arsenal** submissions were due March 13 (also closed)
- **Check if late submissions are accepted** by emailing cfp@blackhat.com

**Fallback:** Target Black Hat AI Summit (co-located) or Black Hat Europe 2026 (typically November/December CFP).

**Source:** [Black Hat USA 2026 CFP](https://blackhat.com/call-for-papers.html)

---

### 3.4 IEEE Symposium on Security and Privacy (S&P) 2026

| Field | Detail |
|-------|--------|
| **Status** | CLOSED for 2026 (both cycles passed) |
| **Dates** | May 18-21, 2026, San Francisco |
| **Cycle 1 deadline** | June 6, 2025 (passed) |
| **Cycle 2 deadline** | February 5, 2026 (passed) |
| **Alignment** | PHANTOM + MWRASP |

**Action required:** Target IEEE S&P 2027 Cycle 1 (likely ~June 2026 deadline). Start paper preparation now.

**Source:** [IEEE S&P 2026](https://sp2026.ieee-security.org/cfpapers.html)

---

### 3.5 USENIX Security 2026

| Field | Detail |
|-------|--------|
| **Status** | CLOSED for 2026 (Cycle 2 deadline was February 5, 2026) |
| **Dates** | August 12-14, 2026, Baltimore |
| **Alignment** | PHANTOM |

**Action required:** Target USENIX Security 2027 when CFP opens.

**Source:** [USENIX Security '26](https://www.usenix.org/conference/usenixsecurity26/call-for-papers)

---

### 3.6 IEEE SaTML 2026

| Field | Detail |
|-------|--------|
| **Status** | Paper submissions CLOSED (September 24, 2025). Conference March 23-25, 2026, Munich. |
| **Competitions** | 4 competitions running (AgentCTF, Anti-BAD, etc.) -- deadlines likely passed |
| **Alignment** | PHANTOM fits perfectly for future cycles |

**Action required:** Target SaTML 2027 CFP (~September 2026 deadline expected).

**Source:** [IEEE SaTML 2026](https://satml.org)

---

## SECTION 4: PLATFORM OPPORTUNITIES

### 4.1 HackerOne Reputation Building

| Field | Detail |
|-------|--------|
| **Status** | ONGOING |
| **URL** | https://hackerone.com |
| **Alignment** | PHANTOM -- reputation leads to private program invites |

**Strategy:** Submit PHANTOM findings to public programs (Anthropic VDP, xAI, Amazon). Valid submissions build Signal and Impact scores, leading to invitations to higher-paying private programs. Top researchers earn $300K+/year.

The platform saw a 210% jump in valid AI reports in the past year. AI security researchers are in high demand for private invitations.

**Action required:**
1. Create/optimize HackerOne profile emphasizing AI security expertise
2. Submit to public programs first (build reputation)
3. Quality over quantity: HackerOne rewards Signal (accuracy) and Impact (severity)

**Source:** [HackerOne Leaderboard](https://hackerone.com/leaderboard) | [AI Bug Bounty Guide](https://docs.hackerone.com/en/articles/12570435-ai-bug-bounty)

---

### 4.2 Bugcrowd Programs

| Field | Detail |
|-------|--------|
| **Status** | ONGOING |
| **URL** | https://bugcrowd.com |
| **Key programs** | OpenAI (via Bugcrowd), Perplexity (private) |
| **Alignment** | PHANTOM |

**Strategy:** Bugcrowd's invitation system rewards performance with access to higher-quality programs. Submit to OpenAI program via Bugcrowd to build reputation.

**Source:** [Bugcrowd Programs](https://bugcrowd.com)

---

### 4.3 Microsoft Zero Day Quest 2026

| Field | Detail |
|-------|--------|
| **Status** | Research Challenge CLOSED (ended October 4, 2025). Live Event ACTIVE (Feb 17 - Mar 18, 2026, invite-only). |
| **Payout** | Up to $5M total bounty pool. 50% multiplier on critical findings. |
| **Alignment** | PHANTOM -- Copilot is in scope |
| **Confidence** | NOT ELIGIBLE for current Live Event unless already invited |

**Action required for future:** Submit high-quality findings to MSRC throughout 2026 to qualify for Zero Day Quest 2027.

**Source:** [Zero Day Quest](https://www.microsoft.com/en-us/msrc/microsoft-zero-day-quest) | [Live Hacking Event](https://www.microsoft.com/en-us/msrc/zero_day_quest_live_hacking_event)

---

## SECTION 5: PRIORITY ACTION PLAN (NEXT 30 DAYS)

### Week 1 (Feb 26 - Mar 4)

| # | Action | Target | Expected Payout | Time Investment |
|---|--------|--------|----------------|-----------------|
| 1 | Submit PHANTOM to Google AI VRP | Google Gemini Apps | $15,000-$30,000 | 4-8 hours (PoC + writeup) |
| 2 | Apply for Anthropic HackerOne invite | Anthropic | $15,000-$25,000 | 1 hour |
| 3 | Start OpenAI Cybersecurity Grant application | OpenAI | $10,000 + API credits | 8-12 hours (3,000-word proposal) |
| 4 | Email 0DIN with PHANTOM abstract | Mozilla | TBD (severity-based) | 2 hours |

### Week 2 (Mar 5 - Mar 11)

| # | Action | Target | Expected Payout | Time Investment |
|---|--------|--------|----------------|-----------------|
| 5 | Submit PHANTOM to Microsoft Copilot Bounty | Microsoft | $5,000-$30,000 | 4-8 hours |
| 6 | Submit OpenAI Cybersecurity Grant proposal | OpenAI | $10,000+ | 4 hours (finalize) |
| 7 | Register and submit to Huntr | Protect AI | Up to $50,000 | 6-10 hours |
| 8 | Apply for Anthropic Fellows (July 2026) | Anthropic | Stipend + funding + compute | 4-6 hours |

### Week 3-4 (Mar 12 - Mar 25)

| # | Action | Target | Expected Payout | Time Investment |
|---|--------|--------|----------------|-----------------|
| 9 | Submit to xAI/Grok via HackerOne | xAI | TBD | 2-4 hours |
| 10 | Submit to Amazon (public program, Gen AI Apps) | Amazon | Up to $20,000 | 2-4 hours |
| 11 | Prepare DEF CON 34 CFP submission | DEF CON | Credibility + exposure | 12-20 hours |
| 12 | Submit Open Philanthropy EOI (if still open) | Open Phil | $50,000-$5M | 2 hours (300-word EOI) |

### Month 2 (Apr 1 - Apr 30)

| # | Action | Target | Expected Payout | Time Investment |
|---|--------|--------|----------------|-----------------|
| 13 | Submit DEF CON 34 CFP (deadline May 1) | DEF CON | Credibility | 4 hours (finalize) |
| 14 | Assess Q-Day Prize submission feasibility | Project Eleven | 1 BTC (~$85K) | 8-20 hours |
| 15 | Follow up on all pending bounty submissions | All platforms | -- | 2-4 hours |
| 16 | Request Salesforce HackerOne private invite | Salesforce | Up to $32,000 | 1 hour |

---

## SECTION 6: ESTIMATED REVENUE POTENTIAL (90-DAY WINDOW)

### Conservative Estimate (1-2 bounties accepted)

| Source | Amount |
|--------|--------|
| Google AI VRP (1 finding) | $15,000 |
| OpenAI Cybersecurity Grant | $10,000 |
| **Total** | **$25,000** |

### Moderate Estimate (3-4 bounties + grant)

| Source | Amount |
|--------|--------|
| Google AI VRP | $15,000-$30,000 |
| Anthropic Bounty | $15,000-$25,000 |
| Microsoft Copilot | $5,000-$20,000 |
| OpenAI Grant | $10,000 |
| **Total** | **$45,000-$85,000** |

### Optimistic Estimate (5+ bounties + grants + awards)

| Source | Amount |
|--------|--------|
| Google AI VRP | $30,000 |
| Anthropic Bounty | $25,000 |
| Microsoft Copilot | $30,000 |
| Huntr (with 10x multiplier) | $50,000 |
| OpenAI Grant | $10,000 |
| Mozilla 0DIN | $10,000 |
| **Total** | **$155,000** |

**Realistic expectation:** $25,000-$85,000 within 90 days. The key variable is how quickly platforms validate and pay. Google and Microsoft have the fastest triage-to-payment pipelines.

---

## SECTION 7: OPPORTUNITIES NOT WORTH PURSUING

| Opportunity | Reason to Skip |
|-------------|---------------|
| NVIDIA VDP | No paid bounties -- acknowledgment only |
| Perplexity VDP | AI model issues explicitly out of scope |
| USENIX Security 2026 | Submission deadline passed |
| IEEE S&P 2026 | Submission deadline passed |
| IEEE SaTML 2026 | Submission deadline passed |
| Black Hat USA 2026 Briefings | CFP deadline passed (March 20) |
| Munich Security Conference Essay | Deadline passed (January 11, 2026) |
| Microsoft Zero Day Quest Live Event | Invite-only, qualification period ended |
| DARPA AIxCC | Competition finals concluded in 2025 |
| Databricks Bug Bounty | Infrastructure focus, weak PHANTOM alignment |

---

## SECTION 8: KEY FRAMING GUIDANCE

**How to position PHANTOM across all submissions:**

1. **Never call it "prompt injection."** Call it "Covert Data Exfiltration via Structural Formatting Channels." Most programs exclude prompt injection but include data exfiltration.

2. **Emphasize the EXFILTRATION, not the INJECTION.** Per Joseph Thacker (HackerOne researcher): "Don't just say 'Prompt Injection Vulnerability.' Say 'Data Exfiltration via Dynamic Image Rendering.'" Apply this principle: PHANTOM is about what happens AFTER the prompt -- the covert channel in the output formatting.

3. **Highlight cross-model universality.** Claude 93-100%, GPT 89-100%, Gemini 90%+. This isn't a narrow bug in one model -- it's a systemic vulnerability class.

4. **Reference phantom-detect as proof of impact.** Open-source detection tool demonstrates you're not just finding problems, you're building solutions.

5. **Map to OWASP Top 10 for LLMs.** PHANTOM maps to:
   - LLM06: Sensitive Information Disclosure
   - LLM01: Prompt Injection (as an enabler, not the primary finding)
   - LLM07: Insecure Output Handling

6. **For MWRASP submissions (Q-Day Prize, quantum programs):** Position as "practical post-quantum authentication" rather than theoretical crypto research. Emphasize the 30+ provisionals as demonstrating systematic coverage of the problem space.

---

## APPENDIX: QUICK REFERENCE LINKS

| Program | URL | Status |
|---------|-----|--------|
| Google AI VRP | https://bughunters.google.com | OPEN |
| Anthropic VDP (HackerOne) | https://hackerone.com/anthropic-vdp | OPEN (invite) |
| OpenAI Bugcrowd | https://bugcrowd.com/openai | OPEN |
| OpenAI Cybersecurity Grant | https://openai.com/form/cybersecurity-grant-program/ | OPEN (rolling) |
| Microsoft Copilot Bounty | https://www.microsoft.com/en-us/msrc/bounty-ai | OPEN |
| Huntr (Protect AI) | https://huntr.com | OPEN |
| Mozilla 0DIN | https://0din.ai | OPEN |
| Adobe (HackerOne) | https://hackerone.com/adobe | OPEN |
| Amazon VRP (HackerOne) | https://hackerone.com/amazonvrp | OPEN |
| xAI (HackerOne) | https://hackerone.com/x | OPEN |
| Salesforce (HackerOne) | Private/invite | OPEN (invite) |
| Databricks (HackerOne) | https://hackerone.com/databricks | OPEN |
| Anthropic Fellows | https://job-boards.greenhouse.io/anthropic/jobs/5023394008 | OPEN (rolling) |
| Open Philanthropy RFP | https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/ | VERIFY |
| Frontier Model Forum AISF | https://www.frontiermodelforum.org/ai-safety-fund/ | MONITOR |
| DEF CON 34 CFP | https://defcon.org/html/defcon-34/dc-34-cfp.html | OPEN (May 1) |
| AI Village | https://aivillage.org | MONITOR |
| Q-Day Prize | https://www.qdayprize.org | OPEN (Apr 5) |
| HackerOne Profile | https://hackerone.com | BUILD |
| Bugcrowd Profile | https://bugcrowd.com | BUILD |
