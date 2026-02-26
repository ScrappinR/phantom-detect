# AI Security Bounty Programs, Grants & Competitions: 2025-2026 Landscape

**Compiled**: 2026-02-26
**Status**: Active research -- verify deadlines and program status before submission
**Confidence**: MODERATE on program details (sourced from public announcements); LOW on exact current payout amounts (companies update without notice)

---

## TABLE OF CONTENTS

1. [AI-Specific Bug Bounty Programs](#1-ai-specific-bug-bounty-programs)
2. [Government AI Security Initiatives](#2-government-ai-security-initiatives)
3. [Private AI Security Grants & Funding](#3-private-ai-security-grants--funding)
4. [Conference & Competition Prizes](#4-conference--competition-prizes)
5. [Platforms Aggregating AI Bounties](#5-platforms-aggregating-ai-bounties)
6. [Policy Context: Executive Orders & Regulatory Shifts](#6-policy-context)
7. [Opportunity Matrix for Phantom-Detect](#7-opportunity-matrix-for-phantom-detect)

---

## 1. AI-SPECIFIC BUG BOUNTY PROGRAMS

### 1.1 OpenAI Bug Bounty Program (UPGRADED)

| Field | Detail |
|-------|--------|
| **Name** | OpenAI Bug Bounty Program |
| **URL** | https://bugcrowd.com/openai |
| **Platform** | Bugcrowd |
| **Status** | Active, open to all researchers |
| **Max Payout** | **$100,000** (increased from $20,000 in March 2025 -- 5x increase) |
| **Scope** | ChatGPT, OpenAI APIs, corporate infrastructure, website |
| **Focus** | Infrastructure vulns, IDOR, authentication, data exposure. Prompt injection / jailbreaks NOT in scope for bounty |
| **Bonus Periods** | Promotional periods with doubled payouts for specific vuln classes (e.g., IDOR max $13K during April 2025 promo) |
| **Submission** | Via Bugcrowd platform |
| **Source** | https://openai.com/index/security-on-the-path-to-agi/ |

**Key Changes (2025)**: OpenAI also expanded its Cybersecurity Grant Program (see Section 3) and partnered with SpecterOps for red team engagements.

---

### 1.2 Google AI Vulnerability Reward Program (NEW - Oct 2025)

| Field | Detail |
|-------|--------|
| **Name** | Google AI Vulnerability Reward Program (AI VRP) |
| **URL** | https://bughunters.google.com/about/rules/google-friends/5222232590712832/ai-vulnerability-reward-program-rules |
| **Platform** | Google Bug Hunters |
| **Status** | Active, launched October 2025 |
| **Max Payout** | **$30,000** (base reward up to $20K; with report quality multipliers up to $30K) |
| **Scope** | Google Search, Gemini Apps, Google Workspace (Gmail, Drive, Sheets, Calendar) |
| **In-Scope Vulns** | Rogue actions, sensitive data exfiltration, phishing enablement, model theft -- must involve LLM/GenAI interaction as integral part |
| **Out of Scope** | Prompt injections and jailbreaks (no bounty) |
| **Submission** | Via Google Bug Hunters platform |
| **Source** | https://www.infosecurity-magazine.com/news/google-launches-ai-bug-bounty/ |

---

### 1.3 Microsoft Copilot Bounty Program (EXPANDED)

| Field | Detail |
|-------|--------|
| **Name** | Microsoft Copilot Bounty |
| **URL** | https://www.microsoft.com/en-us/msrc/bounty-ai |
| **Platform** | Microsoft Security Response Center (MSRC) |
| **Status** | Active, expanded in 2025 |
| **Payout Range** | $250 - $30,000 (higher at Microsoft's discretion) |
| **Scope** | Microsoft Copilot, Copilot Studio, Dynamics 365 AI features, Power Platform AI |
| **Focus** | Code injection, model manipulation, prompt injection, data leakage |
| **Multiplier** | 100% award multiplier for all Copilot bounty submissions |
| **Submission** | Via MSRC portal |
| **Source** | https://www.microsoft.com/en-us/msrc/bounty-ai |

**Note**: Dynamics 365 & Power Platform bounty expanded to include AI Bounty Award category in 2025.

---

### 1.4 Anthropic Model Safety Bug Bounty (EXPANDED)

| Field | Detail |
|-------|--------|
| **Name** | Anthropic Model Safety Bug Bounty Program |
| **URL** | https://hackerone.com/anthropic-vdp |
| **Platform** | HackerOne (invite-only bounty) + public VDP |
| **Status** | Active, expanded through 2025 |
| **Max Payout** | **$25,000** for unreleased systems; $15,000 for released models |
| **Scope** | Universal jailbreak attacks on Claude models; CBRN and cybersecurity domains prioritized |
| **Model** | Invite-only for paid bounty; VDP open to all |
| **Focus** | Safety classifier bypasses, universal jailbreaks that work across domains |
| **Submission** | Via HackerOne |
| **Source** | https://www.anthropic.com/news/model-safety-bug-bounty |

**Status Update**: Anthropic expanded from VDP-only to paid bounty. Still primarily invite-only for monetary rewards, but expanding broadly.

---

### 1.5 Amazon Nova AI Bug Bounty (NEW - Late 2025)

| Field | Detail |
|-------|--------|
| **Name** | Amazon AI Bug Bounty (Nova Models) |
| **URL** | https://hackerone.com/amazonvrp |
| **Platform** | HackerOne |
| **Status** | Active -- private/invite-only track launched Nov 2025, expanding early 2026 |
| **Payout Range** | $200 - $25,000 |
| **Scope** | Amazon Nova foundation models, Amazon AI applications |
| **Focus** | Cybersecurity issues, CBRN threat detection, model safety |
| **Track Record** | Public program already surfaced 30+ validated findings, $55K+ in rewards paid |
| **Access** | Invite-only for private track; public program remains open |
| **Submission** | Via HackerOne Amazon VRP |
| **Source** | https://www.amazon.science/news/amazon-launches-private-ai-bug-bounty-to-strengthen-nova-models |

---

### 1.6 Mozilla 0Din GenAI Bug Bounty (ACTIVE)

| Field | Detail |
|-------|--------|
| **Name** | Zero Day Investigative Network (0Din) |
| **URL** | https://0din.ai/ |
| **Platform** | 0Din proprietary portal |
| **Status** | Active, launched June 2024, operational through 2025-26 |
| **Payout Range** | $500 - $15,000 (discretionary, based on impact/quality/timing) |
| **Scope** | Cross-vendor: GPT-4, Gemini, LLaMA, Claude, and other major LLMs |
| **Focus** | Guardrail jailbreaks, prompt injection attacks, interpreter jailbreaks |
| **Open To** | Everyone -- no prerequisite |
| **Submission** | Via 0Din portal |
| **Source** | https://hacks.mozilla.org/2024/08/0din-a-genai-bug-bounty-program-securing-tomorrows-ai-together/ |

**Differentiator**: 0Din covers vulnerabilities that fall OUTSIDE the scope of individual vendor bounty programs -- cross-model, cross-platform attacks.

---

### 1.7 Huntr by Protect AI (AI/ML Dedicated Platform)

| Field | Detail |
|-------|--------|
| **Name** | Huntr -- AI/ML Bug Bounty Platform |
| **URL** | https://huntr.com/ |
| **Platform** | Huntr (acquired by Protect AI) |
| **Status** | Active |
| **Max Payout** | Up to $50,000 for critical vulns (contest-specific) |
| **Scope** | 125+ ML supply chain repositories, open-source AI/ML projects, Hugging Face ecosystem |
| **Features** | Monthly contests, structured reporting, CVE assignment (5th largest CNA globally) |
| **Community** | 10,000+ security researchers |
| **Submission** | Via Huntr platform |
| **Source** | https://huntr.com/ |

---

### 1.8 NVIDIA Bug Bounty (NEW - Summer 2025)

| Field | Detail |
|-------|--------|
| **Name** | NVIDIA Bug Bounty & VDP |
| **URL** | https://app.intigriti.com/programs/nvidia/nvidiavdp/detail |
| **Platform** | Intigriti |
| **Status** | Active -- launched summer 2025 |
| **Payout** | Case-by-case (private bounty); VDP has no financial reward |
| **Scope** | NVIDIA products (private bounty), all NVIDIA assets including website (VDP) |
| **AI Focus** | Separate private bug bounty package specifically covering core AI assets |
| **Access** | Private bounty is invite-only; VDP is public |
| **Submission** | Via Intigriti platform |
| **Source** | https://www.intigriti.com/blog/business-insights/intigriti-teams-with-nvidia-to-launch-bug-bounty-vulnerability-disclosure-program |

---

### 1.9 Adobe Firefly AI Bug Bounty (EXPANDED)

| Field | Detail |
|-------|--------|
| **Name** | Adobe Bug Bounty -- Firefly & Content Credentials |
| **URL** | https://hackerone.com/adobe |
| **Platform** | HackerOne |
| **Status** | Active, AI scope added 2024, running through 2025 |
| **Max Payout** | Up to $10,000 for critical vulns |
| **Scope** | Adobe Firefly, Content Credentials |
| **Focus** | Prompt injection, sensitive info disclosure, training data poisoning (OWASP LLM Top 10) |
| **Bonus** | Quarterly Hall of Fame + Adobe merchandise / 12-month Creative Cloud subscription |
| **Submission** | Via HackerOne |
| **Source** | https://blog.adobe.com/en/publish/2024/05/01/adobe-collaborates-with-ethical-hackers-build-safer-more-secure-ai-tools |

---

### 1.10 Salesforce Bug Bounty (AI-INCLUSIVE)

| Field | Detail |
|-------|--------|
| **Name** | Salesforce Bug Bounty Program |
| **URL** | Invite-only (contact security@salesforce.com) |
| **Status** | Active, $23M+ paid to date |
| **Max Payout** | Up to $60,000 (individual payouts observed) |
| **Scale** | 480+ ethical hackers, 4,000+ reports in 2024, $3M+ paid in 2024 alone |
| **AI Scope** | Einstein AI, Agentforce -- includes AI bias testing and autonomous agent behaviors |
| **Access** | Invite-only |
| **Source** | https://www.salesforce.com/news/stories/bug-bounty-program-results-2024/ |

---

### 1.11 xAI / Grok Bug Bounty

| Field | Detail |
|-------|--------|
| **Name** | X / xAI Bug Bounty Program |
| **URL** | https://hackerone.com/x |
| **Platform** | HackerOne |
| **Status** | Active |
| **Scope** | xAI products, Grok |
| **Contact** | vulnerabilities@x.ai (subject: "Responsible Disclosure") |
| **Source** | https://x.ai/security |

**Note**: Specific payout ranges not publicly disclosed. Program tracked via HackerOne.

---

### 1.12 Perplexity AI VDP / Bug Bounty

| Field | Detail |
|-------|--------|
| **Name** | Perplexity Vulnerability Disclosure Program + Private Bug Bounty |
| **URL** | https://bugcrowd.com/engagements/perplexity-vdp-ess |
| **Platform** | Bugcrowd (VDP public; bounty private) |
| **Status** | Active |
| **Contact** | security@perplexity.ai |
| **Source** | https://www.perplexity.ai/hub/security |

**Note**: Perplexity operates a private paid bounty on Bugcrowd alongside the public VDP. Payout details not publicly disclosed.

---

### 1.13 Inflection AI Bug Bounty

| Field | Detail |
|-------|--------|
| **Name** | Inflection Bug Bounty Program |
| **URL** | https://hackerone.com/inflection |
| **Platform** | HackerOne |
| **Status** | Active |
| **Source** | https://hackerone.com/inflection |

**Note**: Payout details not publicly disclosed.

---

### 1.14 Databricks (Includes Mosaic AI)

| Field | Detail |
|-------|--------|
| **Name** | Databricks Bug Bounty Program |
| **URL** | https://hackerone.com/databricks |
| **Platform** | HackerOne |
| **Status** | Active, year-round public program |
| **Scope** | Databricks platform (includes Mosaic AI products) |
| **Source** | https://hackerone.com/databricks |

---

### 1.15 Relevance AI VRP

| Field | Detail |
|-------|--------|
| **Name** | Relevance AI Vulnerability Reward Program |
| **URL** | https://relevanceai.com/vulnerability-reward-program |
| **Status** | Active |

---

## 2. GOVERNMENT AI SECURITY INITIATIVES

### 2.1 DARPA SABER Program (NEW - 2025)

| Field | Detail |
|-------|--------|
| **Name** | Securing Artificial Intelligence for Battlefield Effective Robustness (SABER) |
| **URL** | https://www.darpa.mil/research/programs/saber-securing-artificial-intelligence |
| **Status** | Proposals due June 3, 2025 (extended from May 6); anticipated kickoff Nov 2025 |
| **Duration** | 24-month single-phase program |
| **Focus** | Operational AI red teaming for battlefield AI systems -- ground and aerial autonomous platforms |
| **Technical Areas** | Physical adversarial AI, cybersecurity, electronic warfare (EW) counter-AI techniques |
| **Target** | AI-enabled systems deploying within 1-3 years |
| **Contracting** | BAA HR001125S0009 |
| **PM** | Dr. Nathaniel D. Bastian (LTC, US Army) |
| **Source** | https://www.darpa.mil/news/2025/saber-warfighter-ai |

**Relevance for Phantom-Detect**: HIGH. SABER explicitly seeks counter-AI techniques and AI vulnerability assessment tools. Covert channel / side-channel attacks on AI systems could fit squarely in Technical Team 1 (Attack Techniques and Tools).

---

### 2.2 DARPA AIxCC (Completed Aug 2025)

| Field | Detail |
|-------|--------|
| **Name** | AI Cyber Challenge (AIxCC) |
| **URL** | https://aicyberchallenge.com/ |
| **Status** | Completed -- finals at DEF CON 33 (August 2025) |
| **Total Prizes** | $8.5M awarded at finals + $1.4M follow-on integration prizes |
| **Winners** | 1st: Team Atlanta ($4M), 2nd: Trail of Bits ($3M), 3rd: Theori ($1.5M) |
| **Performance** | Teams identified 86% of synthetic vulns (up from 37% at semis); discovered 18 real non-synthetic vulns |
| **Open Source** | All finalist CRSs being released as open-source software |
| **Source** | https://www.darpa.mil/news/2025/aixcc-results |

---

### 2.3 DOD CDAO AI Bias Bounty

| Field | Detail |
|-------|--------|
| **Name** | DOD AI Bias Bounty (CDAO Responsible AI Division) |
| **URL** | https://www.war.gov/News/Releases/Release/Article/3659519/ |
| **Platform** | ConductorAI-Bugcrowd + BiasBounty.AI |
| **Status** | First bounty ran Jan-Feb 2024; second bounty anticipated |
| **Focus** | LLM bias detection in open-source chatbots |
| **Open To** | Public -- no coding experience required |
| **Funding** | DOD-funded monetary bounties |
| **Source** | https://defensescoop.com/2024/01/29/defense-department-ai-bias-bug-bounty/ |

**Note**: The first bounty completed in 2024. Second exercise expected but no confirmed 2025-2026 dates found. Monitor ai.mil for updates.

---

### 2.4 NIST AI Cybersecurity Framework & Funding

| Field | Detail |
|-------|--------|
| **Name** | NIST Cybersecurity Framework Profile for AI (NISTIR 8596) |
| **URL** | https://csrc.nist.gov/pubs/ir/8596/iprd |
| **Status** | Draft released Dec 16, 2025; comments closed Jan 30, 2026; initial public draft expected 2026 |
| **Community** | 6,500+ individuals in the community of interest |
| **Funding** | $20M invested in AI + cybersecurity via Mitre partnership |
| **Related** | AI Agent Standards Initiative launched Feb 2026 |
| **Source** | https://www.nist.gov/news-events/news/2025/12/draft-nist-guidelines-rethink-cybersecurity-ai-era |

**New (Feb 2026)**: NIST launched the AI Agent Standards Initiative for interoperable and secure AI agents. RFI for security considerations for AI agents published Jan 8, 2026 (Federal Register 2026-00206).

---

### 2.5 CISA AI Security Guidance

| Field | Detail |
|-------|--------|
| **Name** | CISA AI Security Roadmap & Guidance |
| **URL** | https://www.cisa.gov/resources-tools/resources/roadmap-ai |
| **Status** | Guidance issued; agency facing budget cuts |
| **Key Output** | "Principles for Secure Integration of AI in Operational Technology" (joint with Australian ASD/ACSC) |
| **Budget Context** | Proposed 17% budget cut ($134M reduction) in FY2026; $490M+ at risk |
| **AI Corps** | DHS Artificial Intelligence Corps backed by House appropriators |
| **Cyber Grants** | $100M+ in FY2025 State & Local Cybersecurity Grants (general cyber, not AI-specific) |
| **Source** | https://www.cisa.gov/resources-tools/resources/roadmap-ai |

---

### 2.6 UK AI Safety Institute (AISI) Evaluations Bounty

| Field | Detail |
|-------|--------|
| **Name** | AISI Bounty for Novel Evaluations and Agent Scaffolding |
| **URL** | https://www.aisi.gov.uk/blog/evals-bounty |
| **Status** | First round closed Nov 30, 2024; future rounds possible |
| **Payout** | GBP 2,000 for Stage 2 compute + variable bounty based on development time and success |
| **Focus** | Evaluations for dangerous capabilities in frontier AI systems -- offensive cyber, dual-use bio/chem, autonomous systems |
| **Submission** | Via AISI smartergrants portal |
| **Source** | https://www.aisi.gov.uk/blog/evals-bounty |

---

### 2.7 Executive Order Context (CRITICAL)

**Biden EO 14110 (AI Safety)**: Revoked by Trump on Jan 20, 2025.
**Trump EO 14179 (Jan 23, 2025)**: "Removing Barriers to American Leadership in AI" -- shifts from oversight/regulation to deregulation/innovation.
**Trump EO 14306 (June 6, 2025)**: Scrapped Biden cyber-AI initiatives including:
- Pentagon AI cyber defense mandates
- DARPA-private sector AI critical infrastructure pilot
- Federal research prioritization of AI-powered coding security
- Post-quantum cryptography acceleration requirements

**Trump EO (Dec 11, 2025)**: "Ensuring a National Policy Framework for AI" -- minimally burdensome framework, AI Litigation Task Force to preempt state AI laws.

**Net Effect**: Federal AI security regulatory mandates weakened. Private sector and DARPA programs remain the primary drivers. No new government AI bounty programs created under current administration.

Source: https://www.cybersecuritydive.com/news/trump-cybersecurity-executive-order-eliminate-biden-programs/750119/

---

## 3. PRIVATE AI SECURITY GRANTS & FUNDING

### 3.1 OpenAI Cybersecurity Grant Program (EXPANDED)

| Field | Detail |
|-------|--------|
| **Name** | OpenAI Cybersecurity Grant Program |
| **URL** | https://openai.com/form/cybersecurity-grant-program/ |
| **Status** | Active, rolling applications |
| **Funding** | **$10M in API credits** (expanded from original $1M) |
| **Grant Size** | $10,000 increments from the fund |
| **Focus** | Defensive cybersecurity: software patching, model privacy, detection/response, security integration |
| **Exclusions** | Offensive security projects NOT funded |
| **Track Record** | 1,000+ applications reviewed, 28 initiatives funded since launch |
| **Requirements** | Project proposal (3,000 words max), team info, methodology, timeline. Must be 18+. Open-source/public benefit preferred |
| **Source** | https://openai.com/index/openai-cybersecurity-grant-program/ |

---

### 3.2 Coefficient Giving / Open Philanthropy Technical AI Safety RFP

| Field | Detail |
|-------|--------|
| **Name** | Technical AI Safety Research RFP |
| **URL** | https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/ |
| **Status** | Applications closed April 15, 2025 (late submissions accepted through July 15, 2025) |
| **Funding** | **~$40M** committed over 5 months; more available based on quality |
| **Grant Range** | API credits to seed funding for new research organizations |
| **Focus** | Misalignment risk, AI safety technical research |
| **Awarded Examples** | $187K (Ohio State -- alignment faking), $1M (ELLIS Institute -- AI safety), multiple others |
| **Total Grantmaking** | 440+ grants via Navigating Transformative AI fund; $4B+ total since 2014 |
| **Source** | https://coefficientgiving.org/funds/navigating-transformative-ai/request-for-proposals-technical-ai-safety-research/ |

**Note**: Coefficient Giving (formerly Open Philanthropy's AI grantmaking arm) is actively seeking new funders giving >=$250K/year.

---

### 3.3 Frontier Model Forum AI Safety Fund (ACTIVE)

| Field | Detail |
|-------|--------|
| **Name** | AI Safety Fund (AISF) |
| **URL** | https://www.frontiermodelforum.org/ai-safety-fund/ |
| **Status** | Active, new cohort announced Dec 11, 2025 |
| **Total Funding** | $10M+ |
| **Backers** | Anthropic, Google, Microsoft, OpenAI + Patrick J. McGovern Foundation, Packard Foundation, Schmidt Sciences, Jaan Tallinn |
| **Latest Round** | 11 grantees, $5M+, focused on biosecurity, cybersecurity, AI agent evaluation, synthetic content |
| **Grant Range** | $150,000 - $400,000 (first round); larger in second round |
| **Source** | https://www.frontiermodelforum.org/updates/announcement-of-new-ai-safety-fund-grantees/ |

**Notable Cybersecurity-Relevant Grantees (Dec 2025)**:
- **FAR.AI**: Quantifying the Safety-Adversary Gap in LLMs
- **Morgan State University**: Evaluating AI-Assisted Cybersecurity Operations
- **Nemesys Insights LLC**: ICS Benchmark and Human Uplift Study
- **Apollo Research**: Black box scheming monitors for frontier AI agents
- **University of Illinois Urbana-Champaign**: Cybersecurity Risk Evaluations of AI Agents
- **SecureBio**: Evaluations for AI execution of tasks enabling large-scale harm

---

### 3.4 Anthropic Fellows Program (AI Safety Research)

| Field | Detail |
|-------|--------|
| **Name** | Anthropic Fellows Program |
| **URL** | https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/ |
| **Status** | Applications OPEN for May & July 2026 cohorts |
| **Duration** | 4 months per cohort |
| **Compensation** | $3,850/week (USD), ~$15K/month compute budget, Anthropic mentorship |
| **Focus** | Scalable oversight, adversarial robustness, AI control, mechanistic interpretability, **AI security**, model welfare |
| **Prerequisites** | No PhD required; physics, math, CS, cybersecurity backgrounds accepted |
| **Track Record** | 80%+ of fellows produced papers; 40%+ joined Anthropic full-time |
| **Source** | https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/ |

**Key**: "AI security" is explicitly listed as a research area for the 2026 cohorts.

---

### 3.5 Anthropic External Researcher Access Program

| Field | Detail |
|-------|--------|
| **Name** | External Researcher Access Program |
| **URL** | https://support.claude.com/en/articles/9125743 |
| **Status** | Active |
| **Provides** | Free API credits for Claude models |
| **Focus** | AI safety and alignment research (Anthropic-defined priority topics) |
| **Source** | https://support.claude.com/en/articles/9125743 |

---

### 3.6 Anthropic Economic Futures Program

| Field | Detail |
|-------|--------|
| **Name** | Economic Futures Research Awards |
| **URL** | https://www.anthropic.com/economic-futures/program |
| **Status** | Active, rolling applications |
| **Grant Range** | $10,000 - $50,000 + $5,000 in Claude API credits |
| **Focus** | Empirical research on AI economic impacts (labor markets, productivity, value creation) |
| **Partners** | Georgetown McCourt School, LSE Data Science Institute |
| **Source** | https://www.anthropic.com/news/introducing-the-anthropic-economic-futures-program |

---

### 3.7 CSET Georgetown Foundational Research Grants

| Field | Detail |
|-------|--------|
| **Name** | Foundational Research Grants (FRG) |
| **URL** | https://cset.georgetown.edu/foundational-research-grants/ |
| **Status** | Active, new call for research on internal deployment risks |
| **Grant Size** | Up to **$1,000,000** per project |
| **Focus** | AI assurance, technical tools for external scrutiny, frontier AI risks |
| **Funded By** | Open Philanthropy ($100M+ total to CSET through 2025) + Google.org ($2M) |
| **Source** | https://cset.georgetown.edu/foundational-research-grants/ |

**Current RFI**: "Risks From Internal Deployment of Frontier AI Models" -- examining how internal models exceeding public capabilities become targets for theft/sabotage.

---

### 3.8 Foresight Institute AI for Science & Safety Nodes

| Field | Detail |
|-------|--------|
| **Name** | AI for Science & Safety Nodes |
| **URL** | https://foresight.org/grants/grants-ai-for-science-safety/ |
| **Status** | Nodes opening in San Francisco and Berlin in early 2026 |
| **Total Annual Funding** | ~$3M total annually |
| **Grant Range** | $10,000 - $100,000 (higher for AI safety focus areas) |
| **Source** | https://foresight.org/grants/grants-ai-for-science-safety/ |

---

### 3.9 AI Risk Mitigation Fund (ARM Fund)

| Field | Detail |
|-------|--------|
| **Name** | AI Risk Mitigation Fund |
| **URL** | https://www.airiskfund.com/ |
| **Status** | Active |
| **Focus** | Technical research, policy, and training programs for new AI safety researchers |
| **Goal** | Reduce catastrophic risks from advanced AI |
| **Source** | https://www.airiskfund.com/ |

---

### 3.10 SPAR AI Safety Research Fellowship

| Field | Detail |
|-------|--------|
| **Name** | SPAR (Student Program on AI Risks) |
| **URL** | https://sparai.org/ |
| **Status** | Active, Spring 2026 projects listed |
| **Format** | Part-time, remote research fellowship |
| **Focus** | AI safety and policy research with professional mentors |
| **New for 2026** | Biosecurity projects added |
| **Source** | https://sparai.org/ |

---

### 3.11 Meta Research RFPs & Llama Impact Grants

| Field | Detail |
|-------|--------|
| **Name** | Meta Request for Proposals + Llama Impact Grants |
| **URL** | https://ai.meta.com/research/request-for-proposals/ |
| **Status** | Active (Llama Impact Grants closed June 27, 2025) |
| **RFP Range** | $15,000 - $55,000 per award (including 40% overhead) |
| **Llama Grants** | Global program for open-source AI social impact projects |
| **Eligibility** | Full-time faculty at accredited research institutions (RFPs) |
| **Source** | https://ai.meta.com/research/request-for-proposals/ |

**Note**: No Meta grant specifically for AI security research identified. Security-relevant work could potentially fit within broader RFP categories. Monitor for new calls.

---

### 3.12 FAR.AI ($30M+ Multi-Funder)

| Field | Detail |
|-------|--------|
| **Name** | FAR.AI |
| **URL** | https://www.far.ai/ |
| **Status** | Secured $30M+ in 2025 |
| **Funders** | Coefficient Giving, Schmidt Sciences, Survival and Flourishing Fund, CSET, AISF/FMF |
| **Focus** | Frontier AI safety research at scale |
| **Source** | https://www.far.ai/news/30m-multi-funder-support |

---

## 4. CONFERENCE & COMPETITION PRIZES

### 4.1 DARPA AIxCC Finals (DEF CON 33 -- Aug 2025)

| Field | Detail |
|-------|--------|
| **Total Prizes** | $8.5M (finals) + $1.4M (integration follow-on) |
| **1st Place** | Team Atlanta -- $4M |
| **2nd Place** | Trail of Bits -- $3M |
| **3rd Place** | Theori -- $1.5M |
| **Status** | Completed. All finalist CRSs being open-sourced |
| **Source** | https://aicyberchallenge.com/finals-winners-announcement/ |

---

### 4.2 Microsoft Zero Day Quest (Spring 2026)

| Field | Detail |
|-------|--------|
| **Name** | Zero Day Quest Live Hacking Event |
| **URL** | https://www.microsoft.com/en-us/msrc/zero_day_quest_live_hacking_event |
| **Status** | Research Challenge open Aug 4 - Oct 4, 2025; Live Event at Redmond campus March 2026 |
| **Total Prize Pool** | **$5M** (up from $4M in 2024) |
| **Targets** | Azure, Copilot, Dynamics 365, Power Platform, M365, Identity |
| **AI Focus** | Model manipulation, prompt injection, data leakage in Copilot |
| **Research Challenge Bonus** | +50% bounty multiplier for Critical severity findings |
| **Live Event** | Invite-only, up to 45 researchers, Microsoft covers travel ($2K intl, $750 NA) |
| **Training** | PyRIT (Python Risk Identification Toolkit) red team methodology training included |
| **2024 Results** | 600+ submissions, $1.6M paid |
| **Source** | https://www.microsoft.com/en-us/msrc/blog/2025/08/zero-day-quest-join-the-largest-hacking-event-with-up-to-5-million-in-total-bounty-awards/ |

---

### 4.3 DEF CON AI Village CTF

| Field | Detail |
|-------|--------|
| **Name** | AI Village CTF |
| **URL** | https://aivillage.org/ |
| **Status** | Annual at DEF CON |
| **Focus** | Evading, poisoning, stealing, and fooling AI/ML systems |
| **Prizes** | Specific monetary amounts not publicly disclosed; hardware prizes reported at Bug Bounty Village |
| **Source** | https://aivillage.org/ |

---

### 4.4 Amazon Nova AI Challenge

| Field | Detail |
|-------|--------|
| **Name** | Amazon Nova AI Challenge |
| **URL** | https://www.amazon.science/nova-ai-challenge/ |
| **Status** | Winners announced |
| **Focus** | Pushing boundaries of secure AI |
| **Source** | https://www.amazon.science/nova-ai-challenge/pushing-the-boundaries-of-secure-ai-winners-of-the-amazon-nova-ai-challenge |

---

### 4.5 NeurIPS 2025 Competitions

| Field | Detail |
|-------|--------|
| **Name** | NeurIPS 2025 Competition Track |
| **URL** | https://neurips.cc/Conferences/2025/CallForCompetitions |
| **Status** | Competitions announced June 2025 |
| **Example Prizes** | $6K/1st, $4K/2nd, $2K/3rd + $2K student awards (E2LM competition) |
| **AI Safety** | Specific adversarial ML competition details not confirmed for 2025; check competition announcements |
| **Source** | https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/ |

---

### 4.6 MLCommons AILuminate Benchmarks

| Field | Detail |
|-------|--------|
| **Name** | MLCommons AILuminate Safety & Jailbreak Benchmarks |
| **URL** | https://mlcommons.org/ailuminate/safety/ |
| **Status** | v1.0 Safety Benchmark active; v0.5 Jailbreak Benchmark released Oct 2025; v1.0 Jailbreak planned Q1 2026 |
| **Focus** | 12 hazard categories, 24K+ test prompts per language, "Resilience Gap" metric for jailbreak resistance |
| **Participation** | Open working groups -- researchers can submit novel jailbreak attacks for inclusion |
| **Languages** | English, French (Chinese, Hindi in development) |
| **Not a bounty** | This is a benchmark/standard, not a paid bounty -- but contributing novel attacks builds reputation and visibility |
| **Source** | https://mlcommons.org/2025/10/ailuminate-jailbreak-v05/ |

---

## 5. PLATFORMS AGGREGATING AI BOUNTIES

### 5.1 HackerOne

| Field | Detail |
|-------|--------|
| **URL** | https://www.hackerone.com/bug-bounty-programs |
| **AI Programs** | OpenAI (via Bugcrowd), Anthropic VDP, Amazon VRP, xAI/X, Inflection, Databricks, Adobe, and others |
| **AI Bug Bounty Guidance** | https://docs.hackerone.com/en/articles/12570435-ai-bug-bounty |
| **Features** | OWASP LLM Top 10 scoping guidance, AI-specific vulnerability taxonomies |

### 5.2 Bugcrowd

| Field | Detail |
|-------|--------|
| **URL** | https://www.bugcrowd.com/bug-bounty-list/ |
| **AI Programs** | OpenAI, Perplexity VDP, and others |
| **AI Solutions** | https://www.bugcrowd.com/solutions/ai/ -- crowd-powered AI red teaming, AI Bias Assessments |
| **New Capabilities (2025)** | AI Triage Assistant, AI Analytics, AI Connect (MCP integration) |

### 5.3 Intigriti

| Field | Detail |
|-------|--------|
| **URL** | https://www.intigriti.com/ |
| **AI Programs** | NVIDIA (launched 2025), plus others |
| **Community** | 125,000+ ethical hackers |

### 5.4 Huntr (Protect AI)

| Field | Detail |
|-------|--------|
| **URL** | https://huntr.com/ |
| **Focus** | Dedicated AI/ML bug bounty platform -- the only platform exclusively focused on AI/ML |
| **Scope** | 125+ ML repositories, monthly contests, up to $50K payouts |

### 5.5 0Din (Mozilla)

| Field | Detail |
|-------|--------|
| **URL** | https://0din.ai/ |
| **Focus** | Cross-vendor GenAI bounties -- covers vulns outside individual vendor scopes |

### 5.6 Bug Bounty Directory

| Field | Detail |
|-------|--------|
| **URL** | https://www.bugbountydirectory.com |
| **Focus** | Aggregated list of all bug bounty programs (filter for AI-related) |

### 5.7 AISafety.com Funding Directory

| Field | Detail |
|-------|--------|
| **URL** | https://www.aisafety.com/funding |
| **Focus** | Comprehensive directory of AI safety funding opportunities (grants, fellowships, bounties) |

---

## 6. POLICY CONTEXT

### Federal AI Security Landscape Under Trump Administration

The current regulatory environment has shifted away from mandatory AI security requirements:

1. **Biden's AI Safety EO (14110) revoked** Jan 20, 2025 -- all mandatory red-teaming, reporting, and safety testing requirements for AI developers eliminated
2. **Biden's Cybersecurity EO (14144) gutted** June 6, 2025 -- AI cyber defense mandates, DARPA-private sector partnerships, and AI coding security research priorities canceled
3. **CISA facing 17% budget cut** in FY2026 -- $490M+ at risk
4. **Trump's AI framework EO (Dec 2025)** -- "minimally burdensome" regulation; AI Litigation Task Force to preempt state-level AI laws
5. **What survived**: FCC Cyber Trust Mark program, NIST guidance compliance, threat-hunt operations

**Net assessment**: Government-mandated AI security testing is dead for now. The opportunity space has shifted entirely to:
- DARPA programs (SABER, follow-on AIxCC)
- Private sector bounty programs (which are expanding rapidly)
- Foundation grants (which continue to grow)
- Self-regulatory industry standards (MLCommons, Frontier Model Forum)

---

## 7. OPPORTUNITY MATRIX FOR PHANTOM-DETECT

Based on Phantom-Detect's covert channel / steganographic communication research in AI systems, the highest-relevance opportunities ranked:

### Tier 1 -- Direct Fit

| Opportunity | Relevance | Max Payout/Funding | Action |
|-------------|-----------|-------------------|--------|
| OpenAI Cybersecurity Grant | Defensive AI security research | $10K grants + API credits | Apply now (rolling) |
| DARPA SABER | Counter-AI attack techniques | Contract-level funding | Monitor for follow-on opportunities |
| Anthropic Fellows (May/Jul 2026) | AI security research area | ~$15K/week + $15K/mo compute | Apply for May or July cohort |
| Frontier Model Forum AISF | Cybersecurity evaluation grants | $150K-$400K | Watch for next RFP |
| CSET Georgetown FRG | Frontier AI risks, internal deployment risks | Up to $1M | Active call for research ideas |

### Tier 2 -- Strong Fit with Adaptation

| Opportunity | Relevance | Max Payout/Funding | Action |
|-------------|-----------|-------------------|--------|
| 0Din (Mozilla) | Cross-model GenAI vulnerabilities | $500-$15K per finding | Submit findings |
| Huntr (Protect AI) | AI/ML supply chain vulnerabilities | Up to $50K | Submit findings via platform |
| Google AI VRP | Gemini/Google AI vulnerabilities | Up to $30K | Submit if applicable vulns found |
| OpenAI Bug Bounty | Infrastructure/API vulnerabilities | Up to $100K | Submit if applicable vulns found |
| Microsoft Zero Day Quest | Copilot AI vulnerabilities | Part of $5M pool | Qualify via research challenge |
| Coefficient Giving / Open Phil | Technical AI safety research | Part of ~$40M fund | Watch for next RFP cycle |

### Tier 3 -- Reputation/Visibility Building

| Opportunity | Relevance | Max Payout/Funding | Action |
|-------------|-----------|-------------------|--------|
| MLCommons AILuminate | Jailbreak attack contribution | No direct payout | Submit novel attacks for benchmark inclusion |
| DEF CON AI Village CTF | AI attack demonstration | Prizes vary | Compete at DEF CON 34 |
| NeurIPS Competition Track | Adversarial ML research | $2K-$6K prizes | Monitor 2026 competition calls |
| UK AISI Evals Bounty | Dangerous capability evaluations | GBP 2K+ per eval | Watch for next round |

---

## SOURCES

- [OpenAI Bug Bounty / Security](https://openai.com/index/security-on-the-path-to-agi/)
- [OpenAI Cybersecurity Grant Program](https://openai.com/index/openai-cybersecurity-grant-program/)
- [Google AI VRP](https://www.infosecurity-magazine.com/news/google-launches-ai-bug-bounty/)
- [Google AI VRP Rules](https://bughunters.google.com/about/rules/google-friends/5222232590712832/ai-vulnerability-reward-program-rules)
- [Microsoft Copilot Bounty](https://www.microsoft.com/en-us/msrc/bounty-ai)
- [Microsoft Zero Day Quest](https://www.microsoft.com/en-us/msrc/blog/2025/08/zero-day-quest-join-the-largest-hacking-event-with-up-to-5-million-in-total-bounty-awards/)
- [Anthropic Model Safety Bug Bounty](https://www.anthropic.com/news/model-safety-bug-bounty)
- [Anthropic Fellows Program 2026](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)
- [Anthropic Economic Futures](https://www.anthropic.com/news/introducing-the-anthropic-economic-futures-program)
- [Amazon Nova AI Bounty](https://www.amazon.science/news/amazon-launches-private-ai-bug-bounty-to-strengthen-nova-models)
- [Amazon Nova AI Challenge](https://www.amazon.science/nova-ai-challenge/pushing-the-boundaries-of-secure-ai-winners-of-the-amazon-nova-ai-challenge)
- [Mozilla 0Din](https://0din.ai/)
- [0Din Mozilla Hacks](https://hacks.mozilla.org/2024/08/0din-a-genai-bug-bounty-program-securing-tomorrows-ai-together/)
- [Huntr Platform](https://huntr.com/)
- [NVIDIA + Intigriti](https://www.intigriti.com/blog/business-insights/intigriti-teams-with-nvidia-to-launch-bug-bounty-vulnerability-disclosure-program)
- [Adobe Firefly Bounty](https://blog.adobe.com/en/publish/2024/05/01/adobe-collaborates-with-ethical-hackers-build-safer-more-secure-ai-tools)
- [Salesforce Bug Bounty Results](https://www.salesforce.com/news/stories/bug-bounty-program-results-2024/)
- [xAI Security](https://x.ai/security)
- [Perplexity Security](https://www.perplexity.ai/hub/security)
- [Inflection on HackerOne](https://hackerone.com/inflection)
- [Databricks on HackerOne](https://hackerone.com/databricks)
- [DARPA SABER](https://www.darpa.mil/news/2025/saber-warfighter-ai)
- [DARPA AIxCC Results](https://www.darpa.mil/news/2025/aixcc-results)
- [AIxCC Winners](https://aicyberchallenge.com/finals-winners-announcement/)
- [DOD AI Bias Bounty](https://www.war.gov/News/Releases/Release/Article/3659519/)
- [NIST AI Cybersecurity Profile](https://csrc.nist.gov/pubs/ir/8596/iprd)
- [NIST AI Agent Standards](https://www.nist.gov/news-events/news/2026/02/announcing-ai-agent-standards-initiative-interoperable-and-secure)
- [CISA AI Roadmap](https://www.cisa.gov/resources-tools/resources/roadmap-ai)
- [UK AISI Evals Bounty](https://www.aisi.gov.uk/blog/evals-bounty)
- [Trump Cyber EO](https://www.cybersecuritydive.com/news/trump-cybersecurity-executive-order-eliminate-biden-programs/750119/)
- [Coefficient Giving AI Safety RFP](https://coefficientgiving.org/funds/navigating-transformative-ai/request-for-proposals-technical-ai-safety-research/)
- [Frontier Model Forum AISF](https://www.frontiermodelforum.org/ai-safety-fund/)
- [AISF New Grantees Dec 2025](https://www.frontiermodelforum.org/updates/announcement-of-new-ai-safety-fund-grantees/)
- [CSET Georgetown FRG](https://cset.georgetown.edu/foundational-research-grants/)
- [Foresight AI Safety Grants](https://foresight.org/grants/grants-ai-for-science-safety/)
- [AI Risk Mitigation Fund](https://www.airiskfund.com/)
- [SPAR AI](https://sparai.org/)
- [Meta RFPs](https://ai.meta.com/research/request-for-proposals/)
- [FAR.AI $30M Funding](https://www.far.ai/news/30m-multi-funder-support)
- [HackerOne AI Bug Bounty Guidance](https://docs.hackerone.com/en/articles/12570435-ai-bug-bounty)
- [Bugcrowd AI Solutions](https://www.bugcrowd.com/solutions/ai/)
- [Bugcrowd Bug Bounty List](https://www.bugcrowd.com/bug-bounty-list/)
- [AISafety.com Funding](https://www.aisafety.com/funding)
- [MLCommons AILuminate](https://mlcommons.org/ailuminate/safety/)
- [MLCommons Jailbreak Benchmark](https://mlcommons.org/2025/10/ailuminate-jailbreak-v05/)
- [NeurIPS 2025 Competitions](https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/)
- [CSO Online Top Bug Bounties 2025](https://www.csoonline.com/article/657751/top-bug-bounty-programs.html)
- [Cybernews AI Bug Bounty Analysis](https://cybernews.com/ai-news/was-2025-the-year-ai-broke-the-bug-bounty-model/)
- [Relevance AI VRP](https://relevanceai.com/vulnerability-reward-program)
