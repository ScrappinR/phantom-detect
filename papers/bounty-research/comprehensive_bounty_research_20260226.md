# Comprehensive Bug Bounty, VRP, Grant & Research Funding Research

**Date:** February 26, 2026
**Researcher:** Brian Rutherford
**Relevant Projects:** PHANTOM Protocol (AI covert channels, steganography in LLM outputs), MWRASP (quantum cybersecurity, 30+ provisional patents)
**Purpose:** Identify all active monetization and funding pathways for AI security and quantum cybersecurity research

---

## TABLE OF CONTENTS

1. [AI/ML Security Bug Bounties](#1-aiml-security-bug-bounties)
2. [Cybersecurity Grants & Funding](#2-cybersecurity-grants--funding)
3. [Quantum/Post-Quantum Specific Programs](#3-quantumpost-quantum-specific-programs)
4. [Other Relevant Programs](#4-other-relevant-programs)
5. [Alignment to PHANTOM & MWRASP](#5-alignment-scoring)
6. [Priority Action Matrix](#6-priority-action-matrix)

---

## 1. AI/ML SECURITY BUG BOUNTIES

### 1.1 Mozilla 0DIN (0din.ai)

- **URL:** [https://0din.ai/](https://0din.ai/)
- **Status:** OPEN (launched June 2024, ongoing)
- **Scope:** Guardrail jailbreaks, prompt injection attacks, interpreter jailbreaks across LLMs from OpenAI (GPT-4), Google (Gemini), Meta (LLaMA), Anthropic (Claude), and others. Specifically incentivizes research that falls outside the scope of other existing bounty programs.
- **Payout Range:** $500 -- $15,000 per vulnerability, based on impact, report quality, and timing
- **Submission Process:** Submit a high-level abstract of findings with affected model(s). Team responds within 3 business days on scope/likely bounty range. Validation within 2 weeks.
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** HIGH. Covert channel research, steganographic exfiltration via LLM outputs, and prompt injection are squarely in scope. 0DIN explicitly targets vulnerabilities that other programs exclude. This is a primary target.
- **MWRASP Alignment:** LOW. No quantum scope.
- **Sources:** [0din.ai](https://0din.ai/), [Mozilla Hacks announcement](https://hacks.mozilla.org/2024/08/0din-a-genai-bug-bounty-program-securing-tomorrows-ai-together/), [SecurityWeek](https://www.securityweek.com/mozilla-launches-0din-gen-ai-bug-bounty-program/)

---

### 1.2 Google AI Vulnerability Rewards Program

- **URL:** [https://bughunters.google.com/about/rules/google-friends/5222232590712832/ai-vulnerability-reward-program-rules](https://bughunters.google.com/about/rules/google-friends/5222232590712832/ai-vulnerability-reward-program-rules)
- **Status:** OPEN (expanded October 2025)
- **Scope:** Three tiers -- Flagship (Gemini Apps, Google Search, core Workspace apps), Standard (AI Studio, Jules, non-core Workspace), Other AI integrations. Covers data exfiltration, account/data modification attacks via AI products.
- **Payout Range:** Up to $20,000 base for flagship product attacks leading to victim account/data modifications. Up to $15,000 for data exfiltration from flagship products. Quality multipliers and novelty bonuses can push total payouts to $30,000.
- **Out of Scope:** Prompt injections, jailbreaks, and alignment issues are OUT of scope for the AI VRP.
- **Submission Process:** Via Google Bug Hunters platform
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** MODERATE. Data exfiltration through AI systems is in scope. Covert channel extraction of sensitive data via Gemini would qualify. However, prompt injection itself is excluded, which limits some PHANTOM research vectors.
- **MWRASP Alignment:** LOW. No quantum scope.
- **Sources:** [Google Bug Hunters rules](https://bughunters.google.com/about/rules/google-friends/5222232590712832/ai-vulnerability-reward-program-rules), [Infosecurity Magazine](https://www.infosecurity-magazine.com/news/google-launches-ai-bug-bounty/), [The Register](https://www.theregister.com/2025/10/07/google_ai_bug_bounty/)

---

### 1.3 OpenAI Bug Bounty (Bugcrowd)

- **URL:** [https://bugcrowd.com/openai](https://bugcrowd.com/openai)
- **Status:** OPEN (max payout increased March 2025)
- **Scope:** Security-related vulnerabilities in OpenAI systems and API key exposure. Model prompt/response content issues are OUT of scope unless they demonstrate a directly verifiable security impact on an in-scope service.
- **Payout Range:** $200 (low severity) up to $100,000 for "exceptional and differentiated" critical security vulnerabilities. Standard critical max is $20,000. Limited-time bonus promotions run periodically.
- **Submission Process:** Via Bugcrowd platform
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** MODERATE. If covert channel research demonstrates actual data exfiltration or security impact beyond model output manipulation, it could qualify. The "directly verifiable security impact" clause is the gate.
- **MWRASP Alignment:** LOW. No quantum scope.
- **Sources:** [Bugcrowd/OpenAI](https://bugcrowd.com/openai), [Dark Reading](https://www.darkreading.com/cybersecurity-operations/openai-bug-bounty-reward-100k), [Bleeping Computer](https://www.bleepingcomputer.com/news/security/openai-now-pays-researchers-100-000-for-critical-vulnerabilities/)

---

### 1.4 OpenAI GPT-5 Bio Bug Bounty

- **URL:** [https://openai.com/gpt-5-bio-bug-bounty/](https://openai.com/gpt-5-bio-bug-bounty/)
- **Status:** ACTIVE (invite-only, launched August 2025)
- **Scope:** GPT-5 only. Finding a universal jailbreaking prompt that answers all ten bio/chem safety questions from a clean chat without triggering moderation.
- **Payout Range:** $25,000 for first true universal jailbreak clearing all ten questions. $10,000 for first team answering all ten with multiple prompts. Smaller awards for partial wins at OpenAI's discretion.
- **Submission Process:** Application and invite-only. Vetted bio red-teamers. NDA required.
- **Deadline:** Rolling applications, testing began September 9, 2025
- **PHANTOM Alignment:** LOW-MODERATE. The jailbreak methodology is relevant to PHANTOM's understanding of safety bypass mechanisms, but scope is narrowly bio/chem focused.
- **MWRASP Alignment:** NONE.
- **Sources:** [OpenAI GPT-5 Bio Bounty](https://openai.com/gpt-5-bio-bug-bounty/), [VarIndia](https://www.varindia.com/news/find-a-gpt-5-jailbreak-and-win-25-000-from-openai)

---

### 1.5 Anthropic Model Safety Bug Bounty (HackerOne)

- **URL:** [https://hackerone.com/anthropic-vdp](https://hackerone.com/anthropic-vdp)
- **Status:** OPEN (ongoing, expanded to $35,000 max in 2025)
- **Scope:** Two programs: (1) VDP for technical infrastructure vulnerabilities, (2) Model Safety Bug Bounty for universal jailbreaks that bypass Constitutional Classifiers system. Currently testing on Claude Opus 4. Focuses on CBRN-related jailbreaks (bio, chem, radiological, nuclear) and cybersecurity.
- **Payout Range:** Up to $35,000 per novel, universal jailbreak identified. Escalated from $15K (Aug 2024) to $25K (May 2025) to $35K (current). Previous challenge paid out $55,000 total to four teams.
- **Submission Process:** Via HackerOne. Currently rolling applications -- once accepted, submit at any time. Previously invite-only, now accepting applications on rolling basis.
- **Red Teaming Results:** 300,000+ chat interactions from 339 participants in Feb 2025 public challenge. Next-gen Constitutional Classifiers (CC++) withstood 1,700 hours of red-teaming across 198,000 attempts with no universal jailbreak found.
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** HIGH. The cybersecurity domain is explicitly in scope. Universal jailbreaks that could exfiltrate data or bypass safety systems align with PHANTOM's covert channel research. Steganographic techniques for bypassing Constitutional Classifiers would be directly relevant.
- **MWRASP Alignment:** LOW. No quantum scope.
- **Sources:** [Anthropic safety bounty](https://www.anthropic.com/news/model-safety-bug-bounty), [Anthropic expanding bounty](https://www.anthropic.com/news/model-safety-bug-bounty), [HackerOne blog](https://www.hackerone.com/blog/anthropic-expands-their-model-safety-bug-bounty-program)

---

### 1.6 Microsoft Copilot (AI) Bounty Program

- **URL:** [https://www.microsoft.com/en-us/msrc/bounty-ai](https://www.microsoft.com/en-us/msrc/bounty-ai)
- **Status:** OPEN (expanded February 2025)
- **Scope:** Copilot for Telegram, Copilot for WhatsApp, copilot.microsoft.com, copilot.ai. Covers traditional web vulnerabilities plus AI-specific issues. Moderate severity issues now in scope.
- **Payout Range:** $250 -- $30,000. Moderate severity vulnerabilities up to $5,000 (new addition). Microsoft paid $17 million total in bounties across all programs in 2025.
- **Zero Day Quest:** Microsoft's largest hacking event returning spring 2026, focused on cloud and AI vulnerabilities. Inaugural April 2025 event paid $1.6 million. $4 million in potential total rewards.
- **Submission Process:** Via Microsoft Security Response Center (MSRC)
- **Deadline:** Rolling/ongoing. Zero Day Quest 2026 registration expected to open in spring.
- **PHANTOM Alignment:** MODERATE. Copilot data exfiltration, prompt injection leading to data leakage, and cross-context attacks would be relevant. Zero Day Quest is a significant opportunity.
- **MWRASP Alignment:** LOW. No quantum scope directly, but Azure cloud security is adjacent.
- **Sources:** [MSRC Copilot Bounty](https://www.microsoft.com/en-us/msrc/bounty-ai), [MSRC blog](https://msrc.microsoft.com/blog/2025/02/exciting-updates-to-the-copilot-ai-bounty-program-enhancing-security-and-incentivizing-innovation/), [The Register](https://www.theregister.com/2025/02/20/microsoft_copilot_bug_bounty_updated/)

---

### 1.7 Meta Bug Bounty (AI/LLM Scope)

- **URL:** [https://bugbounty.meta.com/](https://bugbounty.meta.com/)
- **Status:** OPEN
- **Scope:** Bugs in LLaMa models at github.com/facebookresearch/llama and codellama. Privacy/security issues associated with Meta's LLMs including training data extraction via model inversion or extraction attacks. Risky content generation reports accepted via separate channel.
- **Payout Range:** Not publicly specified for AI-specific findings. Meta paid $4 million total in 2025, $25 million cumulative.
- **Submission Process:** Via Meta Bug Bounty platform; risky content via developers.facebook.com/llama_output_feedback/; AUP violations via llamausereport@meta.com
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** MODERATE. Training data extraction and model inversion attacks are relevant to PHANTOM's data exfiltration research. LLaMa's open-source nature makes it a natural testbed.
- **MWRASP Alignment:** LOW.
- **Sources:** [Meta Bug Bounty scope](https://bugbounty.meta.com/scope/), [SecurityWeek](https://www.securityweek.com/meta-paid-out-4-million-via-bug-bounty-program-in-2025/)

---

### 1.8 Amazon AI Bug Bounty

- **URL:** [https://hackerone.com/amazonvrp](https://hackerone.com/amazonvrp)
- **Status:** PARTIALLY OPEN (private program expanding in early 2026)
- **Scope:** AI security in Amazon Nova models (powering Alexa, Amazon Bedrock). Jailbreaking, safety alignment, data poisoning attacks. Public program accepts reports under "Gen AI Apps" category.
- **Payout Range:** Private tournament: teams received $250,000 + AWS credits upfront; winners split additional $700,000. Public program: $55,000+ in validated AI findings to date.
- **Submission Process:** Public reports via HackerOne (select "Gen AI Apps"). Private continuous program by invitation, expanding early 2026.
- **Deadline:** Rolling; private invitations expanding early 2026
- **PHANTOM Alignment:** MODERATE. Data poisoning, jailbreaking, and safety alignment testing are relevant. Bedrock-based attacks could demonstrate covert channel feasibility.
- **MWRASP Alignment:** LOW.
- **Sources:** [CyberScoop](https://cyberscoop.com/amazon-bug-bounty-program-ai-nova/), [Amazon Science](https://www.amazon.science/news/amazon-launches-private-ai-bug-bounty-to-strengthen-nova-models)

---

### 1.9 Apple Security Bounty (Private Cloud Compute / Apple Intelligence)

- **URL:** [https://security.apple.com/bounty/](https://security.apple.com/bounty/)
- **Status:** OPEN (major expansion November 2025)
- **Scope:** Private Cloud Compute (PCC) -- the cloud AI infrastructure behind Apple Intelligence. Three PCC bounty categories: accidental data disclosure, external compromise from user requests, physical/internal access compromise. Also covers all iOS, macOS, etc. with updated categories.
- **Payout Range:** Up to $2 million for zero-click exploit chains. Bonus system can push max beyond $5 million. PCC rewards comparable to iOS-level payouts. Median awards ~$18,000, top payouts $500,000+.
- **Security Research Device Program:** 2026 program applications opened, offering unlocked iPhones for security research. SRD gives shell access, custom kernels, custom entitlements.
- **Submission Process:** Via Apple Security Research portal
- **Deadline:** Rolling. SRDP 2026 application deadline was October 31, 2025.
- **PHANTOM Alignment:** MODERATE-HIGH. PCC data exfiltration and unauthorized access to AI cloud infrastructure are directly relevant. Covert channels in Apple Intelligence's cloud processing pipeline would be a significant finding.
- **MWRASP Alignment:** LOW.
- **Sources:** [Apple Security Bounty evolved](https://security.apple.com/blog/apple-security-bounty-evolved/), [Apple PCC research](https://security.apple.com/blog/pcc-security-research/), [SecurityWeek SRDP](https://www.securityweek.com/apple-seeks-researchers-for-2026-iphone-security-program/)

---

### 1.10 NVIDIA Vulnerability Disclosure Program

- **URL:** [https://app.intigriti.com/programs/nvidia/nvidiavdp/detail](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail)
- **Status:** LAUNCHING (partnership with Intigriti, late 2025; private programs rolling out over 6 months)
- **Scope:** NVIDIA products including AI infrastructure. VDP covers all NVIDIA assets. Additional private bug bounty packages to cover core AI elements. Community of 125,000+ ethical hackers.
- **Payout Range:** Not yet publicly specified for the new Intigriti program. Previous acknowledgment-only via PSIRT.
- **Submission Process:** Via Intigriti platform or nvidia.com/product-security/report-vulnerability/
- **Deadline:** Programs launching over next 6 months from late 2025
- **PHANTOM Alignment:** MODERATE. NVIDIA's AI infrastructure (GPUs, CUDA, TensorRT, NeMo) is foundational to LLM deployment. Side-channel attacks on GPU inference would be novel.
- **MWRASP Alignment:** LOW.
- **Sources:** [Intigriti/NVIDIA partnership](https://www.intigriti.com/blog/business-insights/intigriti-teams-with-nvidia-to-launch-bug-bounty-vulnerability-disclosure-program), [NVIDIA Product Security](https://www.nvidia.com/en-us/product-security/)

---

### 1.11 Hugging Face / Huntr (Protect AI)

- **URL:** [https://huntr.com/](https://huntr.com/)
- **Status:** OPEN (world's first dedicated AI/ML bug bounty platform)
- **Scope:** Two programs: Model File Vulnerabilities (MFV) -- attacks at model load time or inference, backdoor manipulations; Open Source Vulnerabilities (OSV) -- security flaws in 125+ ML supply chain repositories. Covers AI/ML packages, libraries, frameworks, foundation models.
- **Payout Range:** Up to $50,000 for critical vulnerabilities. Monthly payouts on the 25th via Stripe Connect. Monthly contests with additional prizes.
- **Key Results:** 17,000+ security researchers. 4.47 million model versions scanned on Hugging Face Hub. 352,000 unsafe/suspicious issues identified across 51,700 models. First contest focused on Hugging Face Transformers with $50,000 prize.
- **Submission Process:** Via huntr.com platform. Both researchers and maintainers are paid.
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** HIGH. Model file vulnerabilities, backdoor manipulation, and ML supply chain attacks are directly relevant to PHANTOM's covert channel and steganography research. Embedding covert data in model weights or using inference-time channels aligns perfectly.
- **MWRASP Alignment:** LOW.
- **Sources:** [Huntr](https://huntr.com/), [Protect AI acquisition](https://www.businesswire.com/news/home/20230808746694/en/Protect-AI-Acquires-huntr-Launches-Worlds-First-Artificial-Intelligence-and-Machine-Learning-Bug-Bounty-Platform), [IEEE S&P 2025 poster](https://sp2025.ieee-security.org/downloads/posters/sp25posters-final9.pdf)

---

### 1.12 xAI (Grok) Bug Bounty

- **URL:** [https://hackerone.com/x](https://hackerone.com/x)
- **Status:** OPEN (but process criticized)
- **Scope:** System security vulnerabilities in xAI products including Grok. Reports via HackerOne or vulnerabilities@x.ai.
- **Payout Range:** Not publicly specified. Reports of small bonuses for duplicate/regression findings.
- **Known Issues:** Criticized for lack of transparent disclosure process. GitGuardian exposed leaked API key providing access to unreleased Grok models; xAI's response was inadequate.
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** LOW-MODERATE. Grok is a valid LLM target but the program's operational maturity is questionable.
- **MWRASP Alignment:** NONE.
- **Sources:** [HackerOne/X](https://hackerone.com/x), [GitGuardian disclosure](https://blog.gitguardian.com/xai-secret-leak-disclosure/)

---

### 1.13 Cohere Bug Bounty

- **URL:** [https://cohere.com/security](https://cohere.com/security)
- **Status:** OPEN
- **Scope:** Security vulnerabilities in Cohere systems. Researchers report potential vulnerabilities and may receive rewards.
- **Payout Range:** Not publicly specified.
- **Submission Process:** Inquire via security@cohere.com
- **Deadline:** Rolling/ongoing
- **PHANTOM Alignment:** MODERATE. Cohere's enterprise AI platform processes sensitive business data; covert exfiltration channels would be high-impact.
- **MWRASP Alignment:** NONE.
- **Sources:** [Cohere Security](https://cohere.com/security)

---

### 1.14 DeepSeek

- **URL:** No formal program exists
- **Status:** NO FORMAL BUG BOUNTY PROGRAM
- **Known Issues:** In January 2025, Wiz Research discovered an unauthenticated, publicly accessible ClickHouse database exposing 1M+ sensitive log entries (chat histories, API keys, backend details). DeepSeek has responded to direct vulnerability reports via service@deepseek.com within 24 hours but has no formal bounty.
- **PHANTOM Alignment:** HIGH research target (not bounty target). DeepSeek's security posture makes it relevant for demonstrating covert channel risks, but there's no formal payout mechanism.
- **Sources:** [Wiz bug bounty masterclass](https://www.wiz.io/bug-bounty-masterclass/real-world-hacks/real-world-hacks-1-open-deepseek-database), [Theori blog](https://theori.io/blog/deepseek-security-privacy-and-governance-hidden-risks-in-open-source-ai)

---

### 1.15 Mistral AI

- **Status:** NO CONFIRMED PUBLIC BUG BOUNTY PROGRAM as of February 2026. Search returned no evidence of a formal VDP or bounty program.

---

### 1.16 Gray Swan AI Arena

- **URL:** [https://app.grayswan.ai/arena/](https://app.grayswan.ai/arena/)
- **Status:** PERIODIC COMPETITIONS
- **Scope:** Indirect prompt injection challenges. Tests how agents respond to hidden prompts in web content, tool outputs, coding agent traces. Co-sponsored by Anthropic, Meta, OpenAI, Amazon.
- **Payout Range:** $40,000 prize pool (November 2025 competition)
- **Deadline:** Competition-based; watch for future announcements
- **PHANTOM Alignment:** HIGH. Indirect prompt injection in multi-agent systems is core PHANTOM territory. Hidden prompts triggering unintended behaviors is precisely the covert channel attack surface.
- **Sources:** [Gray Swan Arena](https://app.grayswan.ai/arena/challenge/indirect-prompt-injection/rules)

---

## 2. CYBERSECURITY GRANTS & FUNDING

### 2.1 OpenAI Cybersecurity Grant Program

- **URL:** [https://openai.com/form/cybersecurity-grant-program/](https://openai.com/form/cybersecurity-grant-program/)
- **Status:** OPEN (major expansion February 2026 -- $10M in API credits)
- **Scope:** Defensive cybersecurity applications of AI. Priority areas: software patching (AI for vulnerability detection/patching), model privacy (robustness against unintended exposure of private training data), detection and response (capabilities against APTs), security integration (AI integration with security tools). Offensive projects NOT considered.
- **Funding Amount:** Grants in increments of $10,000 from initial $1M fund (API credits, direct funding, equivalents). Now committing $10 million in API credits. New microgrants for rapid prototyping. 28 initiatives funded from 1,000+ applications so far.
- **Trusted Access for Cyber:** New pilot -- identity and trust-based framework for enhanced cyber capabilities. Enterprises can request access; researchers can apply for invite-only program with more permissive models.
- **Submission Process:** Rolling applications via openai.com/form/cybersecurity-grant-program/. Strong preference for practical defensive applications. Must be intended for maximal public benefit.
- **Eligibility:** Broad -- individuals, corporate teams, academic institutions, research organizations worldwide.
- **Deadline:** Rolling
- **PHANTOM Alignment:** HIGH. Model privacy (preventing data exfiltration) and detection/response (detecting covert channels) are priority areas. PHANTOM's defensive detection capabilities for steganographic AI outputs would fit the "detection and response" category well.
- **MWRASP Alignment:** LOW-MODERATE. Post-quantum encryption integration with AI security tools could be framed as "security integration."
- **Sources:** [OpenAI Cybersecurity Grant](https://openai.com/index/openai-cybersecurity-grant-program/), [Trusted Access for Cyber](https://openai.com/index/trusted-access-for-cyber/)

---

### 2.2 DARPA AI Cyber Challenge (AIxCC) -- Transition Phase

- **URL:** [https://aicyberchallenge.com/](https://aicyberchallenge.com/)
- **Status:** COMPETITION COMPLETED; now in TRANSITION PHASE (2025-2026)
- **Scope:** Autonomous Cyber Reasoning Systems (CRS) for vulnerability detection and patching in open-source critical infrastructure software. $29.5M in cumulative prizes awarded.
- **Transition Funding:** DARPA and ARPA-H adding $1.4M in prizes for integration into real-world critical infrastructure software. All finalist CRSs being released as open-source.
- **Relevance:** Competition phase is over. New entrants cannot compete. But the transition to real-world deployment may present consulting/integration opportunities.
- **PHANTOM Alignment:** LOW (competition closed). The CRS tools could be targets for PHANTOM research (adversarial attacks on autonomous security systems).
- **Sources:** [DARPA AIxCC](https://www.darpa.mil/research/programs/ai-cyber), [AIxCC results](https://www.darpa.mil/news/2025/aixcc-results)

---

### 2.3 NSF Security, Privacy, and Trust in Cyberspace (SaTC 2.0)

- **URL:** [https://www.nsf.gov/funding/opportunities/satc-20-security-privacy-trust-cyberspace/nsf25-515/solicitation](https://www.nsf.gov/funding/opportunities/satc-20-security-privacy-trust-cyberspace/nsf25-515/solicitation)
- **Status:** OPEN (accepts proposals on rolling basis)
- **Scope:** Trust in cyber ecosystems -- security, privacy, resilience. Both technical vulnerabilities and social dimensions. AI security explicitly relevant.
- **Funding:**
  - RES (Research): up to $1,200,000, 4 years
  - EDU (Education): up to $500,000, 3 years (currently under revision -- not accepting EDU proposals)
  - SEED: up to $300,000, 2 years
- **Deadline:** Rolling target dates: September 29, 2025 and January 26, 2026 (last Monday in January, annually thereafter). Proposals accepted anytime but may miss panel meetings.
- **PHANTOM Alignment:** HIGH. Covert channels in AI systems, steganographic communication between LLM agents, and AI-enabled data exfiltration are squarely within SaTC's scope.
- **MWRASP Alignment:** HIGH. Post-quantum cryptographic implementations, quantum-resistant authentication, and quantum cybersecurity research all fit.
- **CAVEAT:** NSF funding is subject to current federal appropriations lapse (since February 1, 2026). Grants.gov still accepts applications but processing may be delayed.
- **Sources:** [NSF SaTC 2.0](https://www.nsf.gov/funding/opportunities/satc-20-security-privacy-trust-cyberspace/nsf25-515/solicitation)

---

### 2.4 NSF SaTC Frontiers (Center-Scale)

- **URL:** [https://www.nsf.gov/funding/opportunities/satc-frontiers-secure-trustworthy-cyberspace-frontiers/](https://www.nsf.gov/funding/opportunities/satc-frontiers-secure-trustworthy-cyberspace-frontiers/)
- **Status:** CHECK FOR CURRENT SOLICITATION
- **Scope:** Ambitious, transformative center-scale cybersecurity and privacy projects
- **Funding:** $5,000,000 -- $10,000,000, up to 5 years
- **PHANTOM Alignment:** HIGH (if partnering with a university)
- **MWRASP Alignment:** HIGH (quantum cybersecurity center concept)
- **Caveat:** Requires institutional partnerships. Not suitable as solo researcher.

---

### 2.5 NSF Cybersecurity Innovation for Cyberinfrastructure (CICI)

- **URL:** [https://www.nsf.gov/funding/opportunities/cici-cybersecurity-innovation-cyberinfrastructure/nsf25-531/solicitation](https://www.nsf.gov/funding/opportunities/cici-cybersecurity-innovation-cyberinfrastructure/nsf25-531/solicitation)
- **Status:** OPEN
- **Scope:** Four tracks including **IPAAI (Integrity, Provenance, and Authenticity for AI-Ready Data)** -- directly relevant to AI security
- **Funding:** $8M--$12M total. IPAAI awards up to $900,000 for up to 3 years. UCSS/RSSD up to $600,000. TCR up to $1,200,000.
- **PHANTOM Alignment:** HIGH. The IPAAI track targeting data integrity and provenance for AI systems directly maps to detecting steganographic manipulation and covert channels in AI data pipelines.
- **MWRASP Alignment:** MODERATE. Quantum-secure cyberinfrastructure protection could fit TCR track.
- **Sources:** [NSF CICI](https://www.nsf.gov/funding/opportunities/cici-cybersecurity-innovation-cyberinfrastructure/nsf25-531/solicitation)

---

### 2.6 NSF SBIR/STTR -- Quantum Information Technologies

- **URL:** [https://seedfund.nsf.gov/topics/quantum-information-technologies/](https://seedfund.nsf.gov/topics/quantum-information-technologies/)
- **Status:** PAUSED (Project Pitches paused December 22, 2025 due to SBIR congressional authorization lapse)
- **Scope:** Innovations in information/communications technologies that rely on quantum mechanical properties -- generation, detection, manipulation of quantum states for faster/more secure information processing.
- **Funding:** Phase I: up to $305,000. Phase II: up to $1,250,000 over 24 months + supplemental funding potentially exceeding $500,000 additional.
- **Eligibility:** 50%+ US citizen/permanent resident equity. PI employed 20+ hrs/week. All work in US.
- **Estimated Reopening:** March 2026 (contingent on congressional reauthorization)
- **PHANTOM Alignment:** LOW.
- **MWRASP Alignment:** CRITICAL. This is the primary federal SBIR pathway for quantum cybersecurity/authentication technology. MWRASP's 30+ provisionals in quantum authentication map directly to NSF's QIT topic. Monitor closely for reopening.
- **Reauthorization Status:** Senate deal reportedly near passage (Feb 25, 2026) to extend SBIR/STTR through September 30, 2031. New security screening provisions for foreign risk evaluation.
- **Sources:** [NSF SBIR QIT](https://seedfund.nsf.gov/topics/quantum-information-technologies/), [SBIR.org reauthorization](https://sbir.org/news/sbir-restart-draft-bill-proposal-caps/)

---

### 2.7 NIST SBIR Phase II Awards

- **URL:** [https://www.nist.gov/news-events/news/2026/02/nist-allocates-over-3-million-small-businesses-advancing-ai-biotechnology](https://www.nist.gov/news-events/news/2026/02/nist-allocates-over-3-million-small-businesses-advancing-ai-biotechnology)
- **Status:** AWARDS MADE (February 2026)
- **Scope:** NIST allocated $3.19M to eight small businesses for AI, biotech, semiconductors, quantum, and other key technologies. Phase II prototyping awards, 24-month period.
- **MWRASP Alignment:** HIGH. NIST is actively funding quantum and AI via SBIR. Monitor for future solicitations.
- **Sources:** [NIST SBIR](https://www.nist.gov/news-events/news/2026/02/nist-allocates-over-3-million-small-businesses-advancing-ai-biotechnology)

---

### 2.8 DHS SBIR Program

- **URL:** [https://www.dhs.gov/science-and-technology/sbir](https://www.dhs.gov/science-and-technology/sbir)
- **Status:** NO NEW SOLICITATIONS (authorization expired September 30, 2025)
- **Most Recent:** FY25 solicitation closed January 21, 2025. Included cybersecurity topics.
- **Phase III:** Still possible for companies with prior Phase I/II awards (Phase III uses non-SBIR funds).
- **Funding:** Phase I: up to $175,000 for 5 months.
- **Reauthorization:** Same SBIR reauthorization deal as NSF (see 2.6 above).
- **PHANTOM Alignment:** MODERATE (when reauthorized). DHS/CISA cybersecurity topics historically include AI security.
- **MWRASP Alignment:** MODERATE. DHS cybersecurity topics could include quantum-resistant infrastructure.
- **Sources:** [DHS SBIR](https://www.dhs.gov/science-and-technology/sbir)

---

### 2.9 CISA State and Local Cybersecurity Grant Program (SLCGP)

- **URL:** [https://www.cisa.gov/cybergrants](https://www.cisa.gov/cybergrants)
- **Status:** FY2025 is FINAL YEAR of original $1B four-year appropriation
- **Scope:** State, local, territorial government cybersecurity improvement. Not directly applicable to private research.
- **Funding:** $91.7M in FY2025. $1B total over four years (2022-2025).
- **PHANTOM/MWRASP Alignment:** LOW. Government-to-government grants, not applicable to private research companies.
- **Sources:** [CISA Cyber Grants](https://www.cisa.gov/cybergrants)

---

### 2.10 Horizon Europe -- ECCC Cybersecurity Calls

- **URL:** [https://cybersecurity-centre.europa.eu/](https://cybersecurity-centre.europa.eu/)
- **Status:** 2025 CALLS CLOSED (November 12, 2025 deadline); 2026 calls forthcoming
- **Scope:** EUR 90.55 million total across six topics:
  - **Generative AI for Cybersecurity applications** (EUR 40M)
  - New advanced tools for Operational Cybersecurity (EUR 23.55M)
  - Privacy Enhancing Technologies (EUR 11M)
  - **Security evaluations of PQC primitives** (EUR 4M)
  - **Security of PQC algorithm implementations** (EUR 6M)
  - **Integration of PQC algorithms into high-level protocols** (EUR 6M)
- **PHANTOM Alignment:** HIGH. EUR 40M for "Generative AI for Cybersecurity" is directly relevant. Detection of covert channels in generative AI outputs would fit.
- **MWRASP Alignment:** CRITICAL. EUR 16M specifically for PQC security evaluations and implementations. MWRASP's quantum authentication patents could inform PQC integration research.
- **Eligibility Note:** Generally requires EU-based consortium participation. US companies can sometimes participate as third-country partners but cannot lead.
- **Sources:** [ECCC funding](https://cybersecurity-centre.europa.eu/news/new-eccc-funding-opportunities-under-digital-europe-and-horizon-europe-programmes-are-open-2025-06-12_en), [Horizon Europe 2026-2027 guide](https://www.grantsfinder.eu/blog/horizon-europe-2026-2027-complete-guide)

---

### 2.11 Digital Europe Programme -- Post-Quantum PKI

- **Status:** 2025 CALL CLOSED (October 7, 2025)
- **Scope:** Transition to post-quantum Public Key Infrastructures (EUR 15M)
- **MWRASP Alignment:** HIGH. Watch for 2026-2027 follow-on calls.
- **Sources:** [ECCC funding](https://cybersecurity-centre.europa.eu/news/new-eccc-funding-opportunities-under-digital-europe-and-horizon-europe-programmes-are-open-2025-06-12_en)

---

### 2.12 UK AI Security Institute (AISI) Grants

- **URL:** [https://www.aisi.gov.uk/grants](https://www.aisi.gov.uk/grants)
- **Status:** MULTIPLE PROGRAMS

**Challenge Fund:**
- Budget: GBP 5M total
- Grant size: GBP 50,000 -- GBP 200,000 per project
- Deadline: Open until **March 31, 2026, 11:59am** (UPCOMING)
- Scope: Safeguards, control, alignment, societal resilience. Also welcomes innovative topics on safe/secure AI development.
- 100+ applications received. Monthly payments in arrears.
- **PHANTOM Alignment:** HIGH. Steganographic collusion detection, covert channel safeguards, and AI control mechanisms are explicitly relevant to AISI's priority research areas.

**Systemic AI Safety Grants:**
- Budget: GBP 8M
- Grant size: Up to GBP 200,000 (Phase 1 seed grants)
- Status: Phase 1 CLOSED. 20 projects funded from 300+ applications.
- Scope: Deepfakes/misinformation, cybersecurity against AI-driven attacks, AI system failures in critical sectors.
- **PHANTOM Alignment:** HIGH.

**Alignment Project:**
- Budget: GBP 27M+ (up from GBP 15M at launch)
- Contributors: OpenAI (GBP 5.6M), Microsoft, Anthropic, AWS, CIFAR, Schmidt Sciences, Canadian/Australian AI Safety Institutes
- Status: Round 1 CLOSED (60 projects funded from 800+ applications, 466 institutions, 42 countries). Round 2 expected later 2026.
- **PHANTOM Alignment:** MODERATE-HIGH. AI alignment research intersects with steganographic collusion prevention.

- **MWRASP Alignment:** LOW.
- **Sources:** [AISI Grants](https://www.aisi.gov.uk/grants), [AISI Challenge Fund](https://www.aisi.gov.uk/work/new-updates-to-the-aisi-challenge-fund), [Alignment Project](https://alignmentproject.aisi.gov.uk/)

---

### 2.13 Open Philanthropy -- Technical AI Safety Research

- **URL:** [https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/](https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/)
- **Status:** MAIN RFP CLOSED (April 15, 2025). Future RFPs expected.
- **Scope:** 21 research areas including robust unlearning, alignment, and model behavior. Funded ~$40M in grants. Median grant ~$257K, average ~$1.67M.
- **Eligibility:** Broad -- from API credits to seed funding for new research orgs.
- **PHANTOM Alignment:** MODERATE-HIGH. "Robust unlearning" (removing dangerous knowledge from model weights) and model behavior monitoring are relevant. Steganographic encoding detection would fit.
- **MWRASP Alignment:** LOW.
- **Sources:** [Open Philanthropy RFP](https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/), [Research areas](https://www.openphilanthropy.org/tais-rfp-research-areas/)

---

### 2.14 AI Safety Fund (Frontier Model Forum)

- **URL:** [https://www.frontiermodelforum.org/ai-safety-fund/](https://www.frontiermodelforum.org/ai-safety-fund/)
- **Status:** CHECK FOR CURRENT ROUND
- **Scope:** Three priority areas: Biosecurity, **Cybersecurity**, and AI Agent Evaluation & Synthetic Content
- **Funding:** Grants up to $500,000
- **Eligibility:** Academic labs, non-profits, independent researchers, mission-driven entities
- **PHANTOM Alignment:** HIGH. Cybersecurity is explicitly one of three priority areas. AI agent evaluation connects to multi-agent covert channel research.
- **MWRASP Alignment:** LOW.
- **Sources:** [AI Safety Fund](https://www.frontiermodelforum.org/ai-safety-fund/)

---

### 2.15 Survival and Flourishing Fund (SFF)

- **Status:** Two funding rounds per year
- **Scope:** AI safety research organizations. Second largest AI safety funder after Open Philanthropy.
- **Funding:** ~$30M/year on AI safety (2023 figure). Funded ARC Evals, FAR AI, Redwood Research, Center for Governance of AI.
- **Funder:** Jaan Tallinn (~$900M net worth)
- **PHANTOM Alignment:** MODERATE. Organizational funding for AI safety research groups.
- **Sources:** [LessWrong funding overview](https://www.lesswrong.com/posts/WGpFFJo2uFe5ssgEb/an-overview-of-the-ai-safety-funding-situation)

---

### 2.16 YesWeHack / European Commission Bug Bounty

- **URL:** [https://yeswehack.com/programs](https://yeswehack.com/programs)
- **Status:** OPEN (four-year framework contract, up to EUR 7.68M)
- **Scope:** Bug bounties and VDPs for EU open-source assets and applications. European Commission's preferred provider. Expanding to wider range of open-source projects. Jenkins first selected project.
- **PHANTOM Alignment:** LOW-MODERATE. If EU institutions use AI-based tools, they could come into scope.
- **Sources:** [YesWeHack EU tender](https://www.yeswehack.com/news/european-commission-tender-won-yeswehack)

---

## 3. QUANTUM/POST-QUANTUM SPECIFIC PROGRAMS

### 3.1 NIST Post-Quantum Cryptography Standardization

- **URL:** [https://csrc.nist.gov/Projects/post-quantum-cryptography](https://csrc.nist.gov/Projects/post-quantum-cryptography)
- **Status:** STANDARDS FINALIZED; ongoing evaluation of additional algorithms
- **Standards Released:** FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA) -- August 2024. HQC selected March 2025 as backup.
- **No Bounty Program:** NIST does not operate a bounty for PQC. The original call for proposals closed in 2017.
- **Ongoing Work:** Additional digital signature schemes under evaluation. Draft SP 800-227 and IR 8547 released for public comment.
- **CAVEAT:** NIST PQC website was not being updated as of October 1, 2025 due to federal funding lapse.
- **MWRASP Alignment:** CRITICAL. MWRASP's quantum authentication patents should be positioned against NIST PQC standards. No direct funding pathway here, but the standards define the market.
- **Sources:** [NIST PQC](https://www.nist.gov/pqc), [NIST standardization](https://csrc.nist.gov/Projects/post-quantum-cryptography/post-quantum-cryptography-standardization)

---

### 3.2 NSA CNSA 2.0 Transition Timeline

- **URL:** [https://media.defense.gov/2025/May/30/2003728741/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS.PDF](https://media.defense.gov/2025/May/30/2003728741/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS.PDF)
- **Status:** MANDATORY TRANSITION UNDERWAY
- **Key Deadlines:**
  - Software/firmware signing: Support CNSA 2.0 by 2025, exclusive by 2030
  - Web browsers/servers/cloud: Support CNSA 2.0 by 2025, exclusive by 2033
  - Traditional networking (VPNs, routers): Support by 2026, exclusive by 2030
  - **January 1, 2027:** All new NSS equipment acquisitions must be CNSA 2.0 compliant
  - 2035: All NSS quantum-resistant
- **Approved Algorithms:** CRYSTALS-Kyber (key encapsulation), CRYSTALS-Dilithium (digital signatures), LMS/XMSS (code signing)
- **No Direct Funding:** NSA does not publicly offer grants for PQC research. However, the compliance mandate creates a massive market for PQC products.
- **MWRASP Alignment:** CRITICAL. The CNSA 2.0 compliance mandate directly creates demand for quantum-resistant authentication and cybersecurity products. MWRASP patents in quantum authentication are positioned exactly where federal procurement dollars will flow. The January 2027 deadline for new NSS equipment is a forcing function.
- **Sources:** [NSA CNSA 2.0](https://www.nsa.gov/Press-Room/News-Highlights/Article/Article/3148990/nsa-releases-future-quantum-resistant-qr-algorithm-requirements-for-national-se/), [Encryption Consulting](https://www.encryptionconsulting.com/quantum-proof-with-cnsa-2-0/)

---

### 3.3 DARPA Quantum Benchmarking Initiative (QBI) 2026

- **URL:** [https://www.darpa.mil/research/programs/quantum-benchmarking-initiative](https://www.darpa.mil/research/programs/quantum-benchmarking-initiative)
- **Status:** OPEN SOLICITATION (new entrants welcome)
- **Scope:** Building industrially fault-tolerant quantum computers. Rigorous verification and validation of quantum computing approaches. Goal: utility-scale quantum computer by 2033.
- **Stage B:** 11 companies selected. Developing R&D plans and risk-reduction prototypes.
- **Submission Deadline:** November 14, 2026 (QBI 2026 solicitation via SAM.gov)
- **MWRASP Alignment:** MODERATE. MWRASP's quantum authentication work is adjacent but QBI is focused on quantum computing hardware, not quantum cybersecurity applications.
- **Sources:** [DARPA QBI](https://www.darpa.mil/research/programs/quantum-benchmarking-initiative), [QBI 2026 SAM.gov](https://sam.gov/workspace/contract/opp/c26b38bf041a4d70b00ed619bebb4773/view)

---

### 3.4 DOE National Quantum Information Science Research Centers

- **URL:** [https://www.energy.gov/articles/energy-department-announces-625-million-advance-next-phase-national-quantum-information](https://www.energy.gov/articles/energy-department-announces-625-million-advance-next-phase-national-quantum-information)
- **Status:** AWARDS MADE ($625M, 5-year renewal)
- **Scope:** Five centers at Brookhaven, Argonne, Lawrence Berkeley, Oak Ridge, Fermi National Labs. Focus: quantum computing, communication, sensing, materials/chemistry, foundries.
- **Funding:** $625M total ($125M FY2025; outyears contingent on appropriations)
- **Eligibility:** Must be led by DOE National Laboratories. Other entities proposed as subawardees.
- **MWRASP Alignment:** MODERATE. Potential subawardee role for quantum communication/authentication research. Requires national lab partnership.
- **Sources:** [DOE announcement](https://www.energy.gov/articles/energy-department-announces-625-million-advance-next-phase-national-quantum-information)

---

### 3.5 QED-C (Quantum Economic Development Consortium)

- **URL:** [https://quantumconsortium.org](https://quantumconsortium.org)
- **Status:** ACTIVE (managed by SRI International under NIST cooperative agreement)
- **Scope:** Connecting government, academia, private sector to grow US quantum industry. Technical Advisory Committees for quantum computing, sensing, networking/communications. Standards development, benchmarking, workforce development.
- **Relevance:** Membership provides access to use-case development, benchmarking, and industry connections. Not a direct funding source but a pathway to partnerships and standards influence.
- **MWRASP Alignment:** HIGH for networking/positioning. QED-C membership could connect MWRASP patents to industry partners and federal procurement opportunities.
- **Sources:** [QED-C](https://quantumconsortium.org)

---

### 3.6 NSF National Quantum and Nanotechnology Infrastructure (NQNI)

- **URL:** [https://www.nsf.gov/funding/opportunities/nqni-national-quantum-nanotechnology-infrastructure/nsf26-505/solicitation](https://www.nsf.gov/funding/opportunities/nqni-national-quantum-nanotechnology-infrastructure/nsf26-505/solicitation)
- **Status:** OPEN (Letters of Intent due March 16, 2026)
- **Scope:** Nationwide network of open-access research facilities for quantum and nanoscale technologies
- **Funding:** $12--$20M/year for FY2026--2030. Individual awards $500K--$2M/year for 5 years (renewable to 8-16 years).
- **MWRASP Alignment:** MODERATE. Infrastructure rather than direct research, but could support quantum authentication testing facilities.
- **Sources:** [NSF NQNI](https://www.nsf.gov/funding/opportunities/nqni-national-quantum-nanotechnology-infrastructure/nsf26-505/solicitation)

---

### 3.7 Navy/Marine Corps BAA for Quantum Information Science

- **Status:** OPEN (accepting full proposals until September 30, 2026)
- **Scope:** Basic and applied research in quantum information science
- **MWRASP Alignment:** HIGH. Navy/Marine Corps QIS research directly relevant to defense applications of quantum cybersecurity.
- **Sources:** [DARPA quantum landscape overview](https://www.sc.edu/about/offices_and_divisions/provost/docs/emerging-technology-series-quantum-information-science-landscape-2025.pdf)

---

### 3.8 Post-Quantum Cybersecurity Standards Act (H.R.3259)

- **Status:** PENDING LEGISLATION (119th Congress, 2025-2026)
- **Scope:** Would authorize NIST to provide technical assistance through grants to entities at high risk of quantum cryptoanalytic attacks, for adopting PQC standards and remediating quantum vulnerabilities.
- **MWRASP Alignment:** CRITICAL. If enacted, this creates a direct grant pathway for PQC adoption assistance. MWRASP's patent portfolio could underpin products that help organizations transition.
- **Sources:** [H.R.3259](https://www.congress.gov/bill/119th-congress/house-bill/3259/text)

---

## 4. OTHER RELEVANT PROGRAMS

### 4.1 Internet Bug Bounty (IBB)

- **URL:** [https://hackerone.com/ibb](https://hackerone.com/ibb)
- **Status:** TECHNICALLY ACTIVE but operationally troubled. Reports of 8+ month payment delays. Curl project left after AI slop report flood. HackerOne claims "temporary operational backlog" and expects to resume regular payouts by end of Q1 2026.
- **Scope:** Open-source software vulnerabilities. 80/20 bounty split (researcher/maintainer). 1,000+ flaws uncovered since 2013.
- **PHANTOM/MWRASP Alignment:** LOW. General open-source rather than AI-specific.
- **Sources:** [HackerOne IBB](https://hackerone.com/ibb), [The Register](https://www.theregister.com/2026/01/07/hackerone_ghosted_researcher/), [Daniel Stenberg curl blog](https://daniel.haxx.se/blog/2026/01/26/the-end-of-the-curl-bug-bounty/)

---

### 4.2 IARPA Programs (AI Security & Quantum)

- **URL:** [https://www.iarpa.gov/index.php/research-programs](https://www.iarpa.gov/index.php/research-programs)
- **Status:** ACTIVE (Proposers' Day January 7, 2026 -- 550+ researchers attended)
- **Scope:** AI Trojan attack detection, quantum computing, machine learning, synthetic biology. Approximately 90 programs since inception. 1,000+ publications, dozens of patents from IARPA-funded quantum work.
- **PHANTOM Alignment:** MODERATE-HIGH. AI Trojan detection program is directly relevant to covert channel research.
- **MWRASP Alignment:** HIGH. IARPA has led government quantum investment since 2009.
- **How to Engage:** Monitor SAM.gov for BAAs. Attend Proposers' Days.
- **Sources:** [IARPA programs](https://www.iarpa.gov/index.php/research-programs), [IARPA about](https://www.iarpa.gov/who-we-are/about-us)

---

### 4.3 Cybersecurity and AI Talent Initiative

- **Status:** ACTIVE
- **Scope:** Public-private partnership for cybersecurity workforce. Up to $75,000 in student loan assistance.
- **PHANTOM/MWRASP Alignment:** LOW (workforce program, not research funding).

---

### 4.4 Academic Conference Prizes

The following top-tier security conferences accept AI security papers and award prizes:

- **IEEE S&P (Oakland):** Distinguished Paper Awards. Held annually in May. One of the "Big Four" security conferences. Increasingly accepting AI/ML security papers.
- **USENIX Security:** Distinguished Paper Awards. Held annually in August. Strong AI security track.
- **NeurIPS:** Published steganographic collusion research (Motwani et al., 2024). Top-tier ML venue.
- **NDSS (Network and Distributed System Security):** Annual conference with growing AI security focus.
- **ACM CCS (Computer and Communications Security):** Another "Big Four" venue.

Prize values are typically prestige-based (recognition, not cash), but papers at these venues dramatically increase grant success rates and consulting credibility.

**PHANTOM Alignment:** HIGH. The steganographic collusion and covert channel research has already been published at NeurIPS. PHANTOM Protocol papers would be strong submissions to IEEE S&P or USENIX Security.

---

### 4.5 EU AI Act Compliance Opportunities

- **Status:** Enforcement ongoing; major high-risk system compliance deadline August 2, 2026
- **Scope:** Bug bounty programs and VDPs are explicitly recommended as compliance best practices under the EU AI Act's cybersecurity/robustness requirements. EUR 307M+ EU investment in AI technologies announced January 2026.
- **PHANTOM Alignment:** MODERATE. Organizations seeking EU AI Act compliance will need AI security auditing. PHANTOM detection capabilities could be marketed as compliance tools.
- **MWRASP Alignment:** LOW directly, but quantum-secure AI infrastructure will eventually intersect with EU AI Act requirements.
- **Penalties for Non-Compliance:** Up to EUR 35M or 7% of worldwide annual turnover.
- **Sources:** [EU AI Act](https://artificialintelligenceact.eu/), [HackerOne EU AI Act blog](https://www.hackerone.com/blog/eu-ai-act-enforcement-2025-security-compliance)

---

### 4.6 Humanity AI Initiative

- **Status:** LAUNCHED ($500M, five-year initiative)
- **Scope:** Ensuring people shape AI's future for society's benefit
- **PHANTOM/MWRASP Alignment:** LOW-MODERATE. Broad scope; would require creative framing.
- **Sources:** [Inside Philanthropy](https://www.insidephilanthropy.com/home/whos-funding-ai-regulation-and-safety)

---

## 5. ALIGNMENT SCORING

### PHANTOM Protocol Alignment (AI Covert Channels, Steganography, Data Exfiltration)

| Priority | Program | Max Payout/Funding | Alignment |
|----------|---------|-------------------|-----------|
| 1 | Mozilla 0DIN | $15,000/vuln | Covert channels explicitly in scope |
| 2 | Anthropic Model Safety Bounty | $35,000/jailbreak | Cybersecurity domain explicit; steganographic bypass of classifiers |
| 3 | Huntr (Protect AI) | $50,000/vuln | Model backdoors, supply chain, inference-time attacks |
| 4 | OpenAI Cybersecurity Grant | $10M pool (API credits) | Detection of covert channels as defensive cybersecurity |
| 5 | AISI Challenge Fund | GBP 200,000/project | Safeguards, control, steganographic collusion prevention |
| 6 | Gray Swan AI Arena | $40,000/competition | Indirect prompt injection in multi-agent systems |
| 7 | NSF SaTC 2.0 | $1.2M/project | Covert channels in AI systems, steganographic communication |
| 8 | NSF CICI (IPAAI) | $900,000/project | Data integrity/provenance for AI pipelines |
| 9 | AI Safety Fund (Frontier Model Forum) | $500,000 | Cybersecurity is priority area |
| 10 | Apple PCC Bounty | $2M+ potential | Cloud AI data exfiltration |
| 11 | Google AI VRP | $30,000/vuln | Data exfiltration via AI products |
| 12 | OpenAI Bugcrowd | $100,000/vuln | If security impact demonstrated |
| 13 | Microsoft Copilot + Zero Day Quest | $30,000 + $4M event | AI-specific vulnerabilities |
| 14 | Horizon Europe ECCC | EUR 40M (GenAI for Cybersecurity) | EU consortium required |

### MWRASP Alignment (Quantum Cybersecurity, Post-Quantum Cryptography)

| Priority | Program | Max Payout/Funding | Alignment |
|----------|---------|-------------------|-----------|
| 1 | NSF SBIR (QIT) | $1.55M (Phase I+II) | Direct quantum cybersecurity SBIR (paused; watch for reopening) |
| 2 | NSA CNSA 2.0 Mandate | Market creation | Compliance deadline Jan 2027 creates demand |
| 3 | Horizon Europe ECCC (PQC) | EUR 16M (3 PQC topics) | PQC evaluation, implementation, integration (EU consortium req) |
| 4 | H.R.3259 PQC Standards Act | TBD (pending legislation) | Grant pathway for PQC adoption assistance |
| 5 | NSF SaTC 2.0 | $1.2M/project | Quantum-resistant authentication research |
| 6 | Navy/Marine Corps BAA (QIS) | Varies | Defense quantum cybersecurity R&D (open until Sep 2026) |
| 7 | DOE NQI Research Centers | $625M (subawardee) | Quantum communication research (requires lab partner) |
| 8 | DARPA QBI 2026 | Varies | Quantum computing benchmarking (hardware focused) |
| 9 | IARPA Programs | Varies | Quantum computing + AI security |
| 10 | QED-C Membership | Networking | Standards influence, industry partnerships |
| 11 | NIST SBIR | ~$400K/award | Quantum + AI technology prototyping |
| 12 | Digital Europe PQC PKI | EUR 15M | PQC infrastructure (EU consortium required) |
| 13 | NSF NQNI | $2M/year | Quantum research infrastructure |

---

## 6. PRIORITY ACTION MATRIX

### Immediate Actions (February-March 2026)

| Action | Target | Estimated Value | Effort |
|--------|--------|----------------|--------|
| Submit to Mozilla 0DIN | PHANTOM covert channel findings | $500-$15,000/vuln | LOW -- abstract + POC |
| Apply to Anthropic Model Safety Bounty | HackerOne application | $35,000/jailbreak | LOW -- application |
| Apply to AISI Challenge Fund | GBP 50K-200K project proposal | GBP 200,000 | MODERATE -- due March 31, 2026 |
| Submit to OpenAI Cybersecurity Grant | Defensive detection tool for covert channels | $10,000+ API credits | MODERATE -- proposal |
| Register on Huntr platform | AI/ML supply chain vulns | Up to $50,000/vuln | LOW -- registration |

### Near-Term Actions (Q1-Q2 2026)

| Action | Target | Estimated Value | Effort |
|--------|--------|----------------|--------|
| Monitor NSF SBIR QIT reopening | MWRASP quantum auth SBIR proposal | $305K Phase I, $1.25M Phase II | HIGH -- full proposal |
| Submit NSF SaTC 2.0 proposal | PHANTOM + MWRASP research | Up to $1.2M | HIGH -- academic partnership needed |
| Apply to AI Safety Fund | Cybersecurity research | Up to $500,000 | MODERATE |
| Submit to NSF CICI (IPAAI) | AI data integrity research | Up to $900,000 | HIGH |
| Watch for AISI Alignment Project Round 2 | Steganographic collusion prevention | Varies | MODERATE |
| Watch for Open Philanthropy next AI Safety RFP | Technical AI safety research | ~$250K median | MODERATE |

### Ongoing / Opportunistic

| Action | Target | Notes |
|--------|--------|-------|
| Google AI VRP submissions | Data exfiltration via Gemini products | Rolling, up to $30K |
| OpenAI Bugcrowd submissions | Security-impact findings | Rolling, up to $100K |
| Microsoft Zero Day Quest 2026 | Spring 2026 live hacking event | Watch for registration |
| Amazon AI Bug Bounty | Watch for 2026 invitations | Private program expanding |
| Navy/Marine Corps QIS BAA | MWRASP quantum cybersecurity | Open until Sep 2026 |
| Horizon Europe 2026 calls | PQC + GenAI cybersecurity | Requires EU consortium partner |
| H.R.3259 monitoring | PQC Standards Act | If enacted, creates grant pathway |
| IARPA Proposers' Days | AI security + quantum programs | Watch for BAA announcements on SAM.gov |

---

## KEY RESEARCH LANDSCAPE FINDINGS

### Steganographic Covert Channels in LLMs -- Active Research Area

The academic literature as of 2025-2026 confirms that PHANTOM Protocol's research thesis is validated and gaining momentum:

1. **"Hidden in Plain Text" (2024-2025):** Demonstrated that steganographic collusion in LLMs can arise from misspecified reward incentives during training. Standard mitigations (oversight, paraphrasing) are insufficient against adaptive adversaries. Published at NeurIPS.

2. **"Secret Collusion Among AI Agents" (NeurIPS 2024):** Found that as LLM capabilities increase, steganographic abilities outpace equally capable overseers. Frontier models can autonomously develop collusive behaviors.

3. **"StegoAttack" (May 2025):** Achieved 92% attack success rate using steganography to bypass LLM safety mechanisms. Dropped less than 1% even under external detection (Llama Guard).

4. **"TrojanStego":** Adversary fine-tunes LLM to embed sensitive context via linguistic steganography without controlling inference inputs.

This means PHANTOM Protocol research is timely, funded research areas exist, and the defensive detection capabilities PHANTOM could provide are explicitly needed.

### Federal Quantum Funding -- Massive but Constrained

- DOE: $625M for NQI centers (requires national lab lead)
- NSF: $231M QIS budget line in FY2026
- DARPA: QBI 2026 open for new entrants
- NSF SBIR: Paused but reauthorization imminent
- NIST SBIR: Active, $3.19M recent round
- NSA CNSA 2.0: Not funding research but creating mandatory demand by 2027

The quantum cybersecurity market is being created by federal mandate (CNSA 2.0, NSM-10) whether or not individual grant programs are open. MWRASP's 30+ provisionals are positioned in the path of mandatory procurement.

---

*Research compiled February 26, 2026. All program statuses verified via web search where possible. Programs marked as "OPEN" were confirmed active at time of research. Federal funding subject to appropriations lapse conditions.*
