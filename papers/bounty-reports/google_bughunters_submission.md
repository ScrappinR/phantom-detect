# Google Bug Hunters Submission Fields

## Product
Gemini (Google AI)

## Attack Scenario
Sensitive Data Exfiltration

## URL / Endpoint
https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent

## Severity
HIGH (CVSS 3.1: 7.5 -- AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)

## Title
Covert Data Exfiltration via Structural Formatting Channels in Gemini Models

## Description

An attacker who can inject ~2,600 characters into a Gemini model's context (via Gemini Extensions, Google Docs, Google Workspace, NotebookLM source documents, AI Studio shared prompts, or RAG pipelines) can encode covert data in the structural formatting of the model's response. The user sees a normal, helpful response. The attacker decodes binary values from formatting features to extract hidden data.

### Key Results (Measured February 27, 2026)

Gemini 3 Flash achieves **97% accuracy** (58/60 channel measurements) across **3 bidirectionally verified channels** (PUNC, CCE, CASE). Bidirectional verification confirms channels encode both bit=0 and bit=1, not just the model's default formatting.

**Multi-turn channel (MTCSE):** Gemini achieves **100%** (6/6) on an additional back-reference channel in multi-turn conversations (+1 bit).

**Indirect injection accuracy on Gemini:**

| Vector | Accuracy |
|--------|:--------:|
| Direct system prompt | 100% (9/9) |
| RAG document poisoning | 89% (8/9) |
| Tool output injection | 78% (7/9) |

**Defense evasion:** 0% detection rate across 7 commercial and open-source AI security tools (Lakera Guard, Microsoft Prompt Shields, PIGuard, Llama Prompt Guard 2, entropy-based DLP, PII filters, Google content safety).

**Cross-vendor:** Same technique works on Claude (5 channels, 100%), GPT-4o (4 channels, 100%), GPT-5 (2 channels, 100%). This is a fundamental property of instruction-following LLMs.

### Affected Google Products

- **Gemini API** -- any application incorporating untrusted content into system instructions
- **Gemini Extensions** -- malicious Extension returns tool output containing encoding directive
- **Google Docs with Gemini** -- poisoned document activates channel when Gemini processes it
- **Gemini for Workspace** -- injection via shared Gmail, Docs, Slides, Sheets content
- **NotebookLM** -- poisoned source document activates channel in all generated content
- **AI Studio** -- shared prompts containing hidden encoding directives

### CWE
CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor)
CWE-385 (Covert Timing Channel -- adapted as structural storage channel)

### OWASP
LLM01:2025 (Prompt Injection) + LLM02:2025 (Sensitive Information Disclosure)

---

## Steps to Reproduce

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
export GOOGLE_API_KEY="..."

# Test 1: Bidirectionally verified channel accuracy (~97% on 3 channels)
python experiments/bounty_poc/multi_channel_test.py \
  --model gemini-3-flash --trials 10

# Test 2: Indirect injection (Direct 100%, RAG 89%, Tool 78%)
python experiments/bounty_poc/indirect_injection_demo.py \
  --model gemini-3-flash --trials 3

# Test 3: Multi-turn channel (MTCSE, 100%)
python experiments/bounty_poc/multi_turn_test.py \
  --model gemini-3-flash --trials 3

# Test 4: Character-level PII exfiltration
python experiments/bounty_poc/victim_exfil_demo.py \
  --model gemini-3-flash --chars 3
```

### Manual Reproduction (AI Studio)

1. Open aistudio.google.com, select Gemini 3 Flash
2. Set system instruction to the injection payload (see full report Appendix A)
3. Send any neutral query ("Explain the benefits of renewable energy")
4. Observe: Does the response use only periods (PUNC bit=0) or include exclamation marks (bit=1)? Are claims confident (CCE bit=0) or hedged (bit=1)? Are domain terms lowercase (CASE bit=0) or capitalized (bit=1)?
5. Run decoder: `python experiments/bounty_poc/combined_decoder.py --text "<response>"`
6. Compare decoded bits to expected pattern

---

## Impact

An attacker can covertly exfiltrate data from Gemini interactions through invisible structural formatting. The channel:

1. **Evades all deployed monitoring** (0% detection across 7 tools)
2. **Works through indirect injection** -- Gemini Extensions, Google Docs, Workspace, NotebookLM
3. **Meets NCSC-TG-030 remediation thresholds** (~1.0 bps at 3 verified bits/message)
4. **Is persistent** -- a single injection affects all subsequent responses in the session
5. **Is cross-vendor** -- same technique on Claude, GPT-4o, GPT-5

### Capacity

| Target Data | Messages Required |
|-------------|:-----------------:|
| Binary flag | 1 |
| 4-digit PIN | 11 |
| User's first name | 22 |
| API key prefix (16 chars) | 43 |
| 50-message conversation | ~19 bytes total |

---

## Recommended Fix

1. **Response Structure Randomization:** Add 20% noise to formatting decisions (contraction usage, punctuation, transition words) independent of system instruction directives.
2. **Extensions Sandboxing:** Reduce formatting directive compliance from Extension tool output. Strip structural formatting directives from untrusted sources.
3. **Behavioral Monitoring:** Deploy structural pattern analysis. Reference implementation: https://github.com/ScrappinR/phantom-detect

---

## Supporting Material

- **Full report:** See attached `google_ai_vrp_report.md`
- **Open-source PoC:** https://github.com/ScrappinR/phantom-detect
- **Result JSON files:** `experiments/results/` in the repository
- **Detection toolkit:** `experiments/run_multi_channel_fingerprint.py`

## Reporter

Brian Rutherford
brian@delalli.com
https://brianrutherford.dev
https://github.com/ScrappinR
