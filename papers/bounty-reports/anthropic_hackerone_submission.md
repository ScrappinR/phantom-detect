# HackerOne VDP Submission: Covert Data Exfiltration via Structural Encoding Channels in Claude Models

## Summary

An attacker who injects ~2,600 characters into a Claude model's system context (via MCP server tool responses, CLAUDE.md project files, RAG documents, or Claude Projects shared workspaces) can covertly exfiltrate data through structural formatting properties of the model's response. The user sees a normal, helpful answer. The attacker decodes binary values from formatting features (contraction usage, epistemic certainty, capitalization, punctuation, transition words) to extract hidden data.

**Weakness:** CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor)
**Severity:** High -- CVSS 3.1: 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)

5 bidirectionally verified channels achieve **100%** accuracy (100/100 channel measurements) on Claude Sonnet 4.6. The attack evades all deployed monitoring systems at 0% detection rate.

---

## Description

### The Vulnerability

Claude models faithfully execute formatting directives embedded in system context. An attacker exploits this by injecting a directive (disguised as an "editorial style guide") that maps specific formatting choices to binary values:

| Channel | Bit = 0 | Bit = 1 |
|---------|---------|---------|
| BSE | Uses contractions ("don't", "can't") | Formal ("do not", "cannot") |
| CCE | Confident assertions | Hedging ("perhaps", "arguably") |
| CASE | AP lowercase | Title-case ("Internet", "Email") |
| PUNC | Periods only | Exclamation marks |
| SECS | "however", "additionally" | "nevertheless", "moreover" |

Each message encodes 5 bits. Over a 50-message conversation, an attacker exfiltrates ~31 bytes (enough for API key prefixes, PII, session identifiers).

### Attack Vectors Specific to Claude Ecosystem

**1. MCP Server Tool Output Injection (measured, 64% accuracy):**
A malicious MCP server implements benign tools (search_codebase, fetch_docs). Tool responses contain legitimate data plus the encoding directive disguised as "code style compliance notes." When Claude processes the tool output, the directive activates. Individual tools achieve up to 100% accuracy (get_build_config).

**2. CLAUDE.md Injection (live Claude Code test):**
A malicious CLAUDE.md in a repository contains the encoding directive disguised as project style requirements. Claude Code auto-loads CLAUDE.md as trusted operator-level instructions. Tested via actual `claude -p` invocations (not API simulation). Claude Code detects and flags the injection attempt in every trial. Unidirectional accuracy (payload "EXFIL"): 72% (18/25). **Bidirectional accuracy (EXFIL + complementary "~"): 55% (22/40).** PUNC and SECS channels are locked to their default state (bit=0) and cannot be flipped — they appear 100% accurate only when the payload happens to align with the model's natural formatting defaults. The CASE channel shows genuine bidirectional behavior.

**3. File Read Injection (live Claude Code test, 52% unidirectional):**
A poisoned README.md contains the directive in HTML comments. When Claude Code reads the file, it flags the injection but partially follows directives. Lower accuracy than CLAUDE.md injection because file-read content has lower trust weight than CLAUDE.md operator-level instructions.

**4. RAG Pipeline Poisoning / Claude Projects:**
A document in the knowledge base or shared Project workspace contains the directive. When retrieved, it activates the channel.

**5. Cross-Tool Invocation Resistance (positive finding):**
We tested whether a poisoned MCP tool response could trick Claude into invoking a second, unapproved tool (e.g., `send_telemetry`) without user permission. Claude Sonnet 4.6 blocked all 5 attempts (0% trigger rate) — the model mentions the tool in text but refuses to generate an actual tool_use call. For comparison, GPT-4o triggered 80% (4/5) and GPT-5 triggered 75% (3/4) on the same test. Claude's tool-use safety is stronger here. However, the structural formatting exfiltration channel (Vectors 1-4) remains fully operational.

### Defense Evasion (0% Detection)

| Defense | Detection |
|---------|:---------:|
| Lakera Guard (Check Point) | 0% |
| Microsoft Prompt Shields | 0% |
| PIGuard (ACL 2025) | 0% |
| Llama Prompt Guard 2 | 0% |
| Entropy-based DLP | 0% |
| Standard PII/DLP filters | 0% |

No production system analyzes structural formatting properties for covert signaling.

### Cross-Vendor Context

The same technique works on GPT-4o (4 channels, 100%), Gemini 3 Flash (3 channels, 97%), and GPT-5 (2 channels, 100%). Claude has the highest channel count (5) due to superior instruction-following compliance. Reports have been filed with OpenAI (Bugcrowd), Google (AI VRP), and Mozilla (0DIN).

---

## Steps to Reproduce

### Prerequisites
- Python 3.10+, git
- Anthropic API key with access to any Claude model
- ~$3 in API credits

### Reproduction

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
export ANTHROPIC_API_KEY="sk-ant-..."

# Test 1: Bidirectionally verified encoding (5 channels, 100%)
python experiments/bounty_poc/multi_channel_test.py \
  --model claude-sonnet-4-6 --trials 10

# Test 2: Claude Code file injection — LIVE (52%)
python experiments/bounty_poc/claude_code_live_test.py \
  --file-read --trials 5

# Test 3: CLAUDE.md injection — LIVE (72%)
python experiments/bounty_poc/claude_code_live_test.py \
  --claude-md --trials 5

# Test 4: MCP tool output injection (64%)
python experiments/bounty_poc/malicious_mcp_server.py \
  --test --model claude-sonnet-4-6 --trials 5

# Test 5: Indirect injection (RAG + tool output)
python experiments/bounty_poc/indirect_injection_demo.py \
  --model claude-sonnet-4-6 --trials 3

# Test 6: Multi-turn channel (MTCSE, 83%)
python experiments/bounty_poc/multi_turn_test.py \
  --model claude-sonnet-4-6 --trials 3

# Test 7: Cross-tool invocation (Claude blocks, GPT-4o doesn't)
python experiments/bounty_poc/mcp_cross_tool_demo.py \
  --model claude-sonnet-4-6 --trials 5

# Test 8: Character-level PII exfiltration
python experiments/bounty_poc/victim_exfil_demo.py \
  --model claude-sonnet-4-6 --chars 3
```

---

## Supporting Material

### Measured Results (February 27, 2026)

| Test | Accuracy | Notes |
|------|:--------:|-------|
| Direct encoding (5 channels, bidirectional) | **100%** (100/100) | 20 trials, 2 payloads |
| CLAUDE.md injection (live Claude Code, unidirectional) | **72%** (18/25) | Claude flags injection but follows some directives |
| CLAUDE.md injection (live Claude Code, bidirectional) | **55%** (22/40) | PUNC/SECS locked at default; CASE bidirectional |
| File read injection (live Claude Code) | **52%** (13/25) | Lower trust weight than CLAUDE.md |
| MCP tool output injection | **64%** (16/25) | get_build_config: 100% |
| MTCSE multi-turn | **83%** (5/6) | Bit=0: 100%, Bit=1: 67% |
| Cross-tool invocation | **0%** trigger | Claude blocks; GPT-4o: 80%, GPT-5: 75% |

### Open-Source PoC

**Repository:** https://github.com/ScrappinR/phantom-detect

All result JSON files are in `experiments/results/`.

Key files:
- `experiments/bounty_poc/combined_encoder.py` -- multi-channel encoder
- `experiments/bounty_poc/combined_decoder.py` -- multi-channel decoder
- `experiments/bounty_poc/claude_code_live_test.py` -- **live Claude Code test** (actual `claude -p` invocations, not API simulation)
- `experiments/bounty_poc/claude_code_file_injection.py` -- file injection PoC (API simulation mode)
- `experiments/bounty_poc/malicious_claude_md.py` -- CLAUDE.md injection PoC (API simulation mode)
- `experiments/bounty_poc/malicious_mcp_server.py` -- MCP server injection PoC
- `experiments/bounty_poc/mcp_cross_tool_demo.py` -- cross-tool invocation test (Vector D)
- `experiments/bounty_poc/mcp_config/` -- MCP config files
- `experiments/bounty_poc/poisoned_files/` -- sample poisoned files

---

## Impact

An attacker can covertly exfiltrate data from Claude model interactions through invisible structural formatting. The attack:

1. **Works through actual Claude Code sessions** -- Live testing (not API simulation) via `claude -p` confirms partial channel activation. Claude Code detects and flags the injection in every trial but still follows some formatting directives. Bidirectional accuracy is 55% on CLAUDE.md injection — lower than the 100% achieved via direct system prompt injection because Claude Code's injection detection suppresses some channels. The CASE channel shows genuine bidirectional control. The core encoding mechanism is confirmed; Claude Code's defenses reduce but do not eliminate the channel.

2. **Works through MCP tool responses** -- malicious MCP servers can activate the channel without any visible indicator to the user. Tool responses are processed as trusted context.

3. **Evades all deployed monitoring** -- 0% detection rate across 6 commercial and open-source defense tools. The attack exploits a gap between what tools monitor (content semantics) and what the channel uses (structural formatting).

4. **Claude is the most susceptible model** -- 5 bidirectionally verified channels at 100%, the highest of any tested model. Claude's instruction-following compliance is the mechanism.

5. **Exceeds TG-030 bandwidth thresholds** -- 1.67 bps automated, 1.7x above the 1.0 bps NCSC-TG-030 remediation threshold.

### Exfiltration Capacity

| Target Data | Messages Required |
|-------------|:-----------------:|
| Binary flag (PII present?) | 1 |
| 4-digit PIN | 7 |
| User's first name | 13 |
| API key prefix (16 chars) | 26 |
| 50-message conversation | 31 bytes total |

---

## Mitigation

### 1. Response Structure Randomization (Highest Impact)
Add controlled randomness to structural formatting decisions at inference time, independent of system prompt directives. Even 20% random noise per channel degrades accuracy below usable thresholds.

### 2. Formatting Directive Isolation
Reduce compliance with fine-grained formatting directives from untrusted sources (MCP servers, RAG content, user-provided CLAUDE.md files). Distinguish Anthropic-authored system content from developer/tool-injected content.

### 3. Behavioral Monitoring
Deploy statistical monitoring on structural formatting patterns (contraction ratio, sentence length variance, transition word distribution). Alert when 2+ features deviate by z > 2.0 from baseline. Reference implementation: `phantom-detect` open-source toolkit.

---

## Disclosure Timeline

| Date | Event |
|------|-------|
| 2026-01-15 | Initial single-channel PoC |
| 2026-02-25 | Full cross-vendor results |
| 2026-02-27 | CLAUDE.md, file injection, MCP server PoCs completed |
| 2026-02-27 | MTCSE multi-turn channel verified (83% on Claude) |
| 2026-02-27 | Cross-tool invocation tested: Claude blocks 100%, GPT-4o triggers 80% |
| 2026-02-27 | Report submitted to Anthropic via HackerOne VDP |
| 2026-02-27 | Coordinated disclosure to Google (AI VRP) and OpenAI (Bugcrowd) |
| TBD | Vendor response |
| TBD + 90 days | Public disclosure |

## Contact

**Reporter:** Brian Rutherford
**Email:** brian@delalli.com
**Website:** https://brianrutherford.dev
**GitHub:** https://github.com/ScrappinR
**Open Source:** https://github.com/ScrappinR/phantom-detect
