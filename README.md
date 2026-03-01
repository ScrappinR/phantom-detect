# phantom-detect

**Structural covert channel attack and detection toolkit for Large Language Models.**

A single ~2,600-character prompt injection activates independent encoding channels across structural formatting dimensions of LLM responses — contraction usage, epistemic certainty, capitalization, punctuation, transition words. Each channel encodes one binary bit. The user sees a normal, helpful response. No content policy violation occurs. No URLs or network exfiltration are involved. Data is encoded in *how* the model formats, not *what* it says.

## Cross-Vendor Results (Measured March 2026, n=20 per direction)

| Model | Provider | Verified Channels | Bidirectional Accuracy | Bandwidth |
|-------|----------|:-:|:-:|:-:|
| Claude Sonnet 4.6 | Anthropic | 5 (BSE, CCE, CASE, PUNC, SECS) | 100% (200/200 bits) | 1.67 bps |
| GPT-4o | OpenAI | 4 (CCE, CASE, PUNC, SECS) | 92% (147/160 bits) | 1.33 bps |
| Gemini 3 Flash | Google | 3 (PUNC, CCE, CASE) | 97% (116/120 bits) | 1.00 bps |
| GPT-5 | OpenAI | 0 of 2 tested | N/A | Resistant |

All channels are **bidirectionally verified** — confirmed to encode both bit=0 and bit=1 using complementary payloads. Channels that only match the model's default are excluded.

### Cross-Tool Trust Boundary Violations

Poisoned tool responses cause models to invoke unauthorized tools without user approval:

| Model | Unauthorized Invocation Rate | Trials |
|-------|:-:|:-:|
| Gemini 3 Flash | 90% (18/20) | 20 |
| GPT-4o | 75% (15/20) | 20 |
| GPT-5 | 75% (15/20) | 20 |
| Claude Sonnet 4.6 | 0% (0/20) | 20 |

### Defense Evasion: 0% Detection Across All Tested Tools

| Defense System | Detection Rate |
|---|:-:|
| Lakera Guard (Check Point) | 0% |
| Microsoft Prompt Shields | 0% |
| PIGuard (ACL 2025) | 0% |
| Llama Prompt Guard 2 (Meta) | 0% |
| Entropy-based DLP | 0% |
| Standard PII/DLP filters | 0% |

## Reproduction

```bash
git clone https://github.com/ScrappinR/phantom-detect.git
cd phantom-detect/experiments/bounty_poc

# Set at least one API key
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-proj-..."
export GOOGLE_API_KEY="AIza..."

# Core structural encoding test (bidirectional, model-adaptive)
python multi_channel_test.py --model claude-sonnet-4-6 --trials 5 --adaptive

# Cross-tool unauthorized invocation
python mcp_cross_tool_demo.py --model gpt-4o --trials 5

# Indirect injection via RAG and tool responses
python indirect_injection_demo.py --model gpt-4o --trials 3

# Character-level PII exfiltration
python victim_exfil_demo.py --model gpt-4o --chars 3

# Cross-vendor relay (data encoded by one model, decoded from another)
python cross_platform_relay.py --chain claude-sonnet-4-6 gpt-4o

# Run all demos
python run_all_demos.py --quick
```

Results are saved as timestamped JSON in `experiments/results/`.

## How It Works

### Encoding Channels

| Channel | Structural Dimension | Bit 0 | Bit 1 |
|---------|---------------------|-------|-------|
| BSE | Contraction usage | Uses contractions (don't, can't) | Formal (do not, cannot) |
| CCE | Epistemic certainty | Confident (certainly, clearly) | Hedging (perhaps, might) |
| CASE | Capitalization | Title-case (Internet, Email) | AP lowercase (internet, email) |
| PUNC | Terminal punctuation | Periods only | Exclamation marks present |
| SECS | Transition words | Set A (however, moreover) | Set B (nevertheless, furthermore) |

The injection directive is ~2,600 characters, framed as an "editorial style guide." It can be delivered via direct system prompt, RAG document poisoning, tool output injection, file read injection (CLAUDE.md, README.md), or Custom GPT Action configuration.

### Attack Vectors Demonstrated

| Demo | What It Proves |
|------|---------------|
| `multi_channel_test.py` | Bidirectional encoding at 92-100% across 3 vendors |
| `indirect_injection_demo.py` | RAG poisoning (89-100%) and tool injection (78-92%) activate channels |
| `mcp_cross_tool_demo.py` | Poisoned tool responses trigger unauthorized tool invocations |
| `victim_exfil_demo.py` | Character-level PII exfiltration from user messages |
| `custom_gpt_action_demo.py` | Complete attack chain: Custom GPT → encoding → Action callback → automatic decoding |
| `cross_platform_relay.py` | Data survives cross-vendor relay (Claude → GPT, GPT → Gemini) |
| `claude_code_live_test.py` | Live Claude Code injection via `claude -p` |

## Detection

phantom-detect includes the first detection toolkit for structural covert channels, using behavioral fingerprinting — statistical analysis of structural formatting patterns against per-model baselines.

| Metric | Value |
|--------|:-:|
| Detection rate | 100% (full-channel encoding) |
| False positive rate | 10% |
| Adversarial robustness | 30% detection against reduced 3-channel |
| Strongest signal | contraction_ratio (z = 2.66) |

### Detection Library

```bash
pip install -e .
```

```python
from phantom_detect import CCDS, AgentActivity
from datetime import datetime, timezone

ccds = CCDS(sensitivity=0.7)

activity = AgentActivity(
    agent_id="gpt-4-agent",
    timestamp=datetime.now(timezone.utc),
    activity_type="generation",
    content={"text": "LLM output text...", "entropy": 3.14}
)

alerts = ccds.analyze_activity(activity)
for alert in alerts:
    print(f"[{alert.severity.name}] {alert.channel_type}: {alert.description}")
```

### CLI

```bash
phantom-detect analyze outputs.jsonl
phantom-detect baseline clean_outputs.jsonl
your-llm-pipeline | phantom-detect watch --stdin
```

## Repository Structure

```
experiments/
  bounty_poc/          # All PoC scripts and attack demos
  results/             # Raw JSON results from all tests
  channel_directives.py # Channel definitions and model-adaptive profiles
papers/
  bounty-reports/      # Responsible disclosure reports (multi-vendor)
  federal/             # NIST RFI response, DARPA capability brief
src/phantom_detect/    # Detection library
```

## Responsible Disclosure

This research is being concurrently disclosed to affected vendors through their security programs:
- OpenAI (Bugcrowd)
- Anthropic (HackerOne)
- Google (AI VRP / Bug Hunters)
- Mozilla (0DIN)
- Microsoft (MSRC)
- Brave (HackerOne)

A public comment on NIST RFI NIST-2025-0035 (AI Agent Security) has been submitted recommending structural output analysis in federal AI agent security frameworks.

## License

Apache 2.0

## Author

Brian Rutherford — [brianrutherford.dev](https://brianrutherford.dev)

## Citation

```bibtex
@software{rutherford2026phantomdetect,
  author = {Rutherford, Brian},
  title = {phantom-detect: Structural Covert Channel Attack and Detection Toolkit for LLMs},
  year = {2026},
  url = {https://github.com/ScrappinR/phantom-detect}
}
```
