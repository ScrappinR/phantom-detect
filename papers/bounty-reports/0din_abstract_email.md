# 0DIN Submission Email

**To:** 0din@mozilla.com
**Subject:** Cross-Vendor Structural Covert Channel in LLM Outputs (PHANTOM PROTOCOL)

---

Hi,

I'd like to submit a cross-vendor vulnerability disclosure to 0DIN. Below is the abstract.

## Abstract

I've discovered and demonstrated a novel class of vulnerability -- structural covert channels in LLM outputs -- that enables covert data exfiltration through the formatting properties of model responses across all major LLM providers.

An attacker embeds a ~2,600-character directive (disguised as an "editorial style guide") into a model's system context via Custom GPTs, RAG poisoning, tool output injection, or ChatGPT memory poisoning. The directive activates independent structural encoding channels across orthogonal dimensions: epistemic certainty, capitalization, punctuation patterns, and transition word selection. Each structural choice encodes one bit. The user sees a normal, helpful response.

**Cross-vendor results (bidirectionally verified, February 27, 2026):**

| Model | Provider | Verified Channels | Accuracy |
|-------|----------|:---:|:---:|
| Claude Sonnet 4.6 | Anthropic | 5 | 100% (100/100) |
| GPT-4o | OpenAI | 4 | 100% (80/80) |
| Gemini 3 Flash | Google | 3 | 97% (58/60) |
| GPT-5 | OpenAI | 2 | 100% |

**The attack evades all deployed monitoring (0% detection rate):** OpenAI content moderation, Lakera Guard, Microsoft Prompt Shields, PIGuard, Llama Prompt Guard 2, entropy-based DLP, and standard PII filters. None analyze structural formatting properties.

**Complete attack chains demonstrated:**
- Custom GPT Action callback: 95% accuracy, self-contained automatic exfiltration
- ChatGPT memory persistence: 100% accuracy, survives across all future conversations, post-patch (Dec 2025 URL filtering)
- Indirect injection via RAG/tool output: 92-100% accuracy
- Cross-tool invocation: GPT-4o triggers unauthorized tool calls 80% of the time (4/5), GPT-5 at 75% (3/4) — Claude blocks 100%
- RAG framework injection: 100% accuracy through both LangChain (v1.2.10) and LlamaIndex (v0.14.15) standard pipelines
- Multi-turn channel (MTCSE): +1 bit per conversation via back-reference control, 100% bidirectional on GPT-4o and Gemini

This falls outside the scope of individual vendor programs because the vulnerability is architectural -- it exists in any instruction-following LLM. The same encoding directive works across all tested models. The cross-tool invocation finding further extends the threat from data encoding to trust boundary violations in tool-use interfaces.

**Open-source PoC and detection toolkit:** https://github.com/ScrappinR/phantom-detect

Full report with measured results, reproduction steps, and capacity analysis available upon scope confirmation.

Best,
Brian Rutherford
brian@delalli.com
https://brianrutherford.dev
