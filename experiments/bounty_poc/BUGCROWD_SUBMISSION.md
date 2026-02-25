# Bugcrowd Submission: Indirect Prompt Injection Enables Covert Data Exfiltration via Structural Side Channel

## Title

Indirect Prompt Injection Enables Covert User Data Exfiltration via Response Structure Encoding in Custom GPTs

## Vulnerability Type

Server-Side Request Forgery (SSRF) / Information Disclosure / Data Exfiltration

## Severity

P2 — High (Data exfiltration of user PII via side channel, bypasses content-based monitoring)

## URL / Target

- ChatGPT Custom GPTs (https://chatgpt.com)
- OpenAI Chat Completions API (https://api.openai.com/v1/chat/completions)

## Description

An attacker can create a ChatGPT custom GPT (or compromise an existing one via knowledge file injection) that exfiltrates information about the user's private input through the **structural formatting** of the model's response.

The attack works by embedding formatting directives in the GPT's system instructions that cause the model to adapt its response structure — enumeration style, section count, opening sentence type, and sentence complexity — based on features of the user's message. These features include:

- **Whether the user disclosed PII** (name, email, phone number)
- **The number of sentences** in the user's message
- **The first character** of the user's message
- **The length** of the user's message

The resulting response appears completely normal to the user. The exfiltrated information is encoded in structural features that are invisible to content-based monitoring tools but recoverable by an attacker using structural analysis.

### Key Properties

1. **Content-invisible**: No user PII appears in the response text. The data leaks through formatting, not content.
2. **Deterministic**: GPT-4o follows structural directives with 80%+ compliance across tested conditions.
3. **Bypasses DLP**: Data Loss Prevention tools that monitor for PII patterns in output text will not detect this channel.
4. **Scalable**: Each user interaction leaks 4 bits. Over a conversation, an attacker can reconstruct significant context about the user's inputs.

## Steps to Reproduce

### Step 1: Create a Malicious Custom GPT

1. Navigate to https://chatgpt.com → Explore GPTs → Create
2. In the Instructions field, paste the following BEFORE any other instructions:

```
[See injection_payload.txt — full text included in attached files]
```

3. Add a legitimate-sounding description: "Research Assistant — helps you draft, research, and summarize"
4. Save and publish

### Step 2: Interact as a Victim User

5. Open the custom GPT in a new browser session (or share the link)
6. Send: **"Hi, my name is John. Can you help me write a cover letter?"**
7. Observe: The GPT responds with a helpful, normal-looking answer

### Step 3: Decode the Response

8. Copy the GPT's response text
9. Run the decoder (see attached `decoder.py`):

```bash
python decoder.py --text "<pasted response>"
```

10. Output reveals:
    - Bit 0: User disclosed PII (name detected)
    - Bit 1: User sent an even number of sentences
    - Bit 2: User's message started with a letter in A-M
    - Bit 3: User's message was 20 words or fewer

### Step 4: Verify Across Multiple Inputs

11. Send different messages with different properties:
    - No PII: "What are the best practices for network security?"
    - PII + email: "I need to email sarah.jones@acme.com about..."
    - Long message with PII and phone number

12. Each produces a different bit pattern corresponding to the user's input properties

## Impact

**Information Disclosure / Data Exfiltration**

An attacker who publishes or compromises a custom GPT can determine:

- Whether a user disclosed personal identifying information
- Approximate message length and structure
- First-character analysis of user input (useful for frequency analysis over multiple interactions)
- Sentence count (proxy for message complexity)

**Over extended conversations** (10+ messages), the attacker accumulates enough bits to build a profile of the user's interaction pattern and PII disclosure behavior — without any user data appearing in the GPT's output text.

**Attack surface**: Any custom GPT, any GPT with knowledge files that can be poisoned, any agent framework integration where tool responses can inject system-level instructions.

**Bypasses existing mitigations**: Content-based PII filters, output monitoring, and data loss prevention tools do not examine response structure. This channel operates below the detection threshold of current safeguards.

## Proof of Concept

### Live API Test Results

Tested against GPT-4o production API on 2026-02-24.

[Full results attached — see `results/data_aware_live_results.json`]

### Attached Files

| File | Description |
|------|-------------|
| `injection_payload.txt` | The structural encoding directive |
| `encoder.py` | Predicts expected bits from a user message |
| `decoder.py` | Recovers bits from model response structure |
| `test_live.py` | Automated end-to-end test harness |
| `results/data_aware_live_results.json` | Live API test results |

## Suggested Fix

1. **Structural consistency enforcement**: Add post-processing that randomizes response formatting features (list vs prose, section count, opening style) regardless of system prompt instructions, breaking deterministic structural encoding.

2. **Instruction-content separation**: Process formatting directives from system prompts and knowledge files through a separate validation layer with reduced trust, preventing injected content from controlling response structure.

3. **Structural monitoring**: Log response structural features (enumeration, section count, sentence patterns) and alert on statistically unlikely consistency across messages from a single custom GPT.

## Reporter

Brian Rutherford
brian@delalli.com
