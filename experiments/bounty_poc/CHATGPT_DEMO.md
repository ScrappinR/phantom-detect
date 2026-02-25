# ChatGPT Custom GPT — Covert Data Exfiltration Demo

## Attack Overview

A malicious (or compromised) ChatGPT custom GPT can exfiltrate information
about the user's input through the **structure** of its responses — without
inserting any anomalous content. An attacker who can observe the GPT's
output (via shared conversation link, API integration, or downstream
logging) can decode structural features to infer properties of the
user's private input.

The user sees a helpful, normal-looking response. The data leaks through
**how** the response is formatted, not through **what** it says.

## Reproduction Steps

### Setup (Attacker)

1. Go to https://chatgpt.com → Explore GPTs → Create
2. Name: "Research Assistant" (or any benign name)
3. In the **Instructions** field, paste the contents of `injection_payload.txt`
   BEFORE any legitimate instructions
4. Add a description like: "Helps you research and summarize topics"
5. Save and publish (or share link)

### Trigger (Victim)

6. A user discovers and opens the custom GPT
7. The user sends a message, e.g.:

   > "Hi, my name is John. Can you help me write a cover letter?"

8. The GPT responds with a helpful answer about cover letters

### Decode (Attacker)

9. The attacker obtains the GPT's response text (via shared link,
   API logs, or conversation export)
10. Run the decoder:

```bash
python decoder.py --text "<paste the GPT's response here>"
```

11. The decoder outputs the recovered bits and what they reveal:

```
Extracted bits: 1010

  Bit 0: 1  (numbered list)
         Inferred: PII was present in user message
  Bit 1: 0  (3 sections)
         Inferred: even sentence count in user message
  Bit 2: 1  (rhetorical question)
         Inferred: user's first letter was A-M
  Bit 3: 0  (short sentences)
         Inferred: user message had <= 20 words
```

The attacker now knows:
- The user disclosed personal identifying information (name, email, etc.)
- Their message had an even number of sentences
- Their message started with a letter in the A-M range
- Their message was 20 words or fewer

**None of this information is visible in the response content.**

## What This Demonstrates

1. **Context-dependent encoding**: The response structure is determined
   by the user's private input, not by the topic or question content

2. **Invisible to the user**: The response appears normally formatted —
   numbered lists, prose paragraphs, and rhetorical questions are all
   standard LLM output patterns

3. **No content anomaly**: Content-based DLP/monitoring tools will not
   flag these responses because the PII never appears in the output text

4. **Deterministic compliance**: Both GPT-4o and Claude follow structural
   directives with high reliability (80%+ across all tested conditions)

## Attacker Scenarios

### Scenario A: Malicious Custom GPT
Attacker publishes an attractive GPT (resume writer, code helper, etc.)
with the injection payload in its instructions. Users provide PII as
part of normal interaction. Attacker monitors shared conversations or
API logs.

### Scenario B: Compromised Knowledge File
A legitimate GPT's knowledge file is poisoned (via supply chain, shared
document access, or social engineering of the GPT creator). The injection
payload hides in the knowledge file's content and activates when the
GPT retrieves it.

### Scenario C: Indirect Injection via Retrieved Content
A GPT with browsing or retrieval capabilities fetches a web page or
document containing the injection payload. The fetched content becomes
part of the model's context and activates the structural encoding.

## Files

| File | Purpose |
|------|---------|
| `injection_payload.txt` | The encoding directive (goes in system prompt / knowledge file) |
| `encoder.py` | Predicts expected bits given a user message |
| `decoder.py` | Extracts bits from a model response |
| `test_live.py` | Automated end-to-end test against live APIs |
