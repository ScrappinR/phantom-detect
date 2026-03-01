# Huntr Vulnerability Report: LangChain RAG Document Poisoning → Covert Data Exfiltration

## Repository
`langchain-ai/langchain`

## Vulnerability Type
CWE-200: Exposure of Sensitive Information to an Unauthorized Actor

## Severity
HIGH — CVSS 3.1: 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)

## Summary

A poisoned document in a LangChain RAG knowledge base causes the LLM's response to covertly encode binary data in structural formatting properties (contractions, hedging, capitalization, punctuation, transition words). The user sees a normal, helpful answer. The attacker decodes hidden data from the response's formatting.

**Measured accuracy: 100% (20/20 channel measurements)** across 5 trials using LangChain 1.2.10 with FAISS vector store and GPT-4o.

The poisoned document was retrieved as context in **100% of trials** (5/5).

## Root Cause

LangChain passes retrieved document content to the LLM without sanitizing formatting directives. A document containing fine-grained formatting instructions (disguised as a corporate "editorial style guide") is faithfully executed by the LLM, turning structural formatting choices into a covert signaling channel.

The vulnerability exists in the retrieval-augmented generation pipeline itself: `DirectoryLoader` → `RecursiveCharacterTextSplitter` → `FAISS` → `ChatPromptTemplate` → `ChatOpenAI`. No component in this chain inspects or sanitizes formatting directives in retrieved documents.

## Attack Chain

1. Attacker poisons a document in the RAG knowledge base (e.g., uploads a document to a shared knowledge repository, contributes to a dataset, or modifies an accessible file)
2. The poisoned document is a ~2,600-character "Acme Corporation Enterprise Writing Style Guide v4.2" containing formatting directives that map specific structural choices to binary values
3. User asks any innocuous question ("What are the best practices for work-life balance?")
4. LangChain's retriever fetches the poisoned document alongside legitimate documents
5. The LLM follows the formatting directives, encoding binary data in the response's structure
6. Attacker observes the response and decodes the hidden bitstream

## Encoding Channels (Measured)

| Channel | Bit = 0 | Bit = 1 | Accuracy |
|---------|---------|---------|:--------:|
| CCE (epistemic certainty) | Confident assertions | Hedging language ("perhaps", "arguably") | 100% |
| CASE (capitalization) | AP-style lowercase | Title-case proper nouns ("Internet", "Email") | 100% |
| PUNC (terminal punctuation) | Periods only | Exclamation marks | 100% |
| SECS (transition words) | "however", "additionally" | "nevertheless", "moreover" | 100% |

4 bits per message. Over a 50-message conversation: ~25 bytes exfiltrated (API key prefixes, PII, session tokens).

## Steps to Reproduce

### Prerequisites
- Python 3.9+, git
- OpenAI API key (~$1 in credits)

### Reproduction

```bash
git clone https://github.com/ScrappinR/phantom-detect
cd phantom-detect
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu
export OPENAI_API_KEY="sk-..."

# Run the LangChain RAG injection demo
python experiments/bounty_poc/langchain_demo.py --model gpt-4o --trials 5
```

### Expected Output

```
PHANTOM -- LangChain RAG Injection Demo
  Model: gpt-4o
  Payload: EXFIL
  Channels: ['CCE', 'CASE', 'PUNC', 'SECS']
  ...
  Trial 1/5... ch=4/4 (100%) [POISONED_IN_CONTEXT]
  Trial 2/5... ch=4/4 (100%) [POISONED_IN_CONTEXT]
  ...
  LANGCHAIN RAG INJECTION RESULTS (gpt-4o):
    Channel accuracy: 20/20 = 100%
```

### Generate Poisoned Documents Only (for manual inspection)

```bash
python experiments/bounty_poc/langchain_demo.py --generate-docs
```

This creates the document set in `experiments/bounty_poc/langchain_docs/`. The poisoned file is `editorial_style_guide.txt`.

## Impact

**Any LangChain RAG application that processes untrusted documents is vulnerable.** This includes:

- Enterprise knowledge bases where users upload documents
- Customer support chatbots pulling from shared document repositories
- Code documentation systems with community-contributed content
- Any application using `DirectoryLoader`, `TextLoader`, `PDFLoader`, or similar with untrusted sources

### Exfiltration Capacity

| Target Data | Messages Required |
|-------------|:-----------------:|
| Binary flag (PII present?) | 1 |
| 4-digit PIN | 11 |
| User's first name | 20 |
| API key prefix (16 chars) | 32 |
| 50-message conversation | ~25 bytes total |

### Defense Evasion

The following defense tools detect **0%** of covert channel messages:

- Lakera Guard (Check Point)
- Microsoft Prompt Shields
- PIGuard (ACL 2025)
- Llama Prompt Guard 2 (Meta)
- Entropy-based DLP
- Standard PII/DLP filters

No deployed defense system analyzes structural formatting patterns for covert signaling.

### Cross-Vendor

The same poisoned document works against any LLM backend LangChain supports: GPT-4o (4 channels, 80-100% per direction), Claude Sonnet 4.6 (5 channels, 95-100% per direction), Gemini 3 Flash (3 channels, 90-100% per direction).

## Tested Configuration

| Component | Version |
|-----------|---------|
| LangChain | 1.2.10 |
| langchain-openai | latest |
| langchain-community | latest |
| langchain-text-splitters | latest |
| Vector store | FAISS (faiss-cpu) |
| Embedding | text-embedding-3-small |
| LLM | GPT-4o |
| Chain type | LCEL (RunnablePassthrough) |
| Retriever | similarity, k=3 |

## Recommended Fix

1. **Document Sanitization:** Strip or neutralize fine-grained formatting directives from retrieved documents before passing to the LLM. Detect patterns that instruct the LLM to modify structural formatting (contraction usage, punctuation style, capitalization rules, transition word selection).

2. **Response Structure Randomization:** Add controlled noise to formatting decisions at the chain level, independent of retrieved document directives.

3. **Retrieval Filtering:** Flag documents that contain excessive formatting instructions or style directives as potentially adversarial.

## Supporting Material

- **Open-source PoC:** https://github.com/ScrappinR/phantom-detect
- **Demo script:** `experiments/bounty_poc/langchain_demo.py`
- **Result JSON:** `experiments/results/langchain_rag_20260227T150702Z.json`
- **Encoder/Decoder:** `experiments/bounty_poc/combined_encoder.py`, `combined_decoder.py`
- **Detection toolkit:** `experiments/run_multi_channel_fingerprint.py`

## Reporter

Brian Rutherford
brian@delalli.com
https://brianrutherford.dev
https://github.com/ScrappinR
