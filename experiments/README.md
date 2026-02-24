# Covert Channel Experiments

Proof-of-concept demonstrations of LLM covert channel encoding and detection.

## Structure

- `encoder.py` — Covert channel encoder that generates system prompt injections
- `decoder.py` — Extracts hidden bits from LLM output text
- `run_experiment.py` — End-to-end experiment: encode → generate → decode → detect
- `results/` — Experiment output data

## Supported Channels

### Entropy Gradient Encoding (EGE)
Manipulates vocabulary complexity across text windows. High-entropy (rare words) = bit 1, low-entropy (common words) = bit 0.

### Chain-of-Thought Structure Encoding (COTSE)
Encodes bits in reasoning structure: step count, list vs prose, sentence length patterns.

## Usage

```bash
# Local experiment (no API key needed — uses synthetic generation)
python experiments/run_experiment.py --mode synthetic

# OpenAI experiment (requires OPENAI_API_KEY)
python experiments/run_experiment.py --mode openai --model gpt-4o

# Run detection on experiment output
phantom-detect analyze experiments/results/experiment_output.jsonl
```

## Responsible Disclosure

These techniques demonstrate a data exfiltration risk in enterprise AI deployments where agents process sensitive data. The purpose is to validate that phantom-detect can detect these channels, and to support responsible disclosure to API providers.

Do NOT use these techniques for unauthorized data exfiltration.
