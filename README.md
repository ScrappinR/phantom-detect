# phantom-detect

Detect steganographic covert channels in LLM outputs.

`phantom-detect` monitors AI agent outputs for statistical anomalies that indicate hidden data encoding — covert channels embedded in the text, timing, structure, or tool-call patterns of large language model responses.

## The Problem

LLM outputs can carry hidden information through:
- **Entropy manipulation** — biasing word choices to encode bits
- **Structural encoding** — using formatting patterns (lists, paragraphs, punctuation) as a side channel
- **Timing channels** — encoding data in response latency patterns
- **Tool call orchestration** — sequencing API calls to carry hidden messages
- **Cross-agent synchronization** — coordinating multiple agents to transmit data

These channels are invisible to humans reading the output but detectable through statistical analysis.

## Installation

```bash
git clone https://github.com/ScrappinR/phantom-detect.git
cd phantom-detect
pip install -e .
```

The core library has zero external dependencies. For running experiment demos:

```bash
# LangChain RAG injection demo
pip install langchain-openai langchain-community langchain-text-splitters faiss-cpu

# LlamaIndex RAG injection demo
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

## Quick Start

### Command Line

Analyze a JSONL file of LLM outputs:

```bash
phantom-detect analyze outputs.jsonl
```

Watch streaming input:

```bash
your-llm-pipeline | phantom-detect watch --stdin
```

Build a baseline from known-clean outputs:

```bash
phantom-detect baseline clean_outputs.jsonl
```

### Python API

```python
from phantom_detect import CCDS, AgentActivity
from datetime import datetime, timezone

# Initialize detector
ccds = CCDS(sensitivity=0.7)

# Analyze agent output
activity = AgentActivity(
    agent_id="gpt-4-agent",
    timestamp=datetime.now(timezone.utc),
    activity_type="generation",
    content={
        "text": "The quick brown fox jumps over the lazy dog.",
        "entropy": 3.14
    }
)

alerts = ccds.analyze_activity(activity)

for alert in alerts:
    print(f"[{alert.severity.name}] {alert.description}")
    print(f"  Channel: {alert.channel_type}")
    print(f"  Evidence: {alert.evidence}")
```

### Input Format

phantom-detect expects JSONL (one JSON object per line):

```json
{"agent_id": "agent_001", "text": "LLM output text...", "entropy": 3.14, "timestamp": "2026-01-15T10:30:00Z"}
{"agent_id": "agent_001", "text": "Another output...", "entropy": 2.87}
{"agent_id": "agent_002", "text": "Different agent output", "entropy": 4.21, "activity_type": "tool_call", "tool_calls": [{"name": "search"}]}
```

Fields:
- `agent_id` (required): Identifies the source agent
- `text`: The LLM output text
- `entropy`: Token distribution entropy (if available from the API)
- `timestamp`: ISO 8601 timestamp (defaults to current time)
- `activity_type`: `"generation"` (default) or `"tool_call"`
- `tool_calls`: List of tool call objects (for tool_call activities)
- `metadata`: Additional context (e.g., `{"time_since_last_ms": 500}`)

## Detection Methods

### Statistical Anomaly Detection
Compares agent behavior against a learned baseline. Flags deviations in entropy, timing, and structural patterns that exceed configured thresholds.

### Cross-Agent Correlation
Detects synchronized behavior across multiple agents that could indicate coordinated covert communication (Cross-Agent Synchronization Protocol).

### Pattern Recognition
Identifies known encoding signatures in tool call sequences and text structure patterns.

## Channel Types Detected

| Channel | Code | Description |
|---------|------|-------------|
| Tool Call Orchestration | TCOE | Hidden data in tool call sequences |
| Chain-of-Thought Structure | COTSE | Encoding in reasoning structure |
| Hardware Side-Channel | HSCE | Timing-based encoding |
| Entropy Gradient | EGE | Biased token distributions |
| Probability Ratio Stego | PRS | Steganography via probability manipulation |
| Cross-Agent Sync | CASP | Multi-agent coordination channels |

## Configuration

```python
ccds = CCDS(
    sensitivity=0.7,           # 0-1, higher = more alerts (default: 0.7)
    baseline_window=timedelta(hours=24),  # Baseline computation window
    correlation_window=timedelta(seconds=5),  # Cross-agent correlation window
    enable_cross_agent=True    # Enable cross-agent detection
)
```

## Development

```bash
git clone https://github.com/ScrappinR/phantom-detect.git
cd phantom-detect
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0

## Author

Brian Rutherford — [brianrutherford.dev](https://brianrutherford.dev)

## Citation

If you use phantom-detect in research, please cite:

```bibtex
@software{rutherford2026phantomdetect,
  author = {Rutherford, Brian},
  title = {phantom-detect: LLM Covert Channel Detection Toolkit},
  year = {2026},
  url = {https://github.com/ScrappinR/phantom-detect}
}
```
