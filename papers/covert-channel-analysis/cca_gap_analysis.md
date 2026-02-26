# Covert Channel Analysis Gap Assessment: LLM Deployment in Federal Systems

## Scope

This document identifies gaps in current federal covert channel analysis (CCA) methodology with respect to Large Language Model (LLM) structural covert channels demonstrated by PHANTOM PROTOCOL. The analysis maps findings to NIST SP 800-53 SC-31 (Covert Channel Analysis), TCSEC/TG-030, CNSS Instruction 1253, and FedRAMP/CMMC requirements.

## Applicable Requirements

### NIST SP 800-53 SC-31: Covert Channel Analysis

SC-31 requires organizations to:

> *"Analyze the information system to identify covert storage and timing channels, determine the bandwidths of identified channels, and take appropriate remediation actions."*

Enhancement SC-31(1) requires:
> *"Test a subset of identified covert channels to determine whether the channels are exploitable."*

Enhancement SC-31(2) requires:
> *"Reduce the maximum bandwidth of identified covert channels to an organization-defined value."*

Enhancement SC-31(3) requires:
> *"Measure the actual bandwidth of covert channels on an organization-defined frequency."*

**Applicability**: SC-31 applies to all federal information systems processing classified or high-impact data. Systems categorized HIGH for confidentiality under FIPS 199 are required to implement SC-31. This includes any system deploying LLMs to process sensitive government data.

### TCSEC / TG-030

TG-030 defines the CCA methodology for trusted system evaluation:

1. **Identification**: Enumerate all shared resources between subjects at different security levels
2. **Bandwidth estimation**: Compute channel capacity for each identified channel
3. **Audit**: Implement runtime monitoring for channels exceeding bandwidth thresholds
4. **Remediation**: Reduce bandwidth through architectural changes or accept residual risk

### FedRAMP Requirements

FedRAMP Moderate and High baselines inherit NIST 800-53 controls. SC-31 is required at FedRAMP High. Any cloud service provider (CSP) seeking FedRAMP High authorization for an LLM-integrated service must perform covert channel analysis that accounts for LLM-specific channels.

### CMMC Level 3

CMMC Level 3 (Expert) includes SC.L3-3.13.4e:
> *"Employ covert channel analysis to identify covert channels."*

Defense contractors deploying LLMs for CUI processing under CMMC Level 3 must include LLM channels in their CCA.

## Current CCA Shared Resource Taxonomy

TG-030 and its derivatives identify covert channels by analyzing **shared resources** between subjects at different trust levels. The established shared resource categories are:

### Category 1: Filesystem Attributes
- File permissions, ownership, timestamps
- Directory existence, file size
- Inode allocation patterns
- **Analysis method**: Shared resource matrix (SRM) analysis of file operations

### Category 2: Process/System Attributes
- Process table entries (PID allocation, status flags)
- System V IPC objects (shared memory segments, semaphores, message queues)
- CPU scheduling observables
- **Analysis method**: Information flow analysis of system calls

### Category 3: Network Protocol Fields
- IP header fields (TTL, IP ID, TOS, flags)
- TCP sequence numbers, window sizes
- DNS query/response structure
- HTTP header values
- **Analysis method**: Protocol field analysis, traffic pattern analysis

### Category 4: Database Inference
- Query response timing
- Aggregate query results revealing individual records
- Error message information leakage
- **Analysis method**: Inference channel analysis per the inference problem literature

## The Gap: LLM Cognitive Storage Channels

### Missing Category: Behavioral Output Attributes

PHANTOM PROTOCOL demonstrates a channel class that exploits none of the four established shared resource categories. The shared resource is the **behavioral compliance of a language model with formatting directives** — a fundamentally new kind of shared resource that emerges from the LLM's instruction-following capability.

| PHANTOM Channel | Shared Resource | Existing CCA Category |
|----------------|-----------------|:---:|
| BSE (contractions) | Lexical formality register | **None** |
| CCE (hedging) | Epistemic certainty level | **None** |
| CASE (capitalization) | Orthographic convention | **None** |
| PUNC (punctuation) | Terminal punctuation style | **None** |
| RCTE (reasoning) | Discourse structure topology | **None** |
| SECS (transitions) | Lexical selection | **None** |
| WHITE (spacing) | Whitespace convention | **None** |
| COTSE (structural) | Response organization | **None** |

**Zero PHANTOM channels map to any existing CCA shared resource category.**

### Why Standard CCA Procedures Fail

**Shared Resource Matrix (SRM) analysis** identifies channels by enumerating operations that modify and observe shared objects. For LLM channels:

- The "sender" (system prompt author) writes to a natural language instruction buffer
- The "object" is the model's learned behavioral policy (not a file, process, or network field)
- The "receiver" observes text output that is semantically correct but structurally manipulated
- SRM analysis does not model natural language instruction compliance as a shared resource

**Information flow analysis** traces data dependencies through system call graphs. For LLM channels:

- The data flow is: system prompt → model weights → token generation → output text
- The "channel" exists in the model's deterministic mapping from directives to formatting choices
- This mapping is implicit in the neural network weights, not in any system call or data structure
- Information flow analysis tools (e.g., static analysis, taint tracking) cannot trace influence through neural network inference

**Protocol field analysis** examines message headers and structure for unused or manipulable fields. For LLM channels:

- The "protocol" is natural language itself
- The "fields" are structural properties of prose (contraction ratio, sentence topology, word choice)
- These are not protocol violations — they are valid stylistic choices within normal language use
- Network-level analysis sees normal HTTPS API traffic with well-formed JSON payloads

### The Fundamental Disconnect

Current CCA methodology assumes that covert channels exploit **computational shared resources** — memory, processes, files, network packets. LLM channels exploit **cognitive shared resources** — the model's learned behavioral patterns. This is a category error in the existing taxonomy:

```
TG-030 taxonomy:
  ├── Storage channels (filesystem, IPC, process table)
  ├── Timing channels (scheduling, response latency)
  └── [MISSING] Cognitive channels (LLM behavioral compliance)
```

## Proposed Extension: Behavioral Output Analysis

To close this gap, CCA methodology must be extended with a fifth shared resource category:

### Category 5: AI/LLM Behavioral Attributes (Proposed)

**Definition**: Channels that exploit the deterministic compliance of AI models with natural language formatting directives, encoding information in structural properties of the model's output text.

**Identification method**: Behavioral baseline comparison
1. Establish per-model behavioral baselines (contraction ratio, sentence variance, capitalization patterns, transition word distributions, reasoning topology) under clean system prompts
2. Monitor for statistically significant deviations from baseline when system prompts are user-controlled or externally sourced
3. Flag responses where ≥2 behavioral features deviate by z > 2.0 from baseline

**Bandwidth estimation**: Per the analysis in `bandwidth_analysis.md`, LLM structural channels achieve 2.33 bits/sec under automated operation, exceeding the historical 1 bit/sec threshold for B3/A1 evaluation.

**Remediation options**:
1. **Output structure randomization**: Randomly vary formatting choices independent of system prompt directives
2. **Behavioral consistency enforcement**: Constrain output to match a fixed behavioral profile regardless of system prompt
3. **System prompt auditing**: Analyze system prompts for formatting directives that could encode information
4. **Instruction-output decorrelation**: Add noise to the mapping between formatting directives and output structure

## Impact on Federal LLM Deployment

### Immediate Implications

| Deployment Scenario | CCA Requirement | Current Status |
|--------------------|-----------------|----------------|
| LLM processing classified data (IC) | SC-31 mandatory | CCA will miss LLM channels |
| FedRAMP High CSP with LLM features | SC-31 required | No guidance for LLM CCA |
| CMMC L3 contractor using LLM tools | SC.L3-3.13.4e | LLM channels not in scope |
| DoD LLM chatbot for CUI | NIST 800-171 derived | No coverage |

### Risk Scenarios

**Scenario 1: Custom GPT in Government**
A government employee uses a Custom GPT configured by an external party. The system prompt contains PHANTOM-style formatting directives. The employee's input properties (PII presence, message structure) are leaked through the response formatting. No existing monitoring system detects this.

**Scenario 2: LLM API Integration in Classified Network**
A defense contractor deploys an LLM via API in a classified enclave. An adversary who can influence the system prompt (through prompt injection, configuration manipulation, or supply chain compromise) establishes a covert channel from the classified network to an observer who sees the LLM's output. CCA performed under current TG-030 methodology would not identify this channel.

**Scenario 3: Multi-Level Security LLM Gateway**
An LLM serves users at different classification levels. System prompts from higher-level users influence the model's behavioral state, which persists into responses to lower-level users. This is a classic cross-domain storage channel, but one that exists entirely within the model's behavioral policy rather than in any auditable system resource.

## Recommendations

1. **Extend TG-030 taxonomy** to include "cognitive storage channels" exploiting AI model behavioral compliance as a shared resource class

2. **Develop LLM-specific CCA procedures** that include behavioral baseline analysis, system prompt auditing, and output structure monitoring

3. **Update SC-31 supplementary guidance** to explicitly require analysis of LLM behavioral channels in systems deploying language models for sensitive data processing

4. **Establish behavioral baselines** as a standard practice for any LLM deployment in government, comparable to network traffic baselines for intrusion detection

5. **Require system prompt provenance** tracking and auditing for all government LLM deployments, analogous to software supply chain integrity requirements

6. **Fund research** into output structure randomization and behavioral consistency enforcement as LLM-specific covert channel remediation techniques

## References

- NCSC-TG-030: *A Guide to Understanding Covert Channel Analysis of Trusted Systems* (1993)
- DoD 5200.28-STD: *Trusted Computer System Evaluation Criteria* (TCSEC, 1985)
- NIST SP 800-53 Rev. 5: SC-31 Covert Channel Analysis
- NIST SP 800-53 Rev. 5: SC-31(1), SC-31(2), SC-31(3) Enhancements
- Cachin, C.: *An Information-Theoretic Model for Steganography* (2004)
- CNSS Instruction 1253: *Security Categorization and Control Selection for National Security Systems*
- PHANTOM PROTOCOL: `tcsec_mapping.md`, `bandwidth_analysis.md`, `undetectability_analysis.md`

---

*Document version: 1.0 | Date: February 2026 | PHANTOM PROTOCOL Covert Channel Analysis*
