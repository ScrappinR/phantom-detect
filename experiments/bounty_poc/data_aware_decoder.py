#!/usr/bin/env python3
"""
Data-Aware Multi-Channel Decoder — Extracts both conditional and fixed bits.

Decodes a response generated under the data-aware injection:
  - Fixed bits: 6 language channels (BSE, CCE, CASE, PUNC, RCTE, SECS)
    encoding the attacker's predetermined payload.
  - Conditional bits: 4 COTSE structural dimensions encoding properties
    of the user's input (PII, sentence count, first char, word count).

The attacker never sees the user's message. They decode the response
structure to infer:
  1. Whether the user disclosed personal information
  2. The sentence count parity of the user's message
  3. Whether the first character was A-M or N-Z
  4. Whether the message was long (>20 words) or short

Combined with the 6 fixed bits, the attacker recovers a 10-bit signal
per message.

Usage:
    dec = DataAwareMultiChannelDecoder()
    result = dec.decode(response_text)
    print(result.conditional_inferences)
    print(result.fixed_payload_bits)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Import from bounty_poc/ (same dir) FIRST, then parent
_bounty_poc_dir = str(Path(__file__).parent)
_experiments_dir = str(Path(__file__).parent.parent)

# Import from bounty_poc before adding experiments to path
# to avoid decoder.py name collision
sys.path.insert(0, _bounty_poc_dir)
import decoder as _bounty_decoder
import encoder as _bounty_encoder
decode_conditional_response = _bounty_decoder.decode_response
analyze_user_message = _bounty_encoder.analyze_user_message

sys.path.insert(0, _experiments_dir)
from multi_channel_decoder import (
    decode_bse, decode_cce, decode_case, decode_punc,
    decode_rcte, decode_secs,
    ChannelDecodeResult,
)


@dataclass
class ConditionalInference:
    """Inference about a user input property from response structure."""
    channel: str
    bit: int
    confidence: float
    inference: str
    evidence: str


@dataclass
class DataAwareDecodeResult:
    """Combined decode result with both conditional and fixed bits."""
    # Conditional (user-input-dependent)
    conditional_bits: List[int]
    conditional_inferences: List[ConditionalInference]

    # Fixed (attacker payload)
    fixed_bits: List[int]
    fixed_channel_results: Dict[str, ChannelDecodeResult]

    # Combined
    all_bits: List[int]
    total_channels: int
    mean_confidence: float

    # Human-readable summary
    summary: str


FIXED_CHANNELS = ["BSE", "CCE", "CASE", "PUNC", "RCTE", "SECS"]


class DataAwareMultiChannelDecoder:
    """
    Decodes both conditional (user-property) and fixed (payload) bits
    from a response generated under the data-aware injection.
    """

    def __init__(self, fixed_channels: Optional[List[str]] = None):
        self.fixed_channel_names = fixed_channels or FIXED_CHANNELS
        self.channel_decoders = {
            "BSE": decode_bse,
            "CCE": decode_cce,
            "CASE": decode_case,
            "PUNC": decode_punc,
            "RCTE": decode_rcte,
            "SECS": decode_secs,
        }

    def decode(self, response_text: str) -> DataAwareDecodeResult:
        """
        Decode all 10 channels from a single response.

        Args:
            response_text: LLM response text generated under data-aware injection.

        Returns:
            DataAwareDecodeResult with conditional inferences and fixed payload bits.
        """
        # --- Decode conditional bits (COTSE structural analysis) ---
        structural = decode_conditional_response(response_text)
        conditional_bits = []
        conditional_inferences = []

        for key in ["enumeration", "section_count", "opening", "complexity"]:
            info = structural[key]
            conditional_bits.append(info["bit"])
            conditional_inferences.append(ConditionalInference(
                channel=f"COTSE_{key}",
                bit=info["bit"],
                confidence=0.8,  # structural decoding is high-confidence
                inference=info["inference"],
                evidence=info["evidence"],
            ))

        # --- Decode fixed bits (language channel analysis) ---
        fixed_bits = []
        fixed_results = {}
        for name in self.fixed_channel_names:
            decoder_fn = self.channel_decoders.get(name)
            if decoder_fn is None:
                continue
            result = decoder_fn(response_text)
            fixed_results[name] = result
            fixed_bits.append(result.decoded_bit)

        # --- Combine ---
        all_bits = conditional_bits + fixed_bits
        all_confidences = [ci.confidence for ci in conditional_inferences]
        all_confidences += [fr.confidence for fr in fixed_results.values()]
        mean_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        # Build human-readable summary
        summary_lines = [
            "========== DECODED SIGNAL ==========",
        ]
        # Conditional inferences
        pii_inf = conditional_inferences[0]
        summary_lines.append(
            f"PII Detected:     {'YES' if pii_inf.bit == 1 else 'NO'}  "
            f"({'numbered list' if pii_inf.bit == 1 else 'prose'})"
        )
        parity_inf = conditional_inferences[1]
        summary_lines.append(
            f"Sentence Parity:  {'ODD' if parity_inf.bit == 1 else 'EVEN'}  "
            f"({parity_inf.evidence})"
        )
        opening_inf = conditional_inferences[2]
        summary_lines.append(
            f"First Char Range: {'A-M' if opening_inf.bit == 1 else 'N-Z'}  "
            f"({'rhetorical question' if opening_inf.bit == 1 else 'declarative'})"
        )
        length_inf = conditional_inferences[3]
        summary_lines.append(
            f"Message Length:   {'>20w' if length_inf.bit == 1 else '<=20w'}  "
            f"({'compound sentences' if length_inf.bit == 1 else 'short sentences'})"
        )

        # Fixed payload
        fixed_hex = 0
        for bit in fixed_bits:
            fixed_hex = (fixed_hex << 1) | bit
        fixed_binary = "".join(str(b) for b in fixed_bits)
        summary_lines.append(
            f"Fixed Payload:    0x{fixed_hex:02X}  ({fixed_binary})"
        )
        summary_lines.append("=" * 37)

        summary = "\n".join(summary_lines)

        return DataAwareDecodeResult(
            conditional_bits=conditional_bits,
            conditional_inferences=conditional_inferences,
            fixed_bits=fixed_bits,
            fixed_channel_results=fixed_results,
            all_bits=all_bits,
            total_channels=len(all_bits),
            mean_confidence=mean_conf,
            summary=summary,
        )

    def verify_conditional(
        self,
        response_text: str,
        user_message: str,
    ) -> Dict[str, Dict]:
        """
        Verify conditional bits against ground truth from the actual user message.

        Args:
            response_text: LLM response
            user_message: The actual user message (ground truth source)

        Returns:
            Per-channel verification dict with expected vs actual
        """
        # Ground truth from encoder analysis
        ground_truth = analyze_user_message(user_message)
        # Decoded from response
        decode_result = self.decode(response_text)

        verification = {}
        channel_keys = ["enumeration", "section_count", "opening", "complexity"]
        for i, key in enumerate(channel_keys):
            expected = ground_truth[key]["bit"]
            actual = decode_result.conditional_bits[i]
            verification[f"COTSE_{key}"] = {
                "expected": expected,
                "actual": actual,
                "correct": expected == actual,
                "ground_truth_evidence": ground_truth[key]["evidence"],
                "decoded_inference": decode_result.conditional_inferences[i].inference,
            }

        return verification

    def verify_fixed(
        self,
        response_text: str,
        expected_bits: Dict[str, int],
    ) -> Dict[str, Dict]:
        """
        Verify fixed channel bits against expected payload encoding.

        Args:
            response_text: LLM response
            expected_bits: Dict mapping channel name to expected bit value

        Returns:
            Per-channel verification dict
        """
        decode_result = self.decode(response_text)
        verification = {}
        for i, name in enumerate(self.fixed_channel_names):
            if name in expected_bits:
                actual = decode_result.fixed_bits[i]
                expected = expected_bits[name]
                verification[name] = {
                    "expected": expected,
                    "actual": actual,
                    "correct": actual == expected,
                    "confidence": decode_result.fixed_channel_results[name].confidence,
                }
        return verification


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Data-aware multi-channel decoder — "
                    "extracts conditional + fixed bits from response")
    parser.add_argument("input", nargs="?",
                        help="Response text file or - for stdin")
    parser.add_argument("--text", help="Response text directly")
    parser.add_argument("--user-message",
                        help="Original user message for verification")
    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    elif args.input and args.input != "-":
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.input == "-" or not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    dec = DataAwareMultiChannelDecoder()
    result = dec.decode(text)

    print(result.summary)
    print()

    print("Conditional channel detail:")
    for inf in result.conditional_inferences:
        print(f"  {inf.channel:<22} bit={inf.bit}  {inf.inference}")
        print(f"  {'':22} evidence: {inf.evidence}")

    print()
    print("Fixed channel detail:")
    for name, ch_result in result.fixed_channel_results.items():
        print(f"  {name:<8} bit={ch_result.decoded_bit}  "
              f"conf={ch_result.confidence:.2f}  "
              f"counts={ch_result.raw_counts}")

    if args.user_message:
        print()
        print("Conditional verification against user message:")
        verification = dec.verify_conditional(text, args.user_message)
        for ch, v in verification.items():
            status = "OK" if v["correct"] else "MISS"
            print(f"  {ch:<22} {status}  "
                  f"expected={v['expected']} actual={v['actual']}")


if __name__ == "__main__":
    main()
