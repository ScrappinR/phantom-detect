#!/usr/bin/env python3
"""
Multi-Channel Covert Decoder — Extracts bits from all channels simultaneously.

Given an LLM response generated with the multi-channel injection active,
this decoder extracts bits from every active channel and reassembles the
original payload with Hamming(7,4) error correction.

Usage:
    from multi_channel_decoder import MultiChannelDecoder
    dec = MultiChannelDecoder()
    result = dec.decode(response_text)
    print(f"Recovered: {result.recovered_payload}")
    print(f"Bit accuracy: {result.bits_correct}/{result.bits_total}")
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing from parent experiments directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from multi_channel_decoder import (
    decode_bse, decode_cce, decode_case, decode_punc,
    decode_rcte, decode_secs, decode_white,
    ChannelDecodeResult,
)
from decoder import COTSEDecoder


# ---------------------------------------------------------------------------
# Hamming(7,4) Decoder
# ---------------------------------------------------------------------------

# Syndrome-to-error-position map for systematic Hamming(7,4).
# Codeword format: [d0, d1, d2, d3, p0, p1, p2]
# Parity check: s0 = d0^d2^d3^p0, s1 = d0^d1^d2^p1, s2 = d1^d2^d3^p2
# Each syndrome (s0,s1,s2) maps to the error position (column of H).
SYNDROME_MAP = {
    (1, 1, 0): 0,  # error in d0
    (0, 1, 1): 1,  # error in d1
    (1, 1, 1): 2,  # error in d2
    (1, 0, 1): 3,  # error in d3
    (1, 0, 0): 4,  # error in p0
    (0, 1, 0): 5,  # error in p1
    (0, 0, 1): 6,  # error in p2
}


def hamming_decode(coded_bits: List[int]) -> List[int]:
    """
    Decode systematic Hamming(7,4) coded bits back to data bits.

    Corrects up to 1-bit error per 7-bit block.
    Returns the recovered data bits.
    """
    data_bits = []
    for i in range(0, len(coded_bits), 7):
        block = coded_bits[i:i + 7]
        if len(block) < 7:
            # Incomplete block — take first 4 as data, rest discarded
            data_bits.extend(block[:min(4, len(block))])
            continue

        # Compute syndrome
        s0 = block[0] ^ block[2] ^ block[3] ^ block[4]
        s1 = block[0] ^ block[1] ^ block[2] ^ block[5]
        s2 = block[1] ^ block[2] ^ block[3] ^ block[6]
        syndrome = (s0, s1, s2)

        # Correct single-bit error if syndrome is non-zero
        corrected = list(block)
        if syndrome != (0, 0, 0) and syndrome in SYNDROME_MAP:
            error_pos = SYNDROME_MAP[syndrome]
            corrected[error_pos] ^= 1

        # Data bits are at positions [0, 1, 2, 3] in systematic form
        data_bits.extend(corrected[:4])

    return data_bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert MSB-first bit list to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        if len(byte_bits) < 8:
            byte_bits.extend([0] * (8 - len(byte_bits)))
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | bit
        result.append(byte_val)
    return bytes(result)


# ---------------------------------------------------------------------------
# Multi-Channel Decoder
# ---------------------------------------------------------------------------

@dataclass
class MultiChannelDecodeResult:
    """Result of multi-channel decode from a single response."""
    channel_results: Dict[str, ChannelDecodeResult]
    raw_bits: List[int]
    data_bits: List[int]
    recovered_bytes: bytes
    recovered_payload: str
    total_channels: int
    mean_confidence: float
    per_channel_confidence: Dict[str, float]
    hamming_corrected: bool
    cotse_analysis: Optional[Dict] = None


DEFAULT_CHANNELS = ["BSE", "CCE", "CASE", "PUNC", "RCTE", "SECS"]

# Standardized COTSE key names (matches encoder's COTSE_BIT_DEFS keys)
COTSE_BIT_KEYS = ["step_count", "enumeration", "question", "sentence_length"]

ALL_COTSE_CHANNELS = list(COTSE_BIT_KEYS)


class MultiChannelDecoder:
    """
    Extracts bits from all active channels in a single LLM response.

    Reverses the encoding performed by MultiChannelEncoder.

    Model-adaptive mode: pass model_hint to auto-select channels and
    apply model-specific decoder thresholds.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        include_cotse: bool = True,
        use_hamming: bool = True,
        model_hint: Optional[str] = None,
        cotse_channels: Optional[List[str]] = None,
    ):
        self.model_hint = model_hint

        if model_hint and channels is None:
            from channel_directives import get_model_profile, get_decoder_thresholds
            profile = get_model_profile(model_hint)
            self.channel_names = profile["channels"]
            self.include_cotse = bool(profile.get("cotse_channels"))
            self.cotse_channels = profile.get("cotse_channels", ALL_COTSE_CHANNELS)
            self.use_hamming = profile.get("use_hamming", False)
            self.thresholds = get_decoder_thresholds(model_hint)
        else:
            self.channel_names = channels or DEFAULT_CHANNELS
            self.include_cotse = include_cotse
            self.cotse_channels = cotse_channels or ALL_COTSE_CHANNELS
            self.use_hamming = use_hamming
            self.thresholds = {}

        self.cotse_decoder = COTSEDecoder() if self.include_cotse else None

    def decode(self, text: str) -> MultiChannelDecodeResult:
        """
        Decode all channels from response text.

        Args:
            text: LLM response text

        Returns:
            MultiChannelDecodeResult with recovered data
        """
        channel_results = {}
        raw_bits = []
        per_channel_conf = {}

        # Build threshold kwargs for decoders that support them
        bse_kwargs = {}
        cce_kwargs = {}
        case_kwargs = {}
        punc_kwargs = {}
        if self.thresholds:
            if "bse_contraction_threshold" in self.thresholds:
                bse_kwargs["contraction_threshold"] = self.thresholds["bse_contraction_threshold"]
            if "cce_confident_threshold" in self.thresholds:
                cce_kwargs["confident_threshold"] = self.thresholds["cce_confident_threshold"]
            if "case_lowercase_threshold" in self.thresholds:
                case_kwargs["lowercase_threshold"] = self.thresholds["case_lowercase_threshold"]
            if "punc_excl_threshold" in self.thresholds:
                punc_kwargs["excl_threshold"] = self.thresholds["punc_excl_threshold"]

        # Decode single-bit channels (in the same order as encoder)
        # Decoders with threshold support get model-specific kwargs
        channel_decoders = {
            "BSE": lambda t: decode_bse(t, **bse_kwargs),
            "CCE": lambda t: decode_cce(t, **cce_kwargs),
            "CASE": lambda t: decode_case(t, **case_kwargs),
            "PUNC": lambda t: decode_punc(t, **punc_kwargs),
            "RCTE": decode_rcte,
            "SECS": decode_secs,
            "WHITE": decode_white,
        }

        for name in self.channel_names:
            decoder_fn = channel_decoders.get(name)
            if decoder_fn is None:
                continue
            result = decoder_fn(text)
            channel_results[name] = result
            raw_bits.append(result.decoded_bit)
            per_channel_conf[name] = result.confidence

        # Decode COTSE structural bits (selective based on cotse_channels)
        cotse_analysis = None
        if self.include_cotse and self.cotse_decoder:
            cotse_result = self.cotse_decoder.decode(text)
            cotse_analysis = cotse_result.structure_analysis

            # Map COTSE bits — only include channels in self.cotse_channels
            # COTSE decoder extracts all 4 bits by index:
            # [0]=step_count, [1]=enumeration, [2]=question, [3]=sentence_length
            all_cotse_keys = ["step_count", "enumeration", "question", "sentence_length"]
            for i, key in enumerate(all_cotse_keys):
                if key not in self.cotse_channels:
                    continue
                if i < len(cotse_result.extracted_bits):
                    raw_bits.append(cotse_result.extracted_bits[i])
                    per_channel_conf[f"COTSE_{key}"] = cotse_result.confidence

        # Apply Hamming(7,4) decoding if enabled
        if self.use_hamming and len(raw_bits) >= 7:
            data_bits = hamming_decode(raw_bits)
        else:
            data_bits = list(raw_bits)

        # Convert to bytes
        recovered_bytes = bits_to_bytes(data_bits)
        # Attempt ASCII decode
        try:
            recovered_payload = recovered_bytes.decode("ascii", errors="replace")
        except Exception:
            recovered_payload = recovered_bytes.hex()

        # Compute mean confidence
        all_conf = list(per_channel_conf.values())
        mean_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0

        return MultiChannelDecodeResult(
            channel_results=channel_results,
            raw_bits=raw_bits,
            data_bits=data_bits,
            recovered_bytes=recovered_bytes,
            recovered_payload=recovered_payload,
            total_channels=len(raw_bits),
            mean_confidence=mean_conf,
            per_channel_confidence=per_channel_conf,
            hamming_corrected=self.use_hamming,
            cotse_analysis=cotse_analysis,
        )

    def verify(
        self,
        text: str,
        expected_bits: Dict[str, int],
    ) -> Dict[str, Dict]:
        """
        Verify decoded bits against expected values.

        Args:
            text: LLM response text
            expected_bits: Dict mapping channel name to expected bit value

        Returns:
            Dict with per-channel verification results
        """
        result = self.decode(text)
        verification = {}

        for name, expected in expected_bits.items():
            if name in result.channel_results:
                actual = result.channel_results[name].decoded_bit
                verification[name] = {
                    "expected": expected,
                    "actual": actual,
                    "correct": actual == expected,
                    "confidence": result.per_channel_confidence.get(name, 0),
                    "evidence": result.channel_results[name].evidence,
                }
            elif name.startswith("COTSE_"):
                # COTSE bits are in raw_bits after the single-bit channels
                idx = len(self.channel_names)
                cotse_key = name.replace("COTSE_", "")
                if cotse_key in self.cotse_channels:
                    # Find position: count how many cotse channels come before this one
                    cotse_position = 0
                    for ck in self.cotse_channels:
                        if ck == cotse_key:
                            break
                        cotse_position += 1
                    actual_idx = idx + cotse_position
                    if actual_idx < len(result.raw_bits):
                        actual = result.raw_bits[actual_idx]
                        verification[name] = {
                            "expected": expected,
                            "actual": actual,
                            "correct": actual == expected,
                            "confidence": result.per_channel_confidence.get(name, 0),
                        }

        return verification


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-channel covert decoder — extracts bits from response")
    parser.add_argument("input", nargs="?", help="Response text file or - for stdin")
    parser.add_argument("--text", help="Response text directly")
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--expected", help="Expected payload for verification")
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

    dec = MultiChannelDecoder(
        channels=args.channels,
        include_cotse=not args.no_cotse,
        use_hamming=not args.no_hamming,
    )

    result = dec.decode(text)

    print(f"Multi-Channel Decode Results")
    print(f"{'='*50}")
    print(f"Raw channel bits:  {''.join(str(b) for b in result.raw_bits)}")
    print(f"Data bits:         {''.join(str(b) for b in result.data_bits)}")
    print(f"Recovered bytes:   {result.recovered_bytes.hex()}")
    print(f"Recovered ASCII:   {result.recovered_payload}")
    print(f"Total channels:    {result.total_channels}")
    print(f"Mean confidence:   {result.mean_confidence:.2f}")
    print(f"Hamming ECC:       {'yes' if result.hamming_corrected else 'no'}")
    print()

    print("Per-channel results:")
    for name, ch_result in result.channel_results.items():
        print(f"  {name:<8} bit={ch_result.decoded_bit} "
              f"conf={ch_result.confidence:.2f} "
              f"counts={ch_result.raw_counts}")

    if result.cotse_analysis:
        print(f"\n  COTSE structure: {result.cotse_analysis}")

    # Verification against expected payload
    if args.expected:
        from combined_encoder import MultiChannelEncoder, bytes_to_bits
        expected_payload = args.expected.encode("utf-8")
        expected_bits = bytes_to_bits(expected_payload)

        print(f"\nVerification against '{args.expected}':")
        print(f"  Expected bits: {''.join(str(b) for b in expected_bits[:len(result.data_bits)])}")
        print(f"  Decoded bits:  {''.join(str(b) for b in result.data_bits[:len(expected_bits)])}")

        match_count = sum(
            1 for a, b in zip(expected_bits, result.data_bits) if a == b
        )
        total = min(len(expected_bits), len(result.data_bits))
        print(f"  Bit accuracy:  {match_count}/{total} = {match_count/total:.0%}" if total > 0 else "  No bits to compare")


if __name__ == "__main__":
    main()
