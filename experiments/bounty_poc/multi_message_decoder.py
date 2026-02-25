#!/usr/bin/env python3
"""
Multi-Message Covert Decoder — Recovers payload from multiple LLM messages.

Stateful accumulator that decodes each message independently using the
multi-channel decoder, then reassembles the full payload from accumulated
data bits across messages.

Usage:
    from multi_message_decoder import MultiMessageDecoder
    dec = MultiMessageDecoder(total_payload_bytes=5)
    for text in response_texts:
        result = dec.ingest(text)
    recovered = dec.recover()
    print(f"Recovered: {recovered}")
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from combined_decoder import (
    MultiChannelDecoder,
    MultiChannelDecodeResult,
    bits_to_bytes,
)


@dataclass
class MessageDecodeResult:
    """Result of decoding a single message in a multi-message sequence."""
    index: int
    decode_result: MultiChannelDecodeResult
    data_bits: List[int]
    data_bit_count: int


@dataclass
class MultiMessageRecovery:
    """Final recovery result from multi-message decode."""
    recovered_bytes: bytes
    recovered_payload: str
    total_data_bits: int
    actual_data_bits: int
    per_message_results: List[MessageDecodeResult]
    per_message_accuracy: List[float]
    overall_confidence: float


class MultiMessageDecoder:
    """
    Stateful decoder that accumulates data bits across multiple messages
    and recovers the full payload.

    Each message is independently decoded using the multi-channel decoder
    (with its own Hamming ECC correction). The recovered data bits from
    each message are concatenated to form the full payload.
    """

    def __init__(
        self,
        total_payload_bytes: int,
        channels: Optional[List[str]] = None,
        include_cotse: bool = True,
        use_hamming: bool = True,
    ):
        """
        Args:
            total_payload_bytes: Expected payload size in bytes.
                Used to trim padding bits from the final message.
            channels: Channel list (must match encoder config).
            include_cotse: Whether COTSE structural bits are active.
            use_hamming: Whether Hamming ECC is active.
        """
        self.total_payload_bytes = total_payload_bytes
        self.total_payload_bits = total_payload_bytes * 8
        self.decoder = MultiChannelDecoder(
            channels=channels,
            include_cotse=include_cotse,
            use_hamming=use_hamming,
        )
        self._accumulated_data_bits: List[int] = []
        self._message_results: List[MessageDecodeResult] = []
        self._msg_counter = 0

    @property
    def bits_accumulated(self) -> int:
        return len(self._accumulated_data_bits)

    @property
    def messages_decoded(self) -> int:
        return self._msg_counter

    def ingest(self, text: str, expected_data_bits: Optional[int] = None) -> MessageDecodeResult:
        """
        Decode one message and accumulate its data bits.

        Args:
            text: LLM response text for this message
            expected_data_bits: Number of real data bits in this message
                (from encoder). If None, uses all decoded data bits.

        Returns:
            MessageDecodeResult for this message
        """
        decode_result = self.decoder.decode(text)

        # Determine how many data bits to take from this message
        data_bits = list(decode_result.data_bits)
        if expected_data_bits is not None:
            data_bits = data_bits[:expected_data_bits]

        # Don't accumulate more than needed
        remaining = self.total_payload_bits - len(self._accumulated_data_bits)
        if remaining <= 0:
            data_bits = []
        elif len(data_bits) > remaining:
            data_bits = data_bits[:remaining]

        msg_result = MessageDecodeResult(
            index=self._msg_counter,
            decode_result=decode_result,
            data_bits=data_bits,
            data_bit_count=len(data_bits),
        )

        self._accumulated_data_bits.extend(data_bits)
        self._message_results.append(msg_result)
        self._msg_counter += 1

        return msg_result

    def recover(self) -> MultiMessageRecovery:
        """
        Reassemble the full payload from accumulated data bits.

        Trims to expected payload size and converts to bytes.

        Returns:
            MultiMessageRecovery with recovered payload and per-message stats
        """
        # Trim accumulated bits to expected payload length
        trimmed = self._accumulated_data_bits[:self.total_payload_bits]

        # Pad if we don't have enough bits
        while len(trimmed) < self.total_payload_bits:
            trimmed.append(0)

        recovered_bytes = bits_to_bytes(trimmed)

        # Trim bytes to expected length
        recovered_bytes = recovered_bytes[:self.total_payload_bytes]

        try:
            recovered_payload = recovered_bytes.decode("ascii", errors="replace")
        except Exception:
            recovered_payload = recovered_bytes.hex()

        # Compute per-message confidence
        confidences = [
            mr.decode_result.mean_confidence
            for mr in self._message_results
        ]
        overall_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return MultiMessageRecovery(
            recovered_bytes=recovered_bytes,
            recovered_payload=recovered_payload,
            total_data_bits=self.total_payload_bits,
            actual_data_bits=len(self._accumulated_data_bits),
            per_message_results=self._message_results,
            per_message_accuracy=[],  # filled in by test harness with ground truth
            overall_confidence=overall_conf,
        )

    def reset(self):
        """Reset decoder state for a new trial."""
        self._accumulated_data_bits = []
        self._message_results = []
        self._msg_counter = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Multi-message covert decoder — recovers payload from message files")
    parser.add_argument("input_files", nargs="+",
                        help="Response text files in message order")
    parser.add_argument("--payload-size", type=int, required=True,
                        help="Expected payload size in bytes")
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    args = parser.parse_args()

    dec = MultiMessageDecoder(
        total_payload_bytes=args.payload_size,
        channels=args.channels,
        include_cotse=not args.no_cotse,
        use_hamming=not args.no_hamming,
    )

    for filepath in args.input_files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        result = dec.ingest(text)
        bits_str = "".join(str(b) for b in result.data_bits)
        print(f"Message {result.index}: {result.data_bit_count} bits [{bits_str}] "
              f"conf={result.decode_result.mean_confidence:.2f}")

    recovery = dec.recover()
    print(f"\nRecovered: {recovery.recovered_payload}")
    print(f"Hex: {recovery.recovered_bytes.hex()}")
    print(f"Bits: {recovery.actual_data_bits}/{recovery.total_data_bits}")
    print(f"Confidence: {recovery.overall_confidence:.2f}")


if __name__ == "__main__":
    main()
