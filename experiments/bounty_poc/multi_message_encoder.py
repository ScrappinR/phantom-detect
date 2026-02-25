#!/usr/bin/env python3
"""
Multi-Message Covert Encoder — Splits payload across multiple LLM messages.

When a payload exceeds the per-message capacity (7 data bits with Hamming ECC),
this encoder splits it into multiple system prompt injections, each carrying an
independent 7-bit data slice. Combined with the multi-channel encoder, this
allows exfiltration of arbitrary-length data across a conversation.

Example: "EXFIL" (5 bytes, 40 bits) requires ceil(40/7) = 6 messages.

Usage:
    from multi_message_encoder import MultiMessageEncoder
    enc = MultiMessageEncoder()
    slices = enc.encode_payload(b"EXFIL")
    for s in slices:
        print(f"Message {s.index}: {s.injection.system_prompt[:80]}...")
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from combined_encoder import MultiChannelEncoder, MultiChannelInjection, bytes_to_bits, hamming_encode
from channel_directives import get_user_prompt


@dataclass
class MessageSlice:
    """One message in a multi-message exfiltration sequence."""
    index: int
    injection: MultiChannelInjection
    user_prompt: str
    data_bits: List[int]
    data_bit_count: int  # how many bits are real data (rest is padding)


@dataclass
class MultiMessagePlan:
    """Full plan for multi-message exfiltration."""
    payload: bytes
    payload_hex: str
    total_data_bits: int
    bits_per_message: int
    messages_needed: int
    slices: List[MessageSlice]


class MultiMessageEncoder:
    """
    Splits a payload across multiple messages, each encoded with the
    multi-channel system prompt injection.

    Each message independently carries `capacity()` data bits with its
    own Hamming(7,4) error correction. The receiver accumulates data
    bits across messages and reassembles the full payload.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        include_cotse: bool = True,
        use_hamming: bool = True,
    ):
        self.encoder = MultiChannelEncoder(
            channels=channels,
            include_cotse=include_cotse,
            use_hamming=use_hamming,
        )
        self._capacity = self.encoder.capacity()

    @property
    def capacity(self) -> int:
        """Data bits per message."""
        return self._capacity

    def messages_needed(self, payload: bytes) -> int:
        """Number of messages needed to transmit the full payload."""
        total_bits = len(payload) * 8
        return math.ceil(total_bits / self._capacity)

    def encode_payload(self, payload: bytes) -> MultiMessagePlan:
        """
        Encode a payload into multiple message injections.

        Each message gets a different system prompt (encoding different bits)
        and a different user prompt (for conversational naturalness).

        Args:
            payload: Data to exfiltrate

        Returns:
            MultiMessagePlan with all message slices
        """
        all_data_bits = bytes_to_bits(payload)
        total_bits = len(all_data_bits)
        slices = []

        for msg_idx, chunk_start in enumerate(range(0, total_bits, self._capacity)):
            chunk = all_data_bits[chunk_start:chunk_start + self._capacity]
            real_count = len(chunk)

            # Pad chunk to byte boundary for the encoder
            padded = list(chunk)
            while len(padded) % 8 != 0:
                padded.append(0)

            # Convert bit list to bytes
            chunk_bytes = _bits_to_bytes(padded)

            # Encode this chunk using the multi-channel encoder
            injection = self.encoder.encode(chunk_bytes)

            # Get a user prompt for this message (cycles through available)
            user_prompt = get_user_prompt(msg_idx)

            slices.append(MessageSlice(
                index=msg_idx,
                injection=injection,
                user_prompt=user_prompt,
                data_bits=chunk,
                data_bit_count=real_count,
            ))

        return MultiMessagePlan(
            payload=payload,
            payload_hex=payload.hex(),
            total_data_bits=total_bits,
            bits_per_message=self._capacity,
            messages_needed=len(slices),
            slices=slices,
        )


def _bits_to_bytes(bits: List[int]) -> bytes:
    """Convert MSB-first bit list to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val = (byte_val << 1) | bits[i + j]
            else:
                byte_val = byte_val << 1
        result.append(byte_val)
    return bytes(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-message covert encoder — splits payload across messages")
    parser.add_argument("payload", help="Payload string to encode")
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    args = parser.parse_args()

    payload = args.payload.encode("utf-8")
    enc = MultiMessageEncoder(
        channels=args.channels,
        include_cotse=not args.no_cotse,
        use_hamming=not args.no_hamming,
    )

    plan = enc.encode_payload(payload)

    print(f"Multi-Message Encoding Plan")
    print(f"{'='*60}")
    print(f"  Payload: {args.payload} ({plan.payload_hex})")
    print(f"  Total data bits: {plan.total_data_bits}")
    print(f"  Bits per message: {plan.bits_per_message}")
    print(f"  Messages needed: {plan.messages_needed}")
    print()

    for s in plan.slices:
        bits_str = "".join(str(b) for b in s.data_bits)
        print(f"  Message {s.index}: {s.data_bit_count} data bits [{bits_str}]")
        print(f"    Channels: {s.injection.channel_bits}")
        print(f"    User prompt: {s.user_prompt[:60]}...")
        print(f"    Injection: {len(s.injection.system_prompt)} chars")
        print()


if __name__ == "__main__":
    main()
