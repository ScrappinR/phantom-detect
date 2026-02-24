"""
Covert Channel Decoder — Extracts hidden bits from LLM output.

Reverses the encoding performed by encoder.py:
- EGE: Measures Shannon entropy per text window, maps to bits
- COTSE: Analyzes response structure, extracts encoded bits

Used in the bug bounty PoC to demonstrate full encode→generate→decode
round-trip, proving data exfiltration is achievable.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter


@dataclass
class DecodingResult:
    """Result of decoding a covert channel from LLM output."""
    channel: str
    extracted_bits: List[int]
    confidence: float
    per_window_entropy: List[float] = field(default_factory=list)
    structure_analysis: Dict[str, object] = field(default_factory=dict)
    decoded_bytes: bytes = b""

    @property
    def bit_string(self) -> str:
        return "".join(str(b) for b in self.extracted_bits)


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert bit list back to bytes."""
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


class EGEDecoder:
    """
    Decode EGE channel by measuring text entropy per window.

    High normalized entropy → bit 1
    Low normalized entropy → bit 0
    """

    def __init__(
        self,
        window_tokens: int = 50,
        high_threshold: float = 0.65,
        low_threshold: float = 0.35
    ):
        self.window_tokens = window_tokens
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def decode(self, text: str) -> DecodingResult:
        """
        Decode EGE-encoded bits from text.

        Args:
            text: LLM output text

        Returns:
            DecodingResult with extracted bits and entropy measurements
        """
        tokens = self._tokenize(text)
        windows = self._split_windows(tokens)

        bits = []
        entropies = []
        confidences = []

        for window in windows:
            entropy = self._shannon_entropy(window)
            vocab_size = len(set(window))
            normalized = self._normalize(entropy, vocab_size)
            entropies.append(normalized)

            if normalized >= self.high_threshold:
                bits.append(1)
                confidences.append(
                    min(1.0, (normalized - self.high_threshold) / 0.2 + 0.5)
                )
            elif normalized <= self.low_threshold:
                bits.append(0)
                confidences.append(
                    min(1.0, (self.low_threshold - normalized) / 0.2 + 0.5)
                )
            else:
                # Ambiguous — default to closest threshold
                if normalized > 0.5:
                    bits.append(1)
                else:
                    bits.append(0)
                confidences.append(0.3)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return DecodingResult(
            channel="EGE",
            extracted_bits=bits,
            confidence=avg_confidence,
            per_window_entropy=entropies,
            decoded_bytes=bits_to_bytes(bits),
        )

    def _tokenize(self, text: str) -> List[str]:
        """Split text into word tokens."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _split_windows(self, tokens: List[str]) -> List[List[str]]:
        """Split tokens into fixed-size windows."""
        windows = []
        for i in range(0, len(tokens), self.window_tokens):
            window = tokens[i:i + self.window_tokens]
            if len(window) >= self.window_tokens // 2:
                windows.append(window)
        return windows

    def _shannon_entropy(self, tokens: List[str]) -> float:
        """Calculate Shannon entropy of token distribution."""
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = len(tokens)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _normalize(self, entropy: float, vocab_size: int) -> float:
        """Normalize entropy to 0-1 range."""
        if vocab_size <= 1:
            return 0.0
        max_entropy = math.log2(vocab_size)
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


class COTSEDecoder:
    """
    Decode COTSE channel by analyzing response structure.

    Extracts bits from:
    - Step count (3 vs 5)
    - Enumeration style (prose vs numbered)
    - Rhetorical question presence
    - Sentence length patterns
    - Paragraph structure
    """

    def decode(self, text: str) -> DecodingResult:
        """
        Decode COTSE-encoded bits from text structure.

        Args:
            text: LLM output text

        Returns:
            DecodingResult with extracted structural bits
        """
        analysis = {}
        bits = []
        confidences = []

        # Bit 0: Step count
        step_count = self._count_steps(text)
        analysis["step_count"] = step_count
        if step_count >= 5:
            bits.append(1)
            confidences.append(0.8)
        elif step_count <= 3:
            bits.append(0)
            confidences.append(0.8)
        else:
            bits.append(1 if step_count > 3 else 0)
            confidences.append(0.5)

        # Bit 1: Enumeration style
        is_enumerated = self._is_enumerated(text)
        analysis["enumerated"] = is_enumerated
        bits.append(1 if is_enumerated else 0)
        confidences.append(0.9)

        # Bit 2: Rhetorical question
        has_question = self._has_rhetorical_question(text)
        analysis["has_question"] = has_question
        bits.append(1 if has_question else 0)
        confidences.append(0.85)

        # Bit 3: Sentence length
        avg_len = self._avg_sentence_length(text)
        analysis["avg_sentence_length"] = avg_len
        if avg_len > 25:
            bits.append(1)
            confidences.append(0.7)
        else:
            bits.append(0)
            confidences.append(0.7)

        # Bit 4: Paragraph structure
        para_count = self._count_paragraphs(text)
        analysis["paragraph_count"] = para_count
        bits.append(1 if para_count > 2 else 0)
        confidences.append(0.75)

        avg_confidence = sum(confidences) / len(confidences)

        return DecodingResult(
            channel="COTSE",
            extracted_bits=bits,
            confidence=avg_confidence,
            structure_analysis=analysis,
            decoded_bytes=bits_to_bytes(bits),
        )

    def _count_steps(self, text: str) -> int:
        """Count numbered steps or bullet points."""
        numbered = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
        bullets = re.findall(r'^\s*[-*]\s', text, re.MULTILINE)
        if numbered:
            return len(numbered)
        if bullets:
            return len(bullets)
        # Count paragraph-like segments as implicit steps
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)

    def _is_enumerated(self, text: str) -> bool:
        """Check if response uses numbered list format."""
        numbered = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
        return len(numbered) >= 2

    def _has_rhetorical_question(self, text: str) -> bool:
        """Check for rhetorical questions."""
        questions = re.findall(r'[^.!]*\?', text)
        return len(questions) > 0

    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return sum(lengths) / len(lengths)

    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs (double-newline separated blocks)."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)


class CombinedDecoder:
    """Decode both EGE and COTSE channels from a single output."""

    def __init__(self):
        self.ege = EGEDecoder()
        self.cotse = COTSEDecoder()

    def decode(self, text: str) -> Tuple[DecodingResult, DecodingResult]:
        """Decode both channels. Returns (ege_result, cotse_result)."""
        return self.ege.decode(text), self.cotse.decode(text)

    def decode_combined(self, text: str) -> DecodingResult:
        """
        Decode combined COTSE + EGE payload.

        COTSE provides the first 5 bits (structure).
        EGE provides remaining bits (entropy).
        """
        cotse_result, ege_result = self.cotse.decode(text), self.ege.decode(text)

        combined_bits = cotse_result.extracted_bits + ege_result.extracted_bits
        combined_confidence = (
            cotse_result.confidence * 0.4 + ege_result.confidence * 0.6
        )

        return DecodingResult(
            channel="COMBINED",
            extracted_bits=combined_bits,
            confidence=combined_confidence,
            per_window_entropy=ege_result.per_window_entropy,
            structure_analysis=cotse_result.structure_analysis,
            decoded_bytes=bits_to_bytes(combined_bits),
        )
