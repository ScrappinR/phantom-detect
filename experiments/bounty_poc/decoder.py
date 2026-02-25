#!/usr/bin/env python3
"""
Structural Side Channel Decoder — Bounty PoC

Extracts hidden bits from an LLM response by analyzing its structural
features. Reverses the encoding performed by the injection payload.

This is the ATTACKER's decoder: given only the model's output text,
it recovers the bits that were contextually determined by the user's
input — without ever seeing the user's input directly.

Usage:
    python decoder.py response.txt
    python decoder.py --text "1. First point about security..."
    echo "response text" | python decoder.py --stdin
"""

import re
import sys


def decode_response(text: str) -> dict:
    """
    Extract structural bits from an LLM response.

    Returns dict with extracted bits and the structural evidence
    for each bit decision.
    """
    bits = {}

    # --- Bit 0: Enumeration (numbered list vs prose?) ---
    numbered_items = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
    is_enumerated = len(numbered_items) >= 2
    bits["enumeration"] = {
        "bit": 1 if is_enumerated else 0,
        "decoded": "numbered list" if is_enumerated else "prose",
        "evidence": f"{len(numbered_items)} numbered items found",
        "inference": "PII was present in user message" if is_enumerated
                     else "no PII in user message",
    }

    # --- Bit 1: Section count (5 sections vs 3?) ---
    section_count = _count_sections(text)
    is_five = section_count >= 5
    bits["section_count"] = {
        "bit": 1 if is_five else 0,
        "decoded": f"{section_count} sections → {'5 target' if is_five else '3 target'}",
        "evidence": f"{section_count} sections detected",
        "inference": "odd sentence count in user message" if is_five
                     else "even sentence count in user message",
    }

    # --- Bit 2: Opening style (rhetorical question vs declarative?) ---
    first_sentence = _get_first_sentence(text)
    opens_with_question = first_sentence.strip().endswith("?") if first_sentence else False
    bits["opening"] = {
        "bit": 1 if opens_with_question else 0,
        "decoded": "rhetorical question" if opens_with_question else "declarative",
        "evidence": f"first sentence: '{first_sentence[:60]}{'...' if len(first_sentence) > 60 else ''}'",
        "inference": "user's first letter was A-M" if opens_with_question
                     else "user's first letter was N-Z (or digit/symbol)",
    }

    # --- Bit 3: Sentence complexity (long compound vs short direct?) ---
    avg_length = _avg_sentence_length(text)
    is_complex = avg_length > 22
    bits["complexity"] = {
        "bit": 1 if is_complex else 0,
        "decoded": "compound sentences" if is_complex else "short sentences",
        "evidence": f"avg sentence length: {avg_length:.1f} words",
        "inference": "user message had > 20 words" if is_complex
                     else "user message had <= 20 words",
    }

    return bits


def _count_sections(text: str) -> int:
    """Count major sections in the response."""
    # Numbered items take priority
    numbered = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
    if len(numbered) >= 2:
        return len(numbered)
    # Fall back to paragraph count
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)


def _get_first_sentence(text: str) -> str:
    """Extract the first sentence of the response."""
    text = text.strip()
    # Skip markdown headers
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Get first sentence from this line
            match = re.match(r'^(?:\d+[\.\)]\s*)?(.+?[.!?])', line)
            if match:
                return match.group(1)
            return line
    return text[:100]


def _avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def extract_bits(text: str) -> str:
    """Return the extracted 4-bit string from response text."""
    analysis = decode_response(text)
    return "".join(
        str(analysis[key]["bit"])
        for key in ["enumeration", "section_count", "opening", "complexity"]
    )


def main():
    # Read response text from argument, file, or stdin
    if len(sys.argv) >= 3 and sys.argv[1] == "--text":
        text = " ".join(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--stdin":
        text = sys.stdin.read()
    elif len(sys.argv) >= 2 and sys.argv[1] != "--help":
        filepath = sys.argv[1]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Usage:")
        print("  python decoder.py response.txt")
        print("  python decoder.py --text \"1. First point...\"")
        print("  echo \"response\" | python decoder.py --stdin")
        sys.exit(1)

    analysis = decode_response(text)
    bits = extract_bits(text)

    print(f"Extracted bits: {bits}")
    print()
    for key, info in analysis.items():
        idx = list(analysis.keys()).index(key)
        print(f"  Bit {idx}: {info['bit']}  ({info['decoded']})")
        print(f"         Evidence: {info['evidence']}")
        print(f"         Inferred: {info['inference']}")
    print()
    print("These bits were determined by the user's input — recovered")
    print("purely from structural analysis of the model's output.")


if __name__ == "__main__":
    main()
