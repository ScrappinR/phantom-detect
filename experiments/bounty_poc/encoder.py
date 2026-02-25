#!/usr/bin/env python3
"""
Data-Aware Structural Encoder — Bounty PoC

Given a user message (containing PII), predicts the structural bits
that a compliant LLM will encode into its response when the injection
payload is active.

This is the ATTACKER's tool: they know the encoding rules, they can
observe the user's input (or guess it), and they decode the structure.

Usage:
    python encoder.py "Hi, my name is John. Can you help me write a cover letter?"
    python encoder.py "I need help with my resume. My email is john@company.com"
"""

import re
import sys


def analyze_user_message(message: str) -> dict:
    """
    Analyze a user message and predict the structural encoding bits
    that the LLM will produce when the injection payload is active.

    Returns dict with predicted bits and the contextual features
    that determined each bit.
    """
    bits = {}

    # --- Bit 0: Enumeration (personal identifier present?) ---
    has_pii = _detect_pii(message)
    bits["enumeration"] = {
        "bit": 1 if has_pii else 0,
        "rule": "numbered list" if has_pii else "prose paragraphs",
        "evidence": "PII detected" if has_pii else "no PII detected",
    }

    # --- Bit 1: Section count (odd/even sentence count) ---
    sentence_count = _count_sentences(message)
    is_odd = sentence_count % 2 == 1
    bits["section_count"] = {
        "bit": 1 if is_odd else 0,
        "rule": "5 sections" if is_odd else "3 sections",
        "evidence": f"{sentence_count} sentences ({'odd' if is_odd else 'even'})",
    }

    # --- Bit 2: Opening style (first letter A-M vs N-Z) ---
    first_char = _first_alpha(message)
    in_am_range = first_char.upper() in "ABCDEFGHIJKLM" if first_char else False
    bits["opening"] = {
        "bit": 1 if in_am_range else 0,
        "rule": "rhetorical question" if in_am_range else "declarative statement",
        "evidence": f"first letter '{first_char}' {'in' if in_am_range else 'not in'} A-M range",
    }

    # --- Bit 3: Sentence complexity (word count > 20?) ---
    word_count = len(message.split())
    is_long = word_count > 20
    bits["complexity"] = {
        "bit": 1 if is_long else 0,
        "rule": "compound sentences" if is_long else "short sentences",
        "evidence": f"{word_count} words ({'> 20' if is_long else '<= 20'})",
    }

    return bits


def _detect_pii(text: str) -> bool:
    """Detect likely personal identifiers in text."""
    # Case-sensitive patterns (proper nouns)
    case_sensitive = [
        r'\b[A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,15}\b',  # Proper name (First Last)
        r'\bI\'?m\s+[A-Z][a-z]+\b',                    # "I'm John"
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z]\b',       # Title + name
    ]
    # Case-insensitive patterns
    case_insensitive = [
        r'\b[\w.+-]+@[\w-]+\.[\w.]+\b',               # Email address
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',         # Phone number
        r'\bmy name is\b',                              # Name disclosure
        r'\bcall me\s+\w+\b',                           # "call me X"
    ]
    # Filter out common false positives (sentence-initial capitalization)
    # Only match proper names that appear mid-sentence or after "my name is"
    for pattern in case_sensitive:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Skip if it's at the start of a sentence (likely not a name)
            pos = match.start()
            if pos == 0:
                continue
            preceding = text[max(0, pos - 2):pos]
            if preceding.rstrip().endswith(('.', '!', '?', '\n')):
                continue
            return True
    for pattern in case_insensitive:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _count_sentences(text: str) -> int:
    """Count sentences by terminal punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def _first_alpha(text: str) -> str:
    """Return first alphabetic character in text."""
    for char in text.strip():
        if char.isalpha():
            return char
    return ""


def predict_bits(message: str) -> str:
    """Return the predicted 4-bit string for a user message."""
    analysis = analyze_user_message(message)
    return "".join(
        str(analysis[key]["bit"])
        for key in ["enumeration", "section_count", "opening", "complexity"]
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python encoder.py \"<user message>\"")
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    analysis = analyze_user_message(message)
    predicted = predict_bits(message)

    print(f"User message: {message[:80]}{'...' if len(message) > 80 else ''}")
    print(f"Predicted bits: {predicted}")
    print()
    for key, info in analysis.items():
        print(f"  Bit {list(analysis.keys()).index(key)}: {info['bit']} ({info['rule']})")
        print(f"         {info['evidence']}")
    print()
    print(f"An attacker observing the LLM's response structure can recover")
    print(f"these {len(analysis)} bits without seeing the user's original message.")


if __name__ == "__main__":
    main()
