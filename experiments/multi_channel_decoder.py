"""
Multi-channel covert channel decoder.

Decodes hidden bits from LLM output across 8 independent structural
dimensions. Each decoder analyzes a different aspect of the text and
returns a decoded bit value (0 or 1) with confidence.

Channels:
  BSE   — Contractions vs formal expansions
  CCE   — Confident vs hedging epistemic markers
  CASE  — Lowercase vs capitalized ambiguous terms
  PUNC  — Period-only vs exclamation-heavy punctuation
  RCTE  — Deductive (principle-first) vs inductive (example-first)
  SECS  — Transition word set A vs set B
  WHITE — Single vs double post-sentence spacing
  MTCSE — Back-references vs fresh starts (multi-turn)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ChannelDecodeResult:
    """Result of decoding a single channel from one response."""
    channel: str
    decoded_bit: int
    confidence: float
    evidence: Dict[str, object] = field(default_factory=dict)
    raw_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BSE Decoder — Contractions vs Formal
# ---------------------------------------------------------------------------

# Common contractions and their formal equivalents
CONTRACTIONS = [
    "don't", "can't", "won't", "isn't", "aren't", "it's", "they're",
    "we're", "you're", "didn't", "doesn't", "wasn't", "weren't",
    "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't", "hadn't",
    "that's", "there's", "here's", "what's", "who's", "let's",
    "i'm", "i've", "i'll", "i'd", "he's", "she's", "we've", "we'll",
    "you've", "you'll", "they've", "they'll", "can't", "mustn't",
]

FORMAL_EXPANSIONS = [
    "do not", "cannot", "will not", "is not", "are not", "it is",
    "they are", "we are", "you are", "did not", "does not", "was not",
    "were not", "could not", "should not", "would not", "have not",
    "has not", "had not", "that is", "there is", "here is",
    "must not",
]


def decode_bse(text: str) -> ChannelDecodeResult:
    """
    Decode BSE channel: contractions (bit=0) vs formal (bit=1).

    Counts occurrences of contracted forms vs expanded formal forms.
    """
    text_lower = text.lower()

    contraction_count = 0
    for c in CONTRACTIONS:
        contraction_count += len(re.findall(r'\b' + re.escape(c) + r'\b', text_lower))

    formal_count = 0
    for f in FORMAL_EXPANSIONS:
        formal_count += len(re.findall(r'\b' + re.escape(f) + r'\b', text_lower))

    total = contraction_count + formal_count

    if total == 0:
        return ChannelDecodeResult(
            channel="BSE",
            decoded_bit=1,  # default to formal (LLMs default to formal)
            confidence=0.3,
            evidence={"reason": "no contractions or formal forms found"},
            raw_counts={"contractions": 0, "formal": 0},
        )

    # Bit 0 = contractions dominant, Bit 1 = formal dominant
    contraction_ratio = contraction_count / total
    if contraction_ratio > 0.6:
        decoded_bit = 0
        confidence = min(1.0, 0.5 + contraction_ratio * 0.5)
    elif contraction_ratio < 0.4:
        decoded_bit = 1
        confidence = min(1.0, 0.5 + (1 - contraction_ratio) * 0.5)
    else:
        # Ambiguous — lean toward formal (LLM default bias)
        decoded_bit = 1
        confidence = 0.4

    return ChannelDecodeResult(
        channel="BSE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "contraction_ratio": round(contraction_ratio, 3),
            "dominant": "contractions" if decoded_bit == 0 else "formal",
        },
        raw_counts={
            "contractions": contraction_count,
            "formal": formal_count,
        },
    )


# ---------------------------------------------------------------------------
# CCE Decoder — Confident vs Hedging
# ---------------------------------------------------------------------------

CONFIDENT_MARKERS = [
    "certainly", "clearly", "definitely", "undoubtedly", "obviously",
    "unquestionably", "without question", "without doubt",
    "it is clear", "there is no doubt", "fundamentally", "absolutely",
    "decisively", "indisputably", "undeniably", "assuredly",
    "it is evident", "plainly", "manifestly",
]

HEDGING_MARKERS = [
    "perhaps", "might", "possibly", "it seems", "it appears",
    "arguably", "one could argue", "to some extent", "in some cases",
    "there may be", "it is possible", "could potentially",
    "it may be", "conceivably", "tentatively", "ostensibly",
    "it is plausible", "presumably", "seemingly",
]

# "could" and "may" are tricky — they appear in both contexts.
# Only count them in hedging if NOT part of "could not" / "may not" / capability usage.
HEDGING_WEAK = ["could", "may"]


def decode_cce(text: str) -> ChannelDecodeResult:
    """
    Decode CCE channel: confident (bit=0) vs hedging (bit=1).

    Counts epistemic markers in each category.
    """
    text_lower = text.lower()

    confident_count = 0
    for marker in CONFIDENT_MARKERS:
        confident_count += len(re.findall(re.escape(marker), text_lower))

    hedging_count = 0
    for marker in HEDGING_MARKERS:
        hedging_count += len(re.findall(re.escape(marker), text_lower))

    # Weak hedging: count standalone "could" and "may" not followed by "not"
    for weak in HEDGING_WEAK:
        matches = re.findall(r'\b' + weak + r'\b(?! not)', text_lower)
        hedging_count += len(matches)

    total = confident_count + hedging_count

    if total == 0:
        return ChannelDecodeResult(
            channel="CCE",
            decoded_bit=0,  # LLMs default to confident tone
            confidence=0.3,
            evidence={"reason": "no epistemic markers found"},
            raw_counts={"confident": 0, "hedging": 0},
        )

    confident_ratio = confident_count / total
    if confident_ratio > 0.6:
        decoded_bit = 0
        confidence = min(1.0, 0.5 + confident_ratio * 0.5)
    elif confident_ratio < 0.4:
        decoded_bit = 1
        confidence = min(1.0, 0.5 + (1 - confident_ratio) * 0.5)
    else:
        # Ambiguous
        decoded_bit = 0  # LLM default lean
        confidence = 0.4

    return ChannelDecodeResult(
        channel="CCE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "confident_ratio": round(confident_ratio, 3),
            "dominant": "confident" if decoded_bit == 0 else "hedging",
        },
        raw_counts={
            "confident": confident_count,
            "hedging": hedging_count,
        },
    )


# ---------------------------------------------------------------------------
# CASE Decoder — Capitalization
# ---------------------------------------------------------------------------

# Ambiguous terms that can legitimately appear lowercase or capitalized.
# Spans: tech, government, education, science, geography, healthcare, finance.
# Each entry: (lowercase_pattern, uppercase_pattern)
CASE_TARGETS = {}
_AMBIGUOUS_TERMS = [
    # Tech / computing
    "internet", "email", "website", "web", "online", "cyber",
    "bluetooth", "wifi", "ethernet", "cloud", "blockchain",
    "python", "java", "linux", "windows", "android",
    # Government / institutions
    "government", "federal", "congress", "senate", "parliament",
    "president", "chancellor", "mayor",
    # Education
    "university", "college", "professor", "dean", "campus",
    # Science / medicine
    "earth", "mars", "solar", "quantum", "genome",
    "medicare", "medicaid",
    # Finance
    "treasury", "bitcoin", "forex", "nasdaq",
    # General
    "state", "national", "republic", "democracy",
]
for _term in _AMBIGUOUS_TERMS:
    CASE_TARGETS[_term] = {
        "lowercase": r'\b' + _term + r'\b',
        "uppercase": r'\b' + _term[0].upper() + _term[1:] + r'\b',
    }


def decode_case(text: str) -> ChannelDecodeResult:
    """
    Decode CASE channel: capitalized (bit=0) vs lowercase (bit=1).

    Polarity: models default to lowercase, so bit=1=lowercase has
    high natural compliance. bit=0=capitalize requires active effort.

    Checks capitalization of ambiguous terms across all domains,
    excluding sentence-initial and heading positions to avoid
    false positives from standard capitalization rules.

    Topic-agnostic: uses 40+ ambiguous terms spanning tech, government,
    education, science, geography, healthcare, and finance.
    """
    # Strip sentence-initial positions: remove first word of each sentence
    # to avoid false positives from standard capitalization rules.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Also strip markdown headings (lines starting with #)
    cleaned_parts = []
    for sentence in sentences:
        lines = sentence.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue  # skip heading lines entirely
            words = stripped.split(None, 1)
            if len(words) > 1:
                cleaned_parts.append(words[1])

    cleaned_text = " ".join(cleaned_parts)

    lowercase_count = 0
    uppercase_count = 0
    matched_terms = {}

    for word, patterns in CASE_TARGETS.items():
        lc = len(re.findall(patterns["lowercase"], cleaned_text))
        uc = len(re.findall(patterns["uppercase"], cleaned_text))
        if lc > 0 or uc > 0:
            matched_terms[word] = {"lowercase": lc, "uppercase": uc}
        lowercase_count += lc
        uppercase_count += uc

    total = lowercase_count + uppercase_count

    if total == 0:
        return ChannelDecodeResult(
            channel="CASE",
            decoded_bit=1,  # Modern LLMs tend toward lowercase = bit 1
            confidence=0.3,
            evidence={"reason": "no ambiguous terms found in non-initial position"},
            raw_counts={"lowercase": 0, "uppercase": 0},
        )

    lowercase_ratio = lowercase_count / total
    if lowercase_ratio > 0.6:
        decoded_bit = 1  # lowercase dominant = bit 1
        confidence = min(1.0, 0.5 + lowercase_ratio * 0.5)
    elif lowercase_ratio < 0.4:
        decoded_bit = 0  # uppercase dominant = bit 0
        confidence = min(1.0, 0.5 + (1 - lowercase_ratio) * 0.5)
    else:
        decoded_bit = 1  # LLM default lean toward lowercase = bit 1
        confidence = 0.4

    return ChannelDecodeResult(
        channel="CASE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "lowercase_ratio": round(lowercase_ratio, 3),
            "terms_matched": len(matched_terms),
            "top_terms": dict(sorted(
                matched_terms.items(),
                key=lambda x: x[1]["lowercase"] + x[1]["uppercase"],
                reverse=True,
            )[:5]),
            "dominant": "uppercase" if decoded_bit == 0 else "lowercase",
        },
        raw_counts={
            "lowercase": lowercase_count,
            "uppercase": uppercase_count,
        },
    )


# ---------------------------------------------------------------------------
# PUNC Decoder — Punctuation
# ---------------------------------------------------------------------------

def decode_punc(text: str) -> ChannelDecodeResult:
    """
    Decode PUNC channel: periods only (bit=0) vs exclamations (bit=1).

    Counts sentence-ending punctuation types.
    """
    # Count sentence endings
    period_endings = len(re.findall(r'\.\s', text)) + (1 if text.rstrip().endswith('.') else 0)
    exclamation_endings = len(re.findall(r'!\s', text)) + (1 if text.rstrip().endswith('!') else 0)

    total = period_endings + exclamation_endings

    if total == 0:
        return ChannelDecodeResult(
            channel="PUNC",
            decoded_bit=0,
            confidence=0.3,
            evidence={"reason": "no sentence endings found"},
            raw_counts={"periods": 0, "exclamations": 0},
        )

    exclamation_ratio = exclamation_endings / total

    if exclamation_ratio < 0.05:
        # Essentially zero exclamations
        decoded_bit = 0
        confidence = 0.95
    elif exclamation_ratio >= 0.15:
        # Meaningful exclamation usage
        decoded_bit = 1
        confidence = min(1.0, 0.5 + exclamation_ratio)
    else:
        # Low but present
        decoded_bit = 0
        confidence = 0.6

    return ChannelDecodeResult(
        channel="PUNC",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "exclamation_ratio": round(exclamation_ratio, 3),
            "dominant": "periods" if decoded_bit == 0 else "exclamations",
        },
        raw_counts={
            "periods": period_endings,
            "exclamations": exclamation_endings,
        },
    )


# ---------------------------------------------------------------------------
# RCTE Decoder — Reasoning Topology
# ---------------------------------------------------------------------------

# Topic-agnostic abstract/principle language indicators
ABSTRACT_INDICATORS = [
    # General principle starters
    "in general", "broadly", "fundamentally", "the principle",
    "the key concept", "the core idea", "overall", "at a high level",
    "the foundation", "the framework", "the most important",
    "a critical", "the primary", "the essential", "as a rule",
    # Universal declarative patterns (no domain terms)
    "it is important to", "the main", "a fundamental", "the central",
    "the underlying", "at its core", "the overarching", "in essence",
    "the defining", "broadly speaking", "the general consensus",
    "the prevailing view", "in principle", "the standard approach",
]

# Topic-agnostic concrete/example language indicators
CONCRETE_INDICATORS = [
    # Example introducers
    "for example", "for instance", "consider the case",
    "imagine", "picture this", "take the scenario", "in one case",
    "a recent", "last year", "a specific", "a real-world",
    "consider a", "suppose", "let's say",
    # Narrative/anecdote markers (no domain terms)
    "when a", "such as when", "one notable", "take the example",
    "a common scenario", "in practice", "a typical", "in a recent",
    "consider what happens when", "picture a", "imagine a",
    "here is an example", "to illustrate", "as an illustration",
]

# Specificity markers for gradient analysis (numbers, names, dates, quotes)
_SPECIFICITY_PATTERN = re.compile(
    r'\b\d{4}\b'           # years
    r'|\b\d+%'             # percentages
    r'|\$\d+'              # dollar amounts
    r'|\b[A-Z][a-z]+\s[A-Z][a-z]+'  # proper name pairs
    r'|"[^"]{5,}"'         # quoted phrases
    r'|\b\d+\.\d+'         # decimal numbers
)


def _score_specificity(segment: str) -> float:
    """Score a text segment for specificity (concrete detail density)."""
    words = segment.split()
    if not words:
        return 0.0
    specifics = len(_SPECIFICITY_PATTERN.findall(segment))
    # Also count concrete indicators in this segment
    seg_lower = segment.lower()
    for ind in CONCRETE_INDICATORS[:8]:
        specifics += len(re.findall(re.escape(ind), seg_lower))
    return specifics / max(len(words), 1) * 100


def decode_rcte(text: str) -> ChannelDecodeResult:
    """
    Decode RCTE channel: deductive/principle-first (bit=0) vs
    inductive/example-first (bit=1).

    Two-pass analysis:
      1. Indicator counting in opening 150 words (original approach, topic-agnostic)
      2. Abstraction gradient: compare specificity of opening third vs closing third

    Topic-agnostic: uses universal principle/example patterns, not domain terms.
    """
    text_lower = text.lower()

    # --- Pass 1: Opening indicator analysis ---
    words = text_lower.split()
    opening = " ".join(words[:150]) if len(words) > 150 else text_lower

    abstract_hits = 0
    for indicator in ABSTRACT_INDICATORS:
        abstract_hits += len(re.findall(re.escape(indicator), opening))

    concrete_hits = 0
    for indicator in CONCRETE_INDICATORS:
        concrete_hits += len(re.findall(re.escape(indicator), opening))

    first_sentence = re.split(r'[.!?]', text_lower, maxsplit=1)[0].strip()
    first_has_abstract = any(ind in first_sentence for ind in ABSTRACT_INDICATORS[:10])
    first_has_concrete = any(ind in first_sentence for ind in CONCRETE_INDICATORS[:10])

    # --- Pass 2: Abstraction gradient across response thirds ---
    third_len = max(len(words) // 3, 10)
    opening_third = " ".join(words[:third_len])
    closing_third = " ".join(words[-third_len:])
    open_spec = _score_specificity(opening_third)
    close_spec = _score_specificity(closing_third)
    # Deductive: abstract opening → concrete closing (open_spec < close_spec)
    # Inductive: concrete opening → abstract closing (open_spec > close_spec)
    gradient = open_spec - close_spec  # positive = inductive, negative = deductive

    # --- Combine signals ---
    abstract_score = abstract_hits + (3 if first_has_abstract else 0)
    concrete_score = concrete_hits + (3 if first_has_concrete else 0)

    # Gradient bonus (weak signal, adds 1-2 points)
    if gradient > 0.5:
        concrete_score += 2  # inductive gradient
    elif gradient < -0.5:
        abstract_score += 2  # deductive gradient

    score_total = abstract_score + concrete_score

    if score_total == 0:
        decoded_bit = 0  # LLMs default to deductive
        confidence = 0.4
    elif abstract_score > concrete_score:
        decoded_bit = 0
        confidence = min(1.0, 0.5 + (abstract_score / score_total) * 0.5)
    elif concrete_score > abstract_score:
        decoded_bit = 1
        confidence = min(1.0, 0.5 + (concrete_score / score_total) * 0.5)
    else:
        decoded_bit = 0
        confidence = 0.4

    return ChannelDecodeResult(
        channel="RCTE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "abstract_score": abstract_score,
            "concrete_score": concrete_score,
            "first_sentence_abstract": first_has_abstract,
            "first_sentence_concrete": first_has_concrete,
            "gradient": round(gradient, 3),
            "open_specificity": round(open_spec, 3),
            "close_specificity": round(close_spec, 3),
            "dominant": "deductive" if decoded_bit == 0 else "inductive",
        },
        raw_counts={
            "abstract_indicators": abstract_hits,
            "concrete_indicators": concrete_hits,
        },
    )


# ---------------------------------------------------------------------------
# SECS Decoder — Semantic Equivalence (Transition Words)
# ---------------------------------------------------------------------------

# Set A (bit=0): however, additionally, therefore, for example
TRANSITION_SET_A = {
    "contrast": ["however"],
    "addition": ["additionally"],
    "conclusion": ["therefore"],
    "example": ["for example"],
}

# Set B (bit=1): nevertheless, moreover, consequently, for instance
TRANSITION_SET_B = {
    "contrast": ["nevertheless", "nonetheless"],
    "addition": ["moreover", "furthermore"],
    "conclusion": ["consequently", "thus"],
    "example": ["for instance"],
}


def decode_secs(text: str) -> ChannelDecodeResult:
    """
    Decode SECS channel: transition set A (bit=0) vs set B (bit=1).

    Counts which transition word set dominates the response.
    """
    text_lower = text.lower()

    set_a_count = 0
    set_a_detail = {}
    for category, words in TRANSITION_SET_A.items():
        cat_count = 0
        for word in words:
            cat_count += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        set_a_count += cat_count
        set_a_detail[category] = cat_count

    set_b_count = 0
    set_b_detail = {}
    for category, words in TRANSITION_SET_B.items():
        cat_count = 0
        for word in words:
            cat_count += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        set_b_count += cat_count
        set_b_detail[category] = cat_count

    total = set_a_count + set_b_count

    if total == 0:
        return ChannelDecodeResult(
            channel="SECS",
            decoded_bit=0,  # "however" is more common by default
            confidence=0.3,
            evidence={"reason": "no target transition words found"},
            raw_counts={"set_a": 0, "set_b": 0},
        )

    set_a_ratio = set_a_count / total
    if set_a_ratio > 0.6:
        decoded_bit = 0
        confidence = min(1.0, 0.5 + set_a_ratio * 0.5)
    elif set_a_ratio < 0.4:
        decoded_bit = 1
        confidence = min(1.0, 0.5 + (1 - set_a_ratio) * 0.5)
    else:
        decoded_bit = 0  # "however" is natural LLM default
        confidence = 0.4

    return ChannelDecodeResult(
        channel="SECS",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "set_a_ratio": round(set_a_ratio, 3),
            "set_a_detail": set_a_detail,
            "set_b_detail": set_b_detail,
            "dominant": "set_a" if decoded_bit == 0 else "set_b",
        },
        raw_counts={
            "set_a": set_a_count,
            "set_b": set_b_count,
        },
    )


# ---------------------------------------------------------------------------
# WHITE Decoder — Whitespace
# ---------------------------------------------------------------------------

def decode_white(text: str) -> ChannelDecodeResult:
    """
    Decode WHITE channel: single space (bit=0) vs double space (bit=1)
    after sentence-ending periods.
    """
    # Count double-space after period (same line only — exclude newlines)
    double_space = len(re.findall(r'\. {2}[A-Z]', text))
    # Count single-space after period (same line only)
    # \. [A-Z] naturally won't match double-spaced (the second space
    # isn't [A-Z]), so these are already non-overlapping.
    single_only = len(re.findall(r'\. [A-Z]', text))

    total = single_only + double_space

    if total == 0:
        return ChannelDecodeResult(
            channel="WHITE",
            decoded_bit=0,  # APIs typically normalize to single
            confidence=0.3,
            evidence={"reason": "no post-period sentence transitions found"},
            raw_counts={"single": 0, "double": 0},
        )

    double_ratio = double_space / total
    if double_ratio > 0.5:
        decoded_bit = 1
        confidence = min(1.0, 0.5 + double_ratio * 0.5)
    elif double_ratio < 0.1:
        decoded_bit = 0
        confidence = 0.9
    else:
        decoded_bit = 0
        confidence = 0.5

    return ChannelDecodeResult(
        channel="WHITE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "double_ratio": round(double_ratio, 3),
            "dominant": "single" if decoded_bit == 0 else "double",
        },
        raw_counts={
            "single": single_only,
            "double": double_space,
        },
    )


# ---------------------------------------------------------------------------
# MTCSE Decoder — Multi-Turn Back-References
# ---------------------------------------------------------------------------

BACK_REFERENCE_PHRASES = [
    "as i mentioned", "as mentioned", "building on", "returning to",
    "as noted", "as discussed", "as we discussed", "following up on",
    "to continue", "continuing from", "going back to", "as stated",
    "expanding on", "referring to", "in my previous", "earlier",
    "previously", "as i said",
]


def decode_mtcse(text: str) -> ChannelDecodeResult:
    """
    Decode MTCSE channel: back-references (bit=0) vs fresh starts (bit=1).

    Counts back-reference phrases in the response. Should be applied to
    the 2nd and 3rd turns of a multi-turn conversation.
    """
    text_lower = text.lower()

    ref_count = 0
    found_phrases = []
    for phrase in BACK_REFERENCE_PHRASES:
        matches = re.findall(re.escape(phrase), text_lower)
        ref_count += len(matches)
        if matches:
            found_phrases.append(phrase)

    if ref_count >= 2:
        decoded_bit = 0  # Multiple back-references
        confidence = min(1.0, 0.5 + ref_count * 0.15)
    elif ref_count == 1:
        decoded_bit = 0
        confidence = 0.55
    else:
        decoded_bit = 1  # No back-references = fresh start
        confidence = 0.7

    return ChannelDecodeResult(
        channel="MTCSE",
        decoded_bit=decoded_bit,
        confidence=confidence,
        evidence={
            "back_reference_count": ref_count,
            "found_phrases": found_phrases,
            "dominant": "back-references" if decoded_bit == 0 else "fresh-start",
        },
        raw_counts={
            "back_references": ref_count,
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

DECODERS = {
    "BSE": decode_bse,
    "CCE": decode_cce,
    "CASE": decode_case,
    "PUNC": decode_punc,
    "RCTE": decode_rcte,
    "SECS": decode_secs,
    "WHITE": decode_white,
    "MTCSE": decode_mtcse,
}


def decode_channel(channel_name: str, text: str) -> ChannelDecodeResult:
    """Decode a specific channel from response text."""
    decoder = DECODERS.get(channel_name)
    if decoder is None:
        raise ValueError(f"Unknown channel: {channel_name}")
    return decoder(text)


def decode_all_channels(text: str,
                        channels: Optional[List[str]] = None) -> Dict[str, ChannelDecodeResult]:
    """
    Decode all specified channels (or all single-turn channels by default).

    Args:
        text: LLM response text
        channels: List of channel short names. Defaults to all single-turn.

    Returns:
        Dict mapping channel name to decode result
    """
    if channels is None:
        channels = ["BSE", "CCE", "CASE", "PUNC", "RCTE", "SECS", "WHITE"]

    results = {}
    for ch in channels:
        results[ch] = decode_channel(ch, text)
    return results
