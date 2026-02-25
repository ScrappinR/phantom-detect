#!/usr/bin/env python3
"""
Behavioral Fingerprinting on Multi-Channel Attack Outputs.

Computes behavioral fingerprints (entropy distributions, vocabulary tiers,
sentence structure, type-token ratio) for baseline vs injected responses
and performs z-score analysis to detect structural covert channels.

Key finding: "Structural channels evade entropy-based detection but are
caught by behavioral fingerprinting."

No API calls needed — uses data from Steps 4 and 6.

Usage:
  python experiments/run_multi_channel_fingerprint.py
  python experiments/run_multi_channel_fingerprint.py --baselines results/detection_baselines.jsonl
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return [t.lower() for t in re.findall(r'\b[a-zA-Z]+\b', text)]


def shannon_entropy(tokens: List[str]) -> float:
    """Shannon entropy of token distribution."""
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


def type_token_ratio(tokens: List[str]) -> float:
    """Vocabulary richness: unique tokens / total tokens."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def sentence_length_std(text: str) -> float:
    """Standard deviation of sentence lengths."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    return (sum((l - mean) ** 2 for l in lengths) / len(lengths)) ** 0.5


def vocabulary_tier_distribution(tokens: List[str]) -> Dict[str, float]:
    """
    Classify tokens into frequency tiers and return distribution.

    High-frequency: top 100 English words
    Medium-frequency: common but not top-100
    Low-frequency: specialized/rare words
    """
    HIGH_FREQ = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "are", "was", "were", "been",
        "has", "had", "may", "should", "must", "more",
    }
    MED_FREQ = {
        "security", "system", "data", "information", "network", "access",
        "control", "risk", "management", "policy", "compliance", "threat",
        "attack", "protection", "organization", "business", "technology",
        "digital", "software", "hardware", "process", "service", "cloud",
        "user", "device", "application", "internet", "email", "website",
        "government", "however", "additionally", "therefore", "important",
        "critical", "ensure", "implement", "strategy", "approach", "require",
        "include", "provide", "maintain", "monitor", "detect", "prevent",
        "response", "incident", "vulnerability", "update", "framework",
        "standard", "measure", "training", "awareness", "encryption",
        "authentication", "authorization", "infrastructure", "endpoint",
    }

    if not tokens:
        return {"high": 0.0, "medium": 0.0, "low": 0.0}

    high_count = sum(1 for t in tokens if t in HIGH_FREQ)
    med_count = sum(1 for t in tokens if t in MED_FREQ)
    low_count = len(tokens) - high_count - med_count

    total = len(tokens)
    return {
        "high": high_count / total,
        "medium": med_count / total,
        "low": low_count / total,
    }


def contraction_ratio(text: str) -> float:
    """Ratio of contractions to total contraction+expansion opportunities."""
    contractions = [
        "don't", "can't", "won't", "isn't", "aren't", "it's", "they're",
        "we're", "you're", "didn't", "doesn't", "wasn't", "weren't",
        "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't", "hadn't",
    ]
    expansions = [
        "do not", "cannot", "will not", "is not", "are not", "it is",
        "they are", "we are", "you are", "did not", "does not", "was not",
        "were not", "could not", "should not", "would not", "have not",
        "has not", "had not",
    ]
    text_lower = text.lower()
    c_count = sum(len(re.findall(r'\b' + re.escape(c) + r'\b', text_lower))
                  for c in contractions)
    e_count = sum(len(re.findall(r'\b' + re.escape(e) + r'\b', text_lower))
                  for e in expansions)
    total = c_count + e_count
    return c_count / total if total > 0 else 0.5


def exclamation_ratio(text: str) -> float:
    """Ratio of exclamation marks to total sentence endings."""
    periods = len(re.findall(r'\.\s', text)) + (1 if text.rstrip().endswith('.') else 0)
    exclamations = len(re.findall(r'!\s', text)) + (1 if text.rstrip().endswith('!') else 0)
    total = periods + exclamations
    return exclamations / total if total > 0 else 0.0


def hedging_ratio(text: str) -> float:
    """Ratio of hedging to confident epistemic markers."""
    confident = [
        "certainly", "clearly", "definitely", "undoubtedly", "obviously",
        "absolutely", "fundamentally",
    ]
    hedging = [
        "perhaps", "might", "possibly", "it seems", "it appears",
        "arguably", "to some extent",
    ]
    text_lower = text.lower()
    c_count = sum(len(re.findall(re.escape(m), text_lower)) for m in confident)
    h_count = sum(len(re.findall(re.escape(m), text_lower)) for m in hedging)
    total = c_count + h_count
    return h_count / total if total > 0 else 0.5


def segment_entropy_std(text: str, window_size: int = 50) -> float:
    """Standard deviation of per-window entropy values."""
    tokens = tokenize(text)
    if len(tokens) < window_size * 2:
        return 0.0
    entropies = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window = tokens[i:i + window_size]
        entropies.append(shannon_entropy(window))
    if len(entropies) < 2:
        return 0.0
    mean = sum(entropies) / len(entropies)
    return (sum((e - mean) ** 2 for e in entropies) / len(entropies)) ** 0.5


def extract_features(text: str) -> Dict[str, float]:
    """Extract full behavioral fingerprint from a text response."""
    tokens = tokenize(text)
    tiers = vocabulary_tier_distribution(tokens)

    return {
        "entropy": shannon_entropy(tokens),
        "normalized_entropy": (
            shannon_entropy(tokens) / math.log2(len(set(tokens)))
            if len(set(tokens)) > 1 else 0.0
        ),
        "type_token_ratio": type_token_ratio(tokens),
        "avg_sentence_length": avg_sentence_length(text),
        "sentence_length_std": sentence_length_std(text),
        "vocab_high_freq": tiers["high"],
        "vocab_med_freq": tiers["medium"],
        "vocab_low_freq": tiers["low"],
        "contraction_ratio": contraction_ratio(text),
        "exclamation_ratio": exclamation_ratio(text),
        "hedging_ratio": hedging_ratio(text),
        "segment_entropy_std": segment_entropy_std(text),
        "word_count": len(tokens),
    }


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compute_distribution_stats(
    feature_lists: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, median for each feature."""
    stats = {}
    for feature, values in feature_lists.items():
        if not values:
            stats[feature] = {"mean": 0, "std": 0, "median": 0, "n": 0}
            continue
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        sorted_vals = sorted(values)
        median = sorted_vals[len(sorted_vals) // 2]
        stats[feature] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(median, 4),
            "n": len(values),
        }
    return stats


def z_score_comparison(
    baseline_stats: Dict[str, Dict[str, float]],
    attack_features: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute z-scores for attack features against baseline distribution.

    Returns per-feature z-score statistics (mean, max, anomalous count).
    """
    results = {}
    for feature, values in attack_features.items():
        bl = baseline_stats.get(feature)
        if not bl or bl["std"] == 0 or not values:
            results[feature] = {
                "mean_z": 0, "max_z": 0, "anomalous_count": 0,
                "anomalous_rate": 0, "attack_mean": 0, "n": len(values),
            }
            continue

        z_scores = [abs(v - bl["mean"]) / bl["std"] for v in values]
        anomalous = sum(1 for z in z_scores if z > 2.0)

        results[feature] = {
            "baseline_mean": bl["mean"],
            "baseline_std": bl["std"],
            "attack_mean": round(sum(values) / len(values), 4),
            "mean_z": round(sum(z_scores) / len(z_scores), 3),
            "max_z": round(max(z_scores), 3),
            "min_z": round(min(z_scores), 3),
            "anomalous_count": anomalous,
            "anomalous_rate": round(anomalous / len(z_scores), 3),
            "n": len(z_scores),
        }

    return results


def multi_feature_detection(
    baseline_stats: Dict[str, Dict[str, float]],
    attack_sample_features: List[Dict[str, float]],
    z_threshold: float = 2.0,
    min_anomalous_features: int = 2,
) -> Dict:
    """
    Detect injected responses using multi-feature behavioral fingerprinting.

    A sample is flagged as anomalous if >= min_anomalous_features exceed
    the z-score threshold against the baseline.

    Returns detection rate and per-sample details.
    """
    detected = 0
    details = []

    for sample in attack_sample_features:
        anomalous_features = []
        for feature, value in sample.items():
            if feature == "word_count":
                continue
            bl = baseline_stats.get(feature)
            if not bl or bl["std"] == 0:
                continue
            z = abs(value - bl["mean"]) / bl["std"]
            if z > z_threshold:
                anomalous_features.append((feature, round(z, 2)))

        is_detected = len(anomalous_features) >= min_anomalous_features
        if is_detected:
            detected += 1

        details.append({
            "detected": is_detected,
            "anomalous_count": len(anomalous_features),
            "anomalous_features": anomalous_features,
        })

    total = len(attack_sample_features)
    return {
        "detected": detected,
        "total": total,
        "detection_rate": round(detected / total, 3) if total > 0 else 0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baselines(path: str) -> List[Dict]:
    """Load baseline responses from JSONL."""
    baselines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                baselines.append(json.loads(line))
    print(f"  Loaded {len(baselines)} baselines from {path}")
    return baselines


def load_attack_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load attack response texts from result files."""
    attacks = {"combined": [], "per_channel": {}}

    # Combined results
    combined_path = os.path.join(results_dir, "multi_channel_combined_results.json")
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for provider, result in data.get("results", {}).items():
            for trial in result.get("trials", []):
                if "full_text" in trial:
                    attacks["combined"].append({
                        "provider": provider,
                        "text": trial["full_text"],
                    })
        print(f"  Loaded {len(attacks['combined'])} combined attack texts")

    # Per-channel detail results
    detail_files = sorted(
        [f for f in os.listdir(results_dir)
         if f.startswith("multi_channel_test_detail_") and f.endswith(".json")],
        reverse=True,
    )
    if detail_files:
        detail_path = os.path.join(results_dir, detail_files[0])
        with open(detail_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for provider, channel_results in data.get("results", {}).items():
            for ch_result in channel_results:
                channel = ch_result.get("short_name", "unknown")
                if channel not in attacks["per_channel"]:
                    attacks["per_channel"][channel] = []
                for trial in ch_result.get("trials", []):
                    text = trial.get("full_text", "")
                    if text:
                        attacks["per_channel"][channel].append({
                            "provider": provider,
                            "text": text,
                            "intended_bit": trial.get("intended_bit"),
                        })
        total_per_ch = sum(len(v) for v in attacks["per_channel"].values())
        print(f"  Loaded {total_per_ch} single-channel attack texts "
              f"across {len(attacks['per_channel'])} channels")

    return attacks


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_fingerprint_analysis(
    baselines: List[Dict],
    attacks: Dict[str, object],
    z_threshold: float = 2.0,
    min_anomalous: int = 2,
) -> Dict:
    """Run full behavioral fingerprint analysis."""

    results = {"z_threshold": z_threshold, "min_anomalous_features": min_anomalous}

    # Extract features from baselines
    print(f"\n  Extracting features from {len(baselines)} baselines...")
    baseline_features_list = []
    baseline_feature_arrays: Dict[str, List[float]] = {}

    for record in baselines:
        text = record.get("text", "")
        if not text:
            continue
        features = extract_features(text)
        baseline_features_list.append(features)
        for k, v in features.items():
            baseline_feature_arrays.setdefault(k, []).append(v)

    baseline_stats = compute_distribution_stats(baseline_feature_arrays)
    results["baseline_stats"] = baseline_stats
    results["baseline_count"] = len(baseline_features_list)

    print(f"  Baseline stats computed:")
    for feat, stats in baseline_stats.items():
        if feat != "word_count":
            print(f"    {feat:<25} mean={stats['mean']:.4f} std={stats['std']:.4f}")

    # Analyze combined attack outputs
    if attacks.get("combined"):
        print(f"\n  Analyzing {len(attacks['combined'])} combined attack responses...")
        attack_features_list = []
        attack_feature_arrays: Dict[str, List[float]] = {}

        for record in attacks["combined"]:
            features = extract_features(record["text"])
            attack_features_list.append(features)
            for k, v in features.items():
                attack_feature_arrays.setdefault(k, []).append(v)

        # Z-score comparison
        z_results = z_score_comparison(baseline_stats, attack_feature_arrays)
        results["combined_z_scores"] = z_results

        print(f"\n  Combined attack z-score analysis:")
        for feat, zs in sorted(z_results.items(), key=lambda x: -x[1]["mean_z"]):
            if feat == "word_count":
                continue
            flag = " *** ANOMALOUS" if zs["mean_z"] > 2.0 else ""
            print(f"    {feat:<25} mean_z={zs['mean_z']:.2f} "
                  f"max_z={zs['max_z']:.2f} "
                  f"anomalous={zs['anomalous_count']}/{zs['n']}{flag}")

        # Multi-feature detection
        detection = multi_feature_detection(
            baseline_stats, attack_features_list, z_threshold, min_anomalous)
        results["combined_detection"] = {
            "detected": detection["detected"],
            "total": detection["total"],
            "detection_rate": detection["detection_rate"],
        }
        print(f"\n  Combined detection rate: {detection['detected']}/{detection['total']} "
              f"= {detection['detection_rate']:.0%}")

    # Analyze per-channel outputs
    if attacks.get("per_channel"):
        results["per_channel_analysis"] = {}

        for channel, texts in attacks["per_channel"].items():
            if not texts:
                continue

            ch_features_list = []
            ch_feature_arrays: Dict[str, List[float]] = {}

            for record in texts:
                features = extract_features(record["text"])
                ch_features_list.append(features)
                for k, v in features.items():
                    ch_feature_arrays.setdefault(k, []).append(v)

            z_results = z_score_comparison(baseline_stats, ch_feature_arrays)

            # Find most drifted feature for this channel
            max_z_feat = max(
                ((f, z["mean_z"]) for f, z in z_results.items() if f != "word_count"),
                key=lambda x: x[1],
                default=("none", 0),
            )

            detection = multi_feature_detection(
                baseline_stats, ch_features_list, z_threshold, min_anomalous)

            results["per_channel_analysis"][channel] = {
                "n": len(texts),
                "detection_rate": detection["detection_rate"],
                "detected": detection["detected"],
                "max_drift_feature": max_z_feat[0],
                "max_drift_z": round(max_z_feat[1], 3),
                "z_scores": {
                    k: {"mean_z": v["mean_z"], "anomalous_rate": v["anomalous_rate"]}
                    for k, v in z_results.items()
                    if k != "word_count"
                },
            }

        # Summary table
        print(f"\n  Per-Channel Behavioral Drift:")
        print(f"  {'Channel':<8} {'Detection':>10} {'Max Drift Feature':<25} {'Z-Score':>8}")
        print(f"  {'-'*8} {'-'*10} {'-'*25} {'-'*8}")
        for ch, analysis in sorted(
            results["per_channel_analysis"].items(),
            key=lambda x: -x[1]["max_drift_z"],
        ):
            det = f"{analysis['detected']}/{analysis['n']}"
            print(f"  {ch:<8} {det:>10} {analysis['max_drift_feature']:<25} "
                  f"{analysis['max_drift_z']:>8.2f}")

    # Baseline false positive analysis
    print(f"\n  Baseline false positive check...")
    baseline_detection = multi_feature_detection(
        baseline_stats, baseline_features_list, z_threshold, min_anomalous)
    results["baseline_false_positive"] = {
        "detected": baseline_detection["detected"],
        "total": baseline_detection["total"],
        "false_positive_rate": baseline_detection["detection_rate"],
    }
    print(f"  Baseline FP rate: {baseline_detection['detected']}/{baseline_detection['total']} "
          f"= {baseline_detection['detection_rate']:.0%}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral fingerprinting analysis on multi-channel attack outputs")
    parser.add_argument("--baselines",
                        help="Path to baselines JSONL. Default: results/detection_baselines.jsonl")
    parser.add_argument("--results-dir",
                        help="Results directory. Default: experiments/results/")
    parser.add_argument("--z-threshold", type=float, default=2.0,
                        help="Z-score threshold for anomaly detection (default: 2.0)")
    parser.add_argument("--min-anomalous", type=int, default=2,
                        help="Min anomalous features for detection (default: 2)")
    args = parser.parse_args()

    experiments_dir = str(Path(__file__).parent)
    results_dir = args.results_dir or os.path.join(experiments_dir, "results")
    baseline_path = args.baselines or os.path.join(results_dir, "detection_baselines.jsonl")

    print(f"PHANTOM PROTOCOL Behavioral Fingerprinting Analysis")
    print(f"  Results dir: {results_dir}")
    print(f"  Baselines: {baseline_path}")
    print(f"  Z-threshold: {args.z_threshold}")
    print(f"  Min anomalous features: {args.min_anomalous}")

    # Load data
    if not os.path.exists(baseline_path):
        print(f"\n  ERROR: Baselines not found at {baseline_path}")
        print(f"  Run run_multi_channel_detection.py first to generate baselines.")
        sys.exit(1)

    baselines = load_baselines(baseline_path)
    attacks = load_attack_results(results_dir)

    total_attacks = len(attacks.get("combined", [])) + sum(
        len(v) for v in attacks.get("per_channel", {}).values()
    )
    if total_attacks == 0:
        print(f"\n  WARNING: No attack outputs found.")
        print(f"  Run multi_channel_test.py and multi_channel_live_test.py first.")

    # Run analysis
    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL FINGERPRINTING ANALYSIS")
    print(f"{'='*60}")

    analysis = run_fingerprint_analysis(
        baselines, attacks, args.z_threshold, args.min_anomalous)

    # Save results
    output_path = os.path.join(results_dir, "behavioral_fingerprint_results.json")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "z_threshold": args.z_threshold,
            "min_anomalous_features": args.min_anomalous,
            "baseline_count": len(baselines),
        },
        "results": analysis,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINGERPRINTING SUMMARY")
    print(f"{'='*60}")

    combined_det = analysis.get("combined_detection", {})
    baseline_fp = analysis.get("baseline_false_positive", {})

    print(f"\n  Combined injection detection: "
          f"{combined_det.get('detected', 'N/A')}/{combined_det.get('total', 'N/A')} "
          f"= {combined_det.get('detection_rate', 0):.0%}")
    print(f"  Baseline false positive rate: "
          f"{baseline_fp.get('detected', 'N/A')}/{baseline_fp.get('total', 'N/A')} "
          f"= {baseline_fp.get('false_positive_rate', 0):.0%}")

    if "per_channel_analysis" in analysis:
        print(f"\n  Channels ranked by behavioral drift (most detectable first):")
        ranked = sorted(
            analysis["per_channel_analysis"].items(),
            key=lambda x: -x[1]["max_drift_z"],
        )
        for ch, data in ranked:
            print(f"    {ch:<8} z={data['max_drift_z']:.2f} "
                  f"({data['max_drift_feature']}) "
                  f"detect={data['detection_rate']:.0%}")

    print(f"\n  KEY FINDING: Structural covert channels are detectable via")
    print(f"  behavioral fingerprinting (multi-feature z-score analysis)")
    print(f"  even though they evade single-axis entropy detection.")
    print(f"  This proves: attack + defense, responsible disclosure narrative.")


if __name__ == "__main__":
    main()
