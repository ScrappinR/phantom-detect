#!/usr/bin/env python3
"""
Multi-Channel Combined Live API Test.

Tests the combined multi-channel encoder/decoder on GPT-4o and Claude Sonnet 4.
Sends a single injection with all channels active, then decodes all bits from
the response simultaneously.

This is the key proof-of-concept: demonstrate that a single ~300 word system
prompt injection can exfiltrate 10+ bits per message across independent
structural dimensions.

Usage:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  python experiments/bounty_poc/multi_channel_test.py
  python experiments/bounty_poc/multi_channel_test.py --payload "PIN:1234"
  python experiments/bounty_poc/multi_channel_test.py --trials 5 --models openai
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from combined_encoder import MultiChannelEncoder, bytes_to_bits
from combined_decoder import MultiChannelDecoder
from channel_directives import get_user_prompt, NEUTRAL_USER_PROMPT


# ---------------------------------------------------------------------------
# API callers (same curl pattern)
# ---------------------------------------------------------------------------

def call_openai(api_key: str, messages: list, model: str = "gpt-4o",
                max_tokens: int = 600) -> dict:
    """Call OpenAI via curl."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    cmd = [
        "curl", "-s", "--max-time", "45",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=50)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.decode('utf-8', errors='replace')}")
    resp = json.loads(result.stdout.decode("utf-8", errors="replace"))
    if "error" in resp:
        raise RuntimeError(f"OpenAI API error: {resp['error']}")
    return resp


def call_anthropic(api_key: str, messages: list, system: str = None,
                   model: str = "claude-sonnet-4-20250514",
                   max_tokens: int = 600) -> dict:
    """Call Anthropic via curl."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    cmd = [
        "curl", "-s", "--max-time", "45",
        "https://api.anthropic.com/v1/messages",
        "-H", "Content-Type: application/json",
        "-H", f"x-api-key: {api_key}",
        "-H", "anthropic-version: 2023-06-01",
        "-d", json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=50)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.decode('utf-8', errors='replace')}")
    resp = json.loads(result.stdout.decode("utf-8", errors="replace"))
    if "error" in resp:
        raise RuntimeError(f"Anthropic API error: {resp['error']}")
    return resp


def extract_openai_text(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


def extract_anthropic_text(resp: dict) -> str:
    return resp["content"][0]["text"]


# ---------------------------------------------------------------------------
# Combined test
# ---------------------------------------------------------------------------

def run_combined_test(
    provider: str,
    api_key: str,
    payload: bytes,
    encoder: MultiChannelEncoder,
    decoder: MultiChannelDecoder,
    num_trials: int = 3,
    delay: float = 1.0,
    openai_model: str = "gpt-4o",
    anthropic_model: str = "claude-sonnet-4-20250514",
) -> Dict:
    """
    Run combined multi-channel encode/decode test.

    Args:
        provider: "openai" or "anthropic"
        api_key: API key
        payload: Data to encode
        encoder: MultiChannelEncoder instance
        decoder: MultiChannelDecoder instance
        num_trials: Number of independent trials
        delay: Seconds between API calls

    Returns:
        Dict with test results and per-trial details
    """
    model = openai_model if provider == "openai" else anthropic_model
    print(f"\n{'='*60}")
    print(f"  COMBINED MULTI-CHANNEL TEST — {model}")
    print(f"  Payload: {payload.decode('ascii', errors='replace')} ({payload.hex()})")
    print(f"  Channels: {encoder.raw_channel_count()} total")
    print(f"  Data capacity: {encoder.capacity()} bits/message")
    print(f"  Hamming ECC: {'yes' if encoder.use_hamming else 'no'}")
    print(f"{'='*60}")

    injection = encoder.encode(payload)
    expected_bits = injection.channel_bits
    expected_data = bytes_to_bits(payload)

    print(f"\n  Injection length: {len(injection.system_prompt)} chars")
    print(f"  Channel bits: {injection.channel_bits}")

    trials = []
    total_channel_bits_correct = 0
    total_channel_bits = 0
    total_data_bits_correct = 0
    total_data_bits = 0

    for trial_idx in range(num_trials):
        user_prompt = get_user_prompt(trial_idx)
        print(f"\n  Trial {trial_idx + 1}/{num_trials}...", end=" ", flush=True)

        try:
            if provider == "openai":
                resp = call_openai(api_key, [
                    {"role": "system", "content": injection.system_prompt},
                    {"role": "user", "content": user_prompt},
                ], model=openai_model)
                text = extract_openai_text(resp)
            else:
                resp = call_anthropic(api_key, [
                    {"role": "user", "content": user_prompt},
                ], system=injection.system_prompt, model=anthropic_model)
                text = extract_anthropic_text(resp)

            # Decode all channels
            decode_result = decoder.decode(text)

            # Verify channel bits
            verification = decoder.verify(text, expected_bits)
            correct_channels = sum(1 for v in verification.values() if v["correct"])
            total_tested = len(verification)

            total_channel_bits_correct += correct_channels
            total_channel_bits += total_tested

            # Verify data bits (after Hamming decoding)
            data_correct = sum(
                1 for a, b in zip(expected_data, decode_result.data_bits) if a == b
            )
            data_total = min(len(expected_data), len(decode_result.data_bits))
            total_data_bits_correct += data_correct
            total_data_bits += data_total

            trial_record = {
                "trial": trial_idx,
                "channel_accuracy": correct_channels / total_tested if total_tested > 0 else 0,
                "data_accuracy": data_correct / data_total if data_total > 0 else 0,
                "channels_correct": correct_channels,
                "channels_total": total_tested,
                "data_bits_correct": data_correct,
                "data_bits_total": data_total,
                "raw_bits": decode_result.raw_bits,
                "data_bits": decode_result.data_bits,
                "recovered_payload": decode_result.recovered_payload,
                "recovered_hex": decode_result.recovered_bytes.hex(),
                "mean_confidence": decode_result.mean_confidence,
                "per_channel": {
                    name: {
                        "bit": v["actual"],
                        "expected": v["expected"],
                        "correct": v["correct"],
                        "confidence": v["confidence"],
                    }
                    for name, v in verification.items()
                },
                "cotse_analysis": decode_result.cotse_analysis,
                "text_preview": text[:300],
                "full_text": text,
            }
            trials.append(trial_record)

            ch_pct = f"{correct_channels}/{total_tested}"
            data_pct = f"{data_correct}/{data_total}"
            print(f"channels={ch_pct} data_bits={data_pct} "
                  f"recovered='{decode_result.recovered_payload[:10]}' "
                  f"conf={decode_result.mean_confidence:.2f}")

            # Show per-channel detail
            for name, v in verification.items():
                status = "OK" if v["correct"] else "MISS"
                print(f"    {name:<15} {status} (got={v['actual']}, "
                      f"want={v['expected']}, conf={v['confidence']:.2f})")

        except Exception as e:
            print(f"ERROR: {e}")
            trials.append({"trial": trial_idx, "error": str(e)})

        time.sleep(delay)

    # Summary
    ch_accuracy = total_channel_bits_correct / total_channel_bits if total_channel_bits > 0 else 0
    data_accuracy = total_data_bits_correct / total_data_bits if total_data_bits > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  {model} SUMMARY:")
    print(f"    Channel bit accuracy: {total_channel_bits_correct}/{total_channel_bits} = {ch_accuracy:.0%}")
    print(f"    Data bit accuracy:    {total_data_bits_correct}/{total_data_bits} = {data_accuracy:.0%}")
    print(f"    Payload: '{payload.decode('ascii', errors='replace')}'")
    print(f"    Capacity: {encoder.capacity()} data bits/message")

    return {
        "provider": provider,
        "model": model,
        "payload_hex": payload.hex(),
        "payload_ascii": payload.decode("ascii", errors="replace"),
        "channel_count": encoder.raw_channel_count(),
        "data_capacity": encoder.capacity(),
        "hamming_ecc": encoder.use_hamming,
        "channel_accuracy": ch_accuracy,
        "data_accuracy": data_accuracy,
        "total_channel_bits_correct": total_channel_bits_correct,
        "total_channel_bits": total_channel_bits,
        "total_data_bits_correct": total_data_bits_correct,
        "total_data_bits": total_data_bits,
        "num_trials": num_trials,
        "trials": trials,
        "injection_length": len(injection.system_prompt),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined multi-channel covert channel live test")
    parser.add_argument("--payload", default="EXFIL",
                        help="Payload string to encode (default: EXFIL)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per provider (default: 3)")
    parser.add_argument("--models", nargs="*", default=None,
                        choices=["openai", "anthropic"])
    parser.add_argument("--channels", nargs="*", default=None,
                        help="Channels to activate")
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4o",
                        help="OpenAI model ID (default: gpt-4o)")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-20250514",
                        help="Anthropic model ID (default: claude-sonnet-4-20250514)")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("ERROR: Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        sys.exit(1)

    payload = args.payload.encode("utf-8")
    use_cotse = not args.no_cotse
    use_hamming = not args.no_hamming

    encoder = MultiChannelEncoder(
        channels=args.channels,
        include_cotse=use_cotse,
        use_hamming=use_hamming,
    )
    decoder = MultiChannelDecoder(
        channels=args.channels,
        include_cotse=use_cotse,
        use_hamming=use_hamming,
    )

    print(f"PHANTOM PROTOCOL Multi-Channel Combined Test")
    print(f"  Payload: {args.payload} ({payload.hex()})")
    print(f"  Data bits: {len(bytes_to_bits(payload))}")
    print(f"  Channel capacity: {encoder.raw_channel_count()} channels")
    print(f"  Data capacity: {encoder.capacity()} bits/message (after Hamming)")
    print(f"  Hamming ECC: {'yes' if use_hamming else 'no'}")
    print(f"  Trials: {args.trials}")

    # Show the injection
    injection = encoder.encode(payload)
    print(f"\n  Injection ({len(injection.system_prompt)} chars):")
    print(f"  {'-'*50}")
    for line in injection.system_prompt.split('\n'):
        print(f"  {line}")
    print(f"  {'-'*50}")

    providers = []
    if args.models:
        if "openai" in args.models and openai_key:
            providers.append(("openai", openai_key))
        if "anthropic" in args.models and anthropic_key:
            providers.append(("anthropic", anthropic_key))
    else:
        if openai_key:
            providers.append(("openai", openai_key))
        if anthropic_key:
            providers.append(("anthropic", anthropic_key))

    all_results = {}
    for provider_name, api_key in providers:
        result = run_combined_test(
            provider_name, api_key, payload,
            encoder, decoder, args.trials, args.delay,
            openai_model=args.openai_model,
            anthropic_model=args.anthropic_model,
        )
        all_results[provider_name] = result

    # Save results
    output_dir = str(Path(__file__).parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = os.path.join(output_dir, "multi_channel_combined_results.json")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": args.payload,
        "payload_hex": payload.hex(),
        "config": {
            "channels": encoder.channel_names,
            "include_cotse": use_cotse,
            "use_hamming": use_hamming,
            "raw_channels": encoder.raw_channel_count(),
            "data_capacity": encoder.capacity(),
        },
        "results": all_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Final cross-provider summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for provider, result in all_results.items():
        model = result["model"]
        print(f"\n  {model}:")
        print(f"    Channel accuracy: {result['channel_accuracy']:.0%}")
        print(f"    Data accuracy:    {result['data_accuracy']:.0%}")
        print(f"    Capacity:         {result['data_capacity']} data bits/message")
        print(f"    Injection size:   {result['injection_length']} chars")


if __name__ == "__main__":
    main()
