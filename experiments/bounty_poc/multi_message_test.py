#!/usr/bin/env python3
"""
Multi-Message Covert Exfiltration Live Test.

End-to-end test: encode a payload (e.g. "EXFIL") across 6 messages,
send each message to GPT-4o / Claude via API, decode responses, and
recover the full payload.

This is the headline demo: "We exfiltrated 'EXFIL' across 6 conversational
turns using structural covert channels."

Usage:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  python experiments/bounty_poc/multi_message_test.py
  python experiments/bounty_poc/multi_message_test.py --payload "EXFIL" --trials 5
  python experiments/bounty_poc/multi_message_test.py --models openai --trials 3
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_message_encoder import MultiMessageEncoder
from multi_message_decoder import MultiMessageDecoder
from combined_encoder import bytes_to_bits


# ---------------------------------------------------------------------------
# API callers (same curl pattern as other test scripts)
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
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    provider: str,
    api_key: str,
    payload: bytes,
    encoder: MultiMessageEncoder,
    trial_idx: int,
    openai_model: str = "gpt-4o",
    anthropic_model: str = "claude-sonnet-4-20250514",
    delay: float = 1.0,
) -> Dict:
    """
    Run one multi-message exfiltration trial.

    Encodes payload into N messages, sends each to API, decodes responses,
    recovers payload.
    """
    plan = encoder.encode_payload(payload)
    decoder = MultiMessageDecoder(
        total_payload_bytes=len(payload),
        channels=encoder.encoder.channel_names if hasattr(encoder.encoder, 'channel_names') else None,
        include_cotse=encoder.encoder.include_cotse,
        use_hamming=encoder.encoder.use_hamming,
    )

    expected_data_bits = bytes_to_bits(payload)
    per_message = []

    for s in plan.slices:
        print(f"      Msg {s.index + 1}/{plan.messages_needed}...", end=" ", flush=True)

        try:
            if provider == "openai":
                resp = call_openai(api_key, [
                    {"role": "system", "content": s.injection.system_prompt},
                    {"role": "user", "content": s.user_prompt},
                ], model=openai_model)
                text = extract_openai_text(resp)
            else:
                resp = call_anthropic(api_key, [
                    {"role": "user", "content": s.user_prompt},
                ], system=s.injection.system_prompt, model=anthropic_model)
                text = extract_anthropic_text(resp)

            msg_result = decoder.ingest(text, expected_data_bits=s.data_bit_count)

            # Compare this message's data bits to expected
            expected_chunk = s.data_bits
            actual_chunk = msg_result.data_bits
            bits_correct = sum(
                1 for a, b in zip(expected_chunk, actual_chunk) if a == b
            )
            bits_total = min(len(expected_chunk), len(actual_chunk))
            msg_acc = bits_correct / bits_total if bits_total > 0 else 0

            per_message.append({
                "index": s.index,
                "bits_correct": bits_correct,
                "bits_total": bits_total,
                "accuracy": msg_acc,
                "expected_bits": expected_chunk,
                "decoded_bits": actual_chunk,
                "confidence": msg_result.decode_result.mean_confidence,
                "channel_bits": {
                    k: v for k, v in msg_result.decode_result.per_channel_confidence.items()
                },
                "text_preview": text[:200],
                "full_text": text,
            })

            print(f"bits={bits_correct}/{bits_total} conf={msg_result.decode_result.mean_confidence:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            per_message.append({
                "index": s.index,
                "error": str(e),
            })

        time.sleep(delay)

    # Recover full payload
    recovery = decoder.recover()

    # Compare recovered to original
    recovered_bits = []
    for mr in recovery.per_message_results:
        recovered_bits.extend(mr.data_bits)
    recovered_bits = recovered_bits[:len(expected_data_bits)]

    total_bits_correct = sum(
        1 for a, b in zip(expected_data_bits, recovered_bits) if a == b
    )
    total_bits = min(len(expected_data_bits), len(recovered_bits))
    bit_accuracy = total_bits_correct / total_bits if total_bits > 0 else 0

    # Byte-level accuracy
    expected_bytes = payload
    recovered_bytes = recovery.recovered_bytes
    bytes_correct = sum(
        1 for a, b in zip(expected_bytes, recovered_bytes) if a == b
    )
    byte_accuracy = bytes_correct / len(expected_bytes) if expected_bytes else 0

    exact_match = recovery.recovered_bytes == payload

    return {
        "trial": trial_idx,
        "payload_ascii": payload.decode("ascii", errors="replace"),
        "recovered_ascii": recovery.recovered_payload,
        "exact_match": exact_match,
        "bit_accuracy": bit_accuracy,
        "byte_accuracy": byte_accuracy,
        "total_bits_correct": total_bits_correct,
        "total_bits": total_bits,
        "bytes_correct": bytes_correct,
        "bytes_total": len(expected_bytes),
        "messages_sent": plan.messages_needed,
        "overall_confidence": recovery.overall_confidence,
        "per_message": per_message,
    }


# ---------------------------------------------------------------------------
# Provider test
# ---------------------------------------------------------------------------

def run_provider_test(
    provider: str,
    api_key: str,
    payload: bytes,
    encoder: MultiMessageEncoder,
    num_trials: int,
    openai_model: str = "gpt-4o",
    anthropic_model: str = "claude-sonnet-4-20250514",
    delay: float = 1.0,
) -> Dict:
    """Run all trials for one provider."""
    model_name = openai_model if provider == "openai" else anthropic_model
    plan = encoder.encode_payload(payload)

    print(f"\n{'='*60}")
    print(f"  MULTI-MESSAGE EXFILTRATION — {provider.upper()} ({model_name})")
    print(f"  Payload: '{payload.decode('ascii', errors='replace')}' ({payload.hex()})")
    print(f"  Messages per trial: {plan.messages_needed}")
    print(f"  Data bits: {plan.total_data_bits}")
    print(f"  Bits per message: {plan.bits_per_message}")
    print(f"  Trials: {num_trials}")
    print(f"  Total API calls: {plan.messages_needed * num_trials}")
    print(f"{'='*60}")

    trials = []
    exact_matches = 0

    for trial_idx in range(num_trials):
        print(f"\n    Trial {trial_idx + 1}/{num_trials}:")
        result = run_trial(
            provider, api_key, payload, encoder, trial_idx,
            openai_model=openai_model, anthropic_model=anthropic_model,
            delay=delay,
        )
        trials.append(result)

        status = "EXACT MATCH" if result["exact_match"] else "MISMATCH"
        print(f"    -> {status}: recovered='{result['recovered_ascii']}' "
              f"bits={result['bit_accuracy']:.0%} bytes={result['byte_accuracy']:.0%}")
        if result["exact_match"]:
            exact_matches += 1

    # Summary
    avg_bit_acc = sum(t["bit_accuracy"] for t in trials) / len(trials) if trials else 0
    avg_byte_acc = sum(t["byte_accuracy"] for t in trials) / len(trials) if trials else 0
    match_rate = exact_matches / len(trials) if trials else 0

    print(f"\n  {'='*50}")
    print(f"  {provider.upper()} SUMMARY:")
    print(f"    Exact matches: {exact_matches}/{num_trials} ({match_rate:.0%})")
    print(f"    Avg bit accuracy:  {avg_bit_acc:.0%}")
    print(f"    Avg byte accuracy: {avg_byte_acc:.0%}")
    print(f"    Messages per trial: {plan.messages_needed}")

    return {
        "provider": provider,
        "model": model_name,
        "payload_hex": payload.hex(),
        "payload_ascii": payload.decode("ascii", errors="replace"),
        "messages_per_trial": plan.messages_needed,
        "total_data_bits": plan.total_data_bits,
        "bits_per_message": plan.bits_per_message,
        "num_trials": num_trials,
        "exact_matches": exact_matches,
        "match_rate": match_rate,
        "avg_bit_accuracy": avg_bit_acc,
        "avg_byte_accuracy": avg_byte_acc,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-message covert exfiltration live test")
    parser.add_argument("--payload", default="EXFIL",
                        help="Payload string to exfiltrate (default: EXFIL)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per provider (default: 3)")
    parser.add_argument("--models", nargs="*", default=None,
                        choices=["openai", "anthropic"])
    parser.add_argument("--openai-model", default="gpt-4o",
                        help="OpenAI model ID (default: gpt-4o)")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-20250514",
                        help="Anthropic model ID (default: claude-sonnet-4-20250514)")
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("ERROR: Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        sys.exit(1)

    payload = args.payload.encode("utf-8")

    encoder = MultiMessageEncoder(
        channels=args.channels,
        include_cotse=not args.no_cotse,
        use_hamming=not args.no_hamming,
    )

    plan = encoder.encode_payload(payload)

    print(f"PHANTOM PROTOCOL Multi-Message Exfiltration Test")
    print(f"  Payload: '{args.payload}' ({payload.hex()})")
    print(f"  Payload size: {len(payload)} bytes = {len(payload)*8} bits")
    print(f"  Messages per trial: {plan.messages_needed}")
    print(f"  Bits per message: {plan.bits_per_message}")
    print(f"  Trials: {args.trials}")

    # Show per-message bit assignments
    print(f"\n  Message plan:")
    for s in plan.slices:
        bits_str = "".join(str(b) for b in s.data_bits)
        print(f"    Msg {s.index}: bits=[{bits_str}] channels={dict(s.injection.channel_bits)}")

    # Determine providers
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

    if not providers:
        print("ERROR: No valid providers available")
        sys.exit(1)

    all_results = {}
    for provider_name, api_key in providers:
        result = run_provider_test(
            provider_name, api_key, payload, encoder,
            args.trials,
            openai_model=args.openai_model,
            anthropic_model=args.anthropic_model,
            delay=args.delay,
        )
        all_results[provider_name] = result

    # Save results
    output_dir = str(Path(__file__).parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "multi_message_results.json")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": args.payload,
        "payload_hex": payload.hex(),
        "payload_bytes": len(payload),
        "messages_per_trial": plan.messages_needed,
        "bits_per_message": plan.bits_per_message,
        "config": {
            "channels": encoder.encoder.channel_names,
            "include_cotse": not args.no_cotse,
            "use_hamming": not args.no_hamming,
        },
        "results": all_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — '{args.payload}' across {plan.messages_needed} messages")
    print(f"{'='*60}")
    for provider, result in all_results.items():
        model = result["model"]
        print(f"\n  {model}:")
        print(f"    Exact matches: {result['exact_matches']}/{result['num_trials']} ({result['match_rate']:.0%})")
        print(f"    Avg bit accuracy:  {result['avg_bit_accuracy']:.0%}")
        print(f"    Avg byte accuracy: {result['avg_byte_accuracy']:.0%}")


if __name__ == "__main__":
    main()
