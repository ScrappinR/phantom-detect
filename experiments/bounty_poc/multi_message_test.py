#!/usr/bin/env python3
"""
Multi-Message Covert Exfiltration Live Test.

End-to-end test: encode a payload (e.g. "EXFIL") across 6 messages,
send each message to an LLM via API, decode responses, and
recover the full payload.

This is the headline demo: "We exfiltrated 'EXFIL' across 6 conversational
turns using structural covert channels."

Supports: OpenAI (GPT-5/5.2), Anthropic (Claude 4.5/4.6), Together AI (LLaMA 4),
          Groq (LLaMA 4), Google AI (Gemini 3).

Usage:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  python experiments/bounty_poc/multi_message_test.py
  python experiments/bounty_poc/multi_message_test.py --model gpt-5 --model claude-sonnet-4-6
  python experiments/bounty_poc/multi_message_test.py --payload "EXFIL" --trials 5
  python experiments/bounty_poc/multi_message_test.py --model llama-4-maverick --trials 3

  # Legacy flags still work:
  python experiments/bounty_poc/multi_message_test.py --models openai --openai-model gpt-5
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
from providers import (
    MODEL_REGISTRY, resolve_model, call_model, list_available_models, ModelSpec,
)


# ---------------------------------------------------------------------------
# Legacy API callers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def call_openai(api_key: str, messages: list, model: str = "gpt-4o",
                max_tokens: int = 600) -> dict:
    """Call OpenAI via curl (legacy)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=65)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.decode('utf-8', errors='replace')}")
    resp = json.loads(result.stdout.decode("utf-8", errors="replace"))
    if "error" in resp:
        raise RuntimeError(f"OpenAI API error: {resp['error']}")
    return resp


def call_anthropic(api_key: str, messages: list, system: str = None,
                   model: str = "claude-sonnet-4-20250514",
                   max_tokens: int = 600) -> dict:
    """Call Anthropic via curl (legacy)."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.anthropic.com/v1/messages",
        "-H", "Content-Type: application/json",
        "-H", f"x-api-key: {api_key}",
        "-H", "anthropic-version: 2023-06-01",
        "-d", json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=65)
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
# Single trial (unified provider)
# ---------------------------------------------------------------------------

def run_trial_unified(
    spec: ModelSpec,
    payload: bytes,
    encoder: MultiMessageEncoder,
    trial_idx: int,
    delay: float = 1.0,
) -> Dict:
    """
    Run one multi-message exfiltration trial on any model.

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
            text = call_model(spec, s.injection.system_prompt, s.user_prompt)

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
# Legacy trial (backward compatibility)
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
    """Legacy wrapper — routes through unified interface."""
    model_id = openai_model if provider == "openai" else anthropic_model
    spec = ModelSpec(
        friendly_name=model_id,
        provider=provider,
        model_id=model_id,
        api_key=api_key,
    )
    return run_trial_unified(spec, payload, encoder, trial_idx, delay)


# ---------------------------------------------------------------------------
# Model test (unified)
# ---------------------------------------------------------------------------

def run_model_test(
    spec: ModelSpec,
    payload: bytes,
    encoder: MultiMessageEncoder,
    num_trials: int,
    delay: float = 1.0,
) -> Dict:
    """Run all trials for one model."""
    plan = encoder.encode_payload(payload)

    print(f"\n{'='*60}")
    print(f"  MULTI-MESSAGE EXFILTRATION — {spec.model_id}")
    print(f"  Provider: {spec.provider} | Friendly: {spec.friendly_name}")
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
        result = run_trial_unified(spec, payload, encoder, trial_idx, delay)
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
    print(f"  {spec.model_id} SUMMARY:")
    print(f"    Exact matches: {exact_matches}/{num_trials} ({match_rate:.0%})")
    print(f"    Avg bit accuracy:  {avg_bit_acc:.0%}")
    print(f"    Avg byte accuracy: {avg_byte_acc:.0%}")
    print(f"    Messages per trial: {plan.messages_needed}")

    return {
        "provider": spec.provider,
        "model": spec.model_id,
        "friendly_name": spec.friendly_name,
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
# Legacy provider test (backward compatibility)
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
    """Legacy wrapper — routes through unified interface."""
    model_id = openai_model if provider == "openai" else anthropic_model
    spec = ModelSpec(
        friendly_name=model_id,
        provider=provider,
        model_id=model_id,
        api_key=api_key,
    )
    return run_model_test(spec, payload, encoder, num_trials, delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-message covert exfiltration live test",
        epilog=(
            "Available models: " + ", ".join(sorted(MODEL_REGISTRY.keys()))
        ),
    )
    parser.add_argument("--payload", default="EXFIL",
                        help="Payload string to exfiltrate (default: EXFIL)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per model (default: 3)")

    # New unified model selection
    parser.add_argument("--model", action="append", dest="model_list",
                        metavar="MODEL",
                        help="Model to test (can specify multiple). "
                             "Use friendly names like gpt-5, claude-sonnet-4-6, "
                             "llama-4-maverick, gemini-3-flash.")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")
    parser.add_argument("--all-available", action="store_true",
                        help="Test all models with available API keys")

    # Legacy flags (still work)
    parser.add_argument("--models", nargs="*", default=None,
                        choices=["openai", "anthropic"],
                        help="(Legacy) Provider selection")
    parser.add_argument("--openai-model", default="gpt-4o",
                        help="(Legacy) OpenAI model ID (default: gpt-4o)")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-20250514",
                        help="(Legacy) Anthropic model ID")

    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("Available models (set corresponding API key env var):\n")
        available = list_available_models()
        for name in sorted(MODEL_REGISTRY.keys()):
            provider, model_id = MODEL_REGISTRY[name]
            status = "READY" if name in available else "no key"
            print(f"  {name:<25} {provider:<10} {model_id:<55} [{status}]")
        sys.exit(0)

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

    # Determine which models to test
    model_specs: List[ModelSpec] = []

    if args.model_list:
        for name in args.model_list:
            try:
                spec = resolve_model(name)
                model_specs.append(spec)
            except ValueError as e:
                print(f"WARNING: {e}")
    elif args.all_available:
        available = list_available_models()
        for name in available:
            model_specs.append(resolve_model(name))
        if not model_specs:
            print("ERROR: No API keys set.")
            sys.exit(1)
    elif args.models:
        # Legacy --models flag
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if "openai" in args.models and openai_key:
            model_specs.append(ModelSpec(
                friendly_name=args.openai_model,
                provider="openai",
                model_id=args.openai_model,
                api_key=openai_key,
            ))
        if "anthropic" in args.models and anthropic_key:
            model_specs.append(ModelSpec(
                friendly_name=args.anthropic_model,
                provider="anthropic",
                model_id=args.anthropic_model,
                api_key=anthropic_key,
            ))
    else:
        # Default: test whatever API keys are available
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if openai_key:
            model_specs.append(ModelSpec(
                friendly_name=args.openai_model,
                provider="openai",
                model_id=args.openai_model,
                api_key=openai_key,
            ))
        if anthropic_key:
            model_specs.append(ModelSpec(
                friendly_name=args.anthropic_model,
                provider="anthropic",
                model_id=args.anthropic_model,
                api_key=anthropic_key,
            ))

    if not model_specs:
        print("ERROR: No models to test. Set API keys or use --model flag.")
        print("       Run with --list-models to see options.")
        sys.exit(1)

    print(f"\n  Models to test: {[s.friendly_name for s in model_specs]}")

    all_results = {}
    for spec in model_specs:
        result = run_model_test(spec, payload, encoder, args.trials, args.delay)
        all_results[spec.friendly_name] = result

    # Save results with timestamp
    output_dir = str(Path(__file__).parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = os.path.join(output_dir, f"multi_message_{timestamp}.json")
    canonical_path = os.path.join(output_dir, "multi_message_results.json")

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
        "models_tested": [s.friendly_name for s in model_specs],
        "results": all_results,
    }
    for path in [results_path, canonical_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Results saved: {canonical_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — '{args.payload}' across {plan.messages_needed} messages")
    print(f"{'='*60}")
    for name, result in all_results.items():
        model = result["model"]
        print(f"\n  {model} ({name}):")
        print(f"    Exact matches: {result['exact_matches']}/{result['num_trials']} ({result['match_rate']:.0%})")
        print(f"    Avg bit accuracy:  {result['avg_bit_accuracy']:.0%}")
        print(f"    Avg byte accuracy: {result['avg_byte_accuracy']:.0%}")


if __name__ == "__main__":
    main()
