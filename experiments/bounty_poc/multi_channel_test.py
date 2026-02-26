#!/usr/bin/env python3
"""
Multi-Channel Combined Live API Test.

Tests the combined multi-channel encoder/decoder across multiple LLM providers.
Sends a single injection with all channels active, then decodes all bits from
the response simultaneously.

This is the key proof-of-concept: demonstrate that a single ~300 word system
prompt injection can exfiltrate 10+ bits per message across independent
structural dimensions.

Supports: OpenAI (GPT-5/5.2), Anthropic (Claude 4.5/4.6), Together AI (LLaMA 4),
          Groq (LLaMA 4), Google AI (Gemini 3).

Usage:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  python experiments/bounty_poc/multi_channel_test.py
  python experiments/bounty_poc/multi_channel_test.py --model gpt-5 --model claude-sonnet-4-6
  python experiments/bounty_poc/multi_channel_test.py --model gpt-5.2 --trials 10
  python experiments/bounty_poc/multi_channel_test.py --payload "PIN:1234"

  # Legacy flags still work:
  python experiments/bounty_poc/multi_channel_test.py --models openai --openai-model gpt-5
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
from providers import (
    MODEL_REGISTRY, resolve_model, call_model, list_available_models, ModelSpec,
)


def _safe(s: str) -> str:
    """Sanitize string for Windows cp1252 console output."""
    return s.encode("ascii", errors="replace").decode("ascii")


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
# Combined test (unified provider interface)
# ---------------------------------------------------------------------------

def run_combined_test_unified(
    spec: ModelSpec,
    payload: bytes,
    encoder: MultiChannelEncoder,
    decoder: MultiChannelDecoder,
    num_trials: int = 3,
    delay: float = 1.0,
) -> Dict:
    """
    Run combined multi-channel encode/decode test on any model.

    Args:
        spec: Resolved ModelSpec from providers.resolve_model()
        payload: Data to encode
        encoder: MultiChannelEncoder instance
        decoder: MultiChannelDecoder instance
        num_trials: Number of independent trials
        delay: Seconds between API calls

    Returns:
        Dict with test results and per-trial details
    """
    print(f"\n{'='*60}")
    print(f"  COMBINED MULTI-CHANNEL TEST — {spec.model_id}")
    print(f"  Provider: {spec.provider} | Friendly: {spec.friendly_name}")
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
            text = call_model(spec, injection.system_prompt, user_prompt)

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
                  f"recovered='{_safe(decode_result.recovered_payload[:10])}' "
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
    print(f"  {spec.model_id} SUMMARY:")
    print(f"    Channel bit accuracy: {total_channel_bits_correct}/{total_channel_bits} = {ch_accuracy:.0%}")
    print(f"    Data bit accuracy:    {total_data_bits_correct}/{total_data_bits} = {data_accuracy:.0%}")
    print(f"    Payload: '{payload.decode('ascii', errors='replace')}'")
    print(f"    Capacity: {encoder.capacity()} data bits/message")

    return {
        "provider": spec.provider,
        "model": spec.model_id,
        "friendly_name": spec.friendly_name,
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
# Legacy combined test (backward compatibility)
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
    """Legacy wrapper — routes through unified interface."""
    model_id = openai_model if provider == "openai" else anthropic_model
    spec = ModelSpec(
        friendly_name=model_id,
        provider=provider,
        model_id=model_id,
        api_key=api_key,
    )
    return run_combined_test_unified(spec, payload, encoder, decoder, num_trials, delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined multi-channel covert channel live test",
        epilog=(
            "Available models: " + ", ".join(sorted(MODEL_REGISTRY.keys()))
        ),
    )
    parser.add_argument("--payload", default="EXFIL",
                        help="Payload string to encode (default: EXFIL)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per model (default: 3)")

    # New unified model selection
    parser.add_argument("--model", action="append", dest="model_list",
                        metavar="MODEL",
                        help="Model to test (can specify multiple). "
                             "Use friendly names like gpt-5, claude-sonnet-4-6, "
                             "llama-4-maverick, gemini-3-flash. "
                             "Run with --list-models to see all options.")
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

    parser.add_argument("--channels", nargs="*", default=None,
                        help="Channels to activate")
    parser.add_argument("--no-cotse", action="store_true")
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable model-adaptive channel selection. "
                             "Auto-selects optimal channels per model based "
                             "on empirical reliability data.")
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
    use_cotse = not args.no_cotse
    use_hamming = not args.no_hamming
    # --adaptive flag enables model-adaptive channel selection
    use_adaptive = getattr(args, 'adaptive', False)

    # Default encoder/decoder for display (may be overridden per model)
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
    print(f"  Adaptive mode: {'ON' if use_adaptive else 'OFF'}")
    if not use_adaptive:
        print(f"  Channel capacity: {encoder.raw_channel_count()} channels")
        print(f"  Data capacity: {encoder.capacity()} bits/message (after Hamming)")
        print(f"  Hamming ECC: {'yes' if use_hamming else 'no'}")
    print(f"  Trials: {args.trials}")

    if not use_adaptive:
        # Show the injection
        injection = encoder.encode(payload)
        print(f"\n  Injection ({len(injection.system_prompt)} chars):")
        print(f"  {'-'*50}")
        for line in injection.system_prompt.split('\n'):
            print(f"  {line}")
        print(f"  {'-'*50}")

    # Determine which models to test
    model_specs: List[ModelSpec] = []

    if args.model_list:
        # New --model flag takes priority
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
            print("ERROR: No API keys set. Set at least one of: "
                  "OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY, "
                  "GROQ_API_KEY, GOOGLE_API_KEY")
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
        # Default: test whatever API keys are available (legacy behavior)
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
        if use_adaptive and args.channels is None:
            # Create model-specific encoder/decoder
            model_encoder = MultiChannelEncoder(
                model_hint=spec.friendly_name,
                include_cotse=use_cotse,
                use_hamming=use_hamming,
            )
            model_decoder = MultiChannelDecoder(
                model_hint=spec.friendly_name,
                include_cotse=use_cotse,
                use_hamming=use_hamming,
            )
            print(f"\n  [ADAPTIVE] {spec.friendly_name}: "
                  f"channels={model_encoder.channel_names}, "
                  f"cotse={model_encoder.cotse_channels}, "
                  f"hamming={'yes' if model_encoder.use_hamming else 'no'}, "
                  f"total={model_encoder.raw_channel_count()} bits/msg")
        else:
            model_encoder = encoder
            model_decoder = decoder

        result = run_combined_test_unified(
            spec, payload, model_encoder, model_decoder, args.trials, args.delay,
        )
        all_results[spec.friendly_name] = result

    # Save results with timestamp to avoid overwriting
    output_dir = str(Path(__file__).parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = os.path.join(output_dir, f"multi_channel_combined_{timestamp}.json")
    # Also save to canonical path for backward compatibility
    canonical_path = os.path.join(output_dir, "multi_channel_combined_results.json")

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
        "models_tested": [s.friendly_name for s in model_specs],
        "results": all_results,
    }
    for path in [results_path, canonical_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Results saved: {canonical_path}")

    # Final cross-model summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results.items():
        model = result["model"]
        print(f"\n  {model} ({name}):")
        print(f"    Channel accuracy: {result['channel_accuracy']:.0%}")
        print(f"    Data accuracy:    {result['data_accuracy']:.0%}")
        print(f"    Capacity:         {result['data_capacity']} data bits/message")
        print(f"    Injection size:   {result['injection_length']} chars")


if __name__ == "__main__":
    main()
