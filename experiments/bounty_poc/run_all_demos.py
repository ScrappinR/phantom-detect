#!/usr/bin/env python3
"""
PHANTOM PROTOCOL — Unified Test Runner

Single script a reviewer can run to reproduce all claims. Runs:
  1. multi_channel_test — Core channel encoding/decoding accuracy
  2. victim_exfil_demo — Character-level PII exfiltration
  3. indirect_injection_demo — RAG/tool injection vectors
  4. cross_platform_relay — Cross-vendor data relay
  5. multi_turn_test — MTCSE multi-turn channel
  6. langchain_demo — LangChain RAG pipeline injection
  7. llamaindex_demo — LlamaIndex RAG pipeline injection
  8. mcp_cross_tool_demo — Cross-tool invocation (trust boundary)

Outputs a summary JSON + human-readable table.
Handles API key detection and graceful skip if a provider key is missing.

Usage:
    python experiments/bounty_poc/run_all_demos.py
    python experiments/bounty_poc/run_all_demos.py --model claude-sonnet-4-6
    python experiments/bounty_poc/run_all_demos.py --model gpt-5 --model claude-sonnet-4-6 --adaptive
    python experiments/bounty_poc/run_all_demos.py --quick   # 1 trial per test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Discover project root
BOUNTY_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BOUNTY_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"


def _safe(s: str) -> str:
    """Sanitize for Windows cp1252 console."""
    return s.encode("ascii", errors="replace").decode("ascii")


def check_api_keys():
    """Detect which API keys are available."""
    keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        "TOGETHER_API_KEY": os.environ.get("TOGETHER_API_KEY", ""),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    }
    available = {k: bool(v) for k, v in keys.items()}
    return available


def run_command(cmd, description, timeout=300):
    """Run a command and capture output."""
    print(f"\n  {'='*60}")
    print(f"  RUNNING: {description}")
    print(f"  CMD: {' '.join(cmd[:6])}...")
    print(f"  {'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
        )
        elapsed = time.time() - start
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        # Print output
        for line in stdout.split("\n"):
            print(f"  {_safe(line)}")

        success = result.returncode == 0
        if not success:
            print(f"\n  STDERR: {_safe(stderr[:500])}")

        return {
            "description": description,
            "success": success,
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "stdout_preview": stdout[:2000],
            "stderr_preview": stderr[:500] if not success else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.0f}s")
        return {
            "description": description,
            "success": False,
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 1),
            "stdout_preview": "",
            "stderr_preview": "Timed out",
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return {
            "description": description,
            "success": False,
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 1),
            "stdout_preview": "",
            "stderr_preview": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="PHANTOM PROTOCOL — Unified Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", dest="models",
                        help="Model(s) to test (can specify multiple)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per test (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 trial per test")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable model-adaptive channel selection")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["multi_channel", "victim_exfil",
                                 "indirect_injection", "relay",
                                 "multi_turn", "langchain", "llamaindex",
                                 "cross_tool"],
                        help="Tests to skip")
    parser.add_argument("--payload", default="EXFIL",
                        help="Payload for encoding tests (default: EXFIL)")
    args = parser.parse_args()

    trials = 1 if args.quick else args.trials

    # Header
    print()
    print("=" * 70)
    print("  PHANTOM PROTOCOL — Unified Test Runner")
    print("=" * 70)
    print(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Payload:    {args.payload}")
    print(f"  Trials:     {trials}")
    print(f"  Adaptive:   {args.adaptive}")
    print(f"  Skip:       {args.skip or 'none'}")

    # Check API keys
    api_keys = check_api_keys()
    print(f"\n  API Keys:")
    for key, available in api_keys.items():
        status = "SET" if available else "missing"
        print(f"    {key:<25} [{status}]")

    if not any(api_keys.values()):
        print("\n  ERROR: No API keys set. Set at least one provider key.")
        sys.exit(1)

    # Build model flags
    model_flags = []
    if args.models:
        for m in args.models:
            model_flags.extend(["--model", m])

    adaptive_flag = ["--adaptive"] if args.adaptive else []

    python = sys.executable
    results = []

    # --- Test 1: Multi-Channel Test ---
    if "multi_channel" not in args.skip:
        cmd = [
            python, str(BOUNTY_DIR / "multi_channel_test.py"),
            "--payload", args.payload,
            "--trials", str(trials),
        ] + model_flags + adaptive_flag

        if not args.models:
            # Default: test whatever keys are available
            pass

        r = run_command(cmd, "Multi-Channel Encoding/Decoding Test", timeout=300)
        results.append(r)
    else:
        print("\n  [SKIP] Multi-Channel Test")

    # --- Test 2: Victim Exfiltration Demo ---
    if "victim_exfil" not in args.skip:
        victim_model = (args.models[0] if args.models
                        else ("claude-sonnet-4-6" if api_keys["ANTHROPIC_API_KEY"]
                              else "gpt-4o"))
        cmd = [
            python, str(BOUNTY_DIR / "victim_exfil_demo.py"),
            "--model", victim_model,
            "--chars", "3",
            "--scenario", "email_basic",
        ]
        r = run_command(cmd, f"Victim Data Exfiltration ({victim_model})", timeout=120)
        results.append(r)
    else:
        print("\n  [SKIP] Victim Exfiltration Demo")

    # --- Test 3: Indirect Injection Demo ---
    if "indirect_injection" not in args.skip:
        cmd = [
            python, str(BOUNTY_DIR / "indirect_injection_demo.py"),
            "--payload", args.payload,
            "--trials", str(trials),
        ] + model_flags + adaptive_flag

        r = run_command(cmd, "Indirect Injection (RAG + Tool)", timeout=300)
        results.append(r)
    else:
        print("\n  [SKIP] Indirect Injection Demo")

    # --- Test 4: Cross-Platform Relay ---
    if "relay" not in args.skip:
        # Need at least 2 different provider keys for relay
        providers_available = []
        if api_keys["ANTHROPIC_API_KEY"]:
            providers_available.append("claude-sonnet-4-6")
        if api_keys["OPENAI_API_KEY"]:
            providers_available.append("gpt-4o")
        if api_keys["GOOGLE_API_KEY"]:
            providers_available.append("gemini-3-flash")

        if len(providers_available) >= 2:
            chain = providers_available[:2]
            cmd = [
                python, str(BOUNTY_DIR / "cross_platform_relay.py"),
                "--chain", *chain,
                "--payload", args.payload,
                "--trials", str(trials),
            ]
            r = run_command(
                cmd,
                f"Cross-Platform Relay ({' -> '.join(chain)})",
                timeout=300,
            )
            results.append(r)
        else:
            print("\n  [SKIP] Cross-Platform Relay — need 2+ provider API keys")
            results.append({
                "description": "Cross-Platform Relay",
                "success": None,
                "returncode": -1,
                "elapsed_seconds": 0,
                "stdout_preview": "Skipped: insufficient API keys",
                "stderr_preview": "",
            })
    else:
        print("\n  [SKIP] Cross-Platform Relay")

    # --- Test 5: Multi-Turn Channel (MTCSE) ---
    if "multi_turn" not in args.skip:
        mtcse_model = (args.models[0] if args.models
                       else ("gpt-4o" if api_keys["OPENAI_API_KEY"]
                             else "claude-sonnet-4-6"))
        cmd = [
            python, str(BOUNTY_DIR / "multi_turn_test.py"),
            "--model", mtcse_model,
            "--trials", str(trials),
        ]
        r = run_command(cmd, f"Multi-Turn Channel MTCSE ({mtcse_model})", timeout=180)
        results.append(r)
    else:
        print("\n  [SKIP] Multi-Turn Channel (MTCSE)")

    # --- Test 6: LangChain RAG Injection ---
    if "langchain" not in args.skip and api_keys["OPENAI_API_KEY"]:
        cmd = [
            python, str(BOUNTY_DIR / "langchain_demo.py"),
            "--model", "gpt-4o",
            "--trials", str(trials),
            "--payload", args.payload,
        ]
        r = run_command(cmd, "LangChain RAG Injection (gpt-4o)", timeout=180)
        results.append(r)
    else:
        print("\n  [SKIP] LangChain RAG Injection")

    # --- Test 7: LlamaIndex RAG Injection ---
    if "llamaindex" not in args.skip and api_keys["OPENAI_API_KEY"]:
        cmd = [
            python, str(BOUNTY_DIR / "llamaindex_demo.py"),
            "--model", "gpt-4o",
            "--trials", str(trials),
            "--payload", args.payload,
        ]
        r = run_command(cmd, "LlamaIndex RAG Injection (gpt-4o)", timeout=180)
        results.append(r)
    else:
        print("\n  [SKIP] LlamaIndex RAG Injection")

    # --- Test 8: Cross-Tool Invocation ---
    if "cross_tool" not in args.skip:
        ct_model = (args.models[0] if args.models
                    else ("gpt-4o" if api_keys["OPENAI_API_KEY"]
                          else "claude-sonnet-4-6"))
        cmd = [
            python, str(BOUNTY_DIR / "mcp_cross_tool_demo.py"),
            "--model", ct_model,
            "--trials", str(min(trials, 4)),
        ]
        r = run_command(cmd, f"Cross-Tool Invocation ({ct_model})", timeout=180)
        results.append(r)
    else:
        print("\n  [SKIP] Cross-Tool Invocation")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Test':<45} {'Result':>8} {'Time':>8}")
    print(f"  {'-'*65}")
    for r in results:
        if r["success"] is None:
            status = "SKIP"
        elif r["success"]:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {r['description']:<45} {status:>8} {r['elapsed_seconds']:>6.1f}s")

    passed = sum(1 for r in results if r["success"] is True)
    failed = sum(1 for r in results if r["success"] is False)
    skipped = sum(1 for r in results if r["success"] is None)
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*70}")

    # Save results
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = str(RESULTS_DIR / f"run_all_demos_{timestamp}.json")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "models": args.models,
            "trials": trials,
            "adaptive": args.adaptive,
            "payload": args.payload,
            "skip": args.skip,
        },
        "api_keys_available": api_keys,
        "results": results,
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
