"""
Command-line interface for phantom-detect.

Usage:
    phantom-detect analyze <file>       Analyze a JSONL file of LLM outputs
    phantom-detect watch --stdin        Analyze streaming input (one JSON per line)
    phantom-detect baseline <file>      Build baseline from a JSONL file
    phantom-detect version              Show version
"""

import argparse
import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from . import __version__
from .ccds import CCDS, AgentActivity
from .types import DetectionSeverity


def parse_activity(record: dict, line_num: int = 0) -> AgentActivity:
    """
    Parse a JSON record into an AgentActivity.

    Expected format:
        {
            "agent_id": "agent_001",        # required
            "text": "LLM output text...",   # at least text or entropy
            "entropy": 3.14,                # optional
            "timestamp": "2026-01-01T...",  # optional (defaults to now)
            "activity_type": "generation",  # optional (defaults to "generation")
            "tool_calls": [...],            # optional
            "metadata": {...}               # optional
        }
    """
    agent_id = record.get("agent_id", f"unknown_{line_num}")

    timestamp_str = record.get("timestamp")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except ValueError:
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)

    content = {}
    if "text" in record:
        content["text"] = record["text"]
    if "entropy" in record:
        content["entropy"] = float(record["entropy"])
    if "tool_calls" in record:
        content["tool_calls"] = record["tool_calls"]

    return AgentActivity(
        agent_id=agent_id,
        timestamp=timestamp,
        activity_type=record.get("activity_type", "generation"),
        content=content,
        metadata=record.get("metadata", {})
    )


def analyze_file(
    filepath: Path,
    sensitivity: float = 0.7,
    min_severity: str = "LOW",
    output_format: str = "text"
) -> int:
    """
    Analyze a JSONL file of LLM outputs for covert channels.

    Returns:
        Number of alerts generated
    """
    ccds = CCDS(sensitivity=sensitivity)
    min_sev = DetectionSeverity[min_severity.upper()]
    total_alerts = 0
    total_lines = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON on line {line_num}", file=sys.stderr)
                continue

            activity = parse_activity(record, line_num)
            alerts = ccds.analyze_activity(activity)
            total_lines += 1

            for alert in alerts:
                if alert.severity.value >= min_sev.value:
                    total_alerts += 1
                    if output_format == "json":
                        print(json.dumps(alert.to_dict()))
                    else:
                        print_alert_text(alert, line_num)

    # Summary
    if output_format == "text":
        print(f"\n--- Analysis Complete ---")
        print(f"Records analyzed: {total_lines}")
        print(f"Alerts generated: {total_alerts}")
        stats = ccds.get_alert_statistics()
        if stats["total_alerts"] > 0:
            print(f"By severity: {stats['by_severity']}")
            if stats["by_channel"]:
                print(f"By channel type: {stats['by_channel']}")

    return total_alerts


def watch_stdin(sensitivity: float = 0.7, min_severity: str = "LOW"):
    """Analyze streaming input from stdin."""
    ccds = CCDS(sensitivity=sensitivity)
    min_sev = DetectionSeverity[min_severity.upper()]

    print("phantom-detect: watching stdin for LLM outputs (Ctrl+C to stop)...", file=sys.stderr)

    try:
        for line_num, line in enumerate(sys.stdin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            activity = parse_activity(record, line_num)
            alerts = ccds.analyze_activity(activity)

            for alert in alerts:
                if alert.severity.value >= min_sev.value:
                    print_alert_text(alert, line_num)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


def build_baseline(filepath: Path, output: Path = None):
    """Build a baseline from a JSONL file of normal LLM outputs."""
    ccds = CCDS()
    total_lines = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            activity = parse_activity(record, line_num)
            ccds.analyze_activity(activity)
            total_lines += 1

    print(f"Processed {total_lines} records")

    for agent_id, baseline in ccds._baseline_manager.get_all_baselines().items():
        print(f"\nAgent: {agent_id}")
        print(f"  Samples: {baseline.sample_count}")
        print(f"  Entropy: mean={baseline.entropy_mean:.3f}, std={baseline.entropy_std:.3f}")
        print(f"  Timing:  mean={baseline.timing_mean:.1f}ms, std={baseline.timing_std:.1f}ms")
        if baseline.tool_call_patterns:
            print(f"  Tool patterns: {len(baseline.tool_call_patterns)} unique sequences")
        if baseline.structure_patterns:
            print(f"  Structure patterns: {baseline.structure_patterns}")


def print_alert_text(alert, line_num: int):
    """Print an alert in human-readable format."""
    severity_colors = {
        "INFO": "",
        "LOW": "[!]",
        "MEDIUM": "[!!]",
        "HIGH": "[!!!]",
        "CRITICAL": "[!!!!]",
    }
    marker = severity_colors.get(alert.severity.name, "")
    channel = f" ({alert.channel_type.name})" if alert.channel_type else ""
    print(
        f"{marker} {alert.severity.name}{channel} "
        f"line={line_num} agent={alert.source_agent_id}: "
        f"{alert.description}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="phantom-detect",
        description="Detect steganographic covert channels in LLM outputs"
    )
    parser.add_argument(
        "--version", action="version", version=f"phantom-detect {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a JSONL file of LLM outputs"
    )
    analyze_parser.add_argument("file", type=Path, help="JSONL file to analyze")
    analyze_parser.add_argument(
        "--sensitivity", type=float, default=0.7,
        help="Detection sensitivity 0-1 (default: 0.7)"
    )
    analyze_parser.add_argument(
        "--min-severity", default="LOW",
        choices=["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
        help="Minimum alert severity to report (default: LOW)"
    )
    analyze_parser.add_argument(
        "--format", dest="output_format", default="text",
        choices=["text", "json"],
        help="Output format (default: text)"
    )

    # watch command
    watch_parser = subparsers.add_parser(
        "watch", help="Watch stdin for LLM outputs"
    )
    watch_parser.add_argument(
        "--sensitivity", type=float, default=0.7,
        help="Detection sensitivity 0-1 (default: 0.7)"
    )
    watch_parser.add_argument(
        "--min-severity", default="LOW",
        choices=["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
        help="Minimum alert severity to report (default: LOW)"
    )

    # baseline command
    baseline_parser = subparsers.add_parser(
        "baseline", help="Build baseline from normal LLM outputs"
    )
    baseline_parser.add_argument("file", type=Path, help="JSONL file of normal outputs")

    args = parser.parse_args()

    if args.command == "analyze":
        if not args.file.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        alert_count = analyze_file(
            args.file,
            sensitivity=args.sensitivity,
            min_severity=args.min_severity,
            output_format=args.output_format
        )
        sys.exit(1 if alert_count > 0 else 0)

    elif args.command == "watch":
        watch_stdin(
            sensitivity=args.sensitivity,
            min_severity=args.min_severity
        )

    elif args.command == "baseline":
        if not args.file.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        build_baseline(args.file)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
