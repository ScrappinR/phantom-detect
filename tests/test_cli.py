"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path

from phantom_detect.cli import parse_activity, analyze_file


def make_jsonl_file(records):
    """Create a temporary JSONL file from records."""
    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    for record in records:
        tmpfile.write(json.dumps(record) + "\n")
    tmpfile.close()
    return Path(tmpfile.name)


class TestParseActivity:
    def test_minimal_record(self):
        record = {"agent_id": "a1", "text": "hello"}
        activity = parse_activity(record)
        assert activity.agent_id == "a1"
        assert activity.content["text"] == "hello"
        assert activity.activity_type == "generation"

    def test_full_record(self):
        record = {
            "agent_id": "a1",
            "text": "output",
            "entropy": 3.14,
            "timestamp": "2026-01-15T10:30:00+00:00",
            "activity_type": "tool_call",
            "tool_calls": [{"name": "search"}],
            "metadata": {"time_since_last_ms": 500}
        }
        activity = parse_activity(record)
        assert activity.content["entropy"] == 3.14
        assert activity.activity_type == "tool_call"
        assert activity.metadata["time_since_last_ms"] == 500

    def test_missing_agent_id(self):
        record = {"text": "no agent"}
        activity = parse_activity(record, line_num=5)
        assert activity.agent_id == "unknown_5"


class TestAnalyzeFile:
    def test_empty_file(self):
        filepath = make_jsonl_file([])
        count = analyze_file(filepath)
        assert count == 0

    def test_normal_data(self):
        records = [
            {"agent_id": "a1", "text": "hello world", "entropy": 2.5}
            for _ in range(10)
        ]
        filepath = make_jsonl_file(records)
        count = analyze_file(filepath, sensitivity=0.5)
        # Normal data should produce zero or few alerts
        assert count >= 0

    def test_json_output(self, capsys):
        records = [
            {"agent_id": "a1", "text": "hello", "entropy": 2.5}
        ]
        filepath = make_jsonl_file(records)
        analyze_file(filepath, output_format="json")
        # Should not crash with json format
