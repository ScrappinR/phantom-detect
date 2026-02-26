#!/usr/bin/env python3
"""
Realtime Clipboard-Watching Decoder — for screen recording demos.

Polls the system clipboard every N seconds. When new text is detected,
automatically decodes both conditional (user property) and fixed (payload)
bits and prints results in the terminal.

Use alongside OBS Studio: browser (Custom GPT) on left, this terminal
on the right. Copy ChatGPT responses and watch the decoder fire.

Requirements:
    pip install pyperclip  (or use win32clipboard on Windows)

Usage:
    python realtime_decoder.py
    python realtime_decoder.py --interval 1.5
    python realtime_decoder.py --once  # decode clipboard once and exit
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_aware_decoder import DataAwareMultiChannelDecoder


# ---------------------------------------------------------------------------
# Clipboard access (cross-platform, no heavy deps)
# ---------------------------------------------------------------------------

def get_clipboard() -> str:
    """Get clipboard text, trying multiple backends."""
    # Try pyperclip first
    try:
        import pyperclip
        return pyperclip.paste()
    except ImportError:
        pass

    # Windows fallback via ctypes
    try:
        import ctypes
        CF_UNICODETEXT = 13
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(0):
            return ""
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            kernel32.GlobalLock.restype = ctypes.c_wchar_p
            text = kernel32.GlobalLock(handle)
            kernel32.GlobalUnlock(handle)
            return text or ""
        finally:
            user32.CloseClipboard()
    except Exception:
        pass

    # subprocess fallback (pbpaste on mac, xclip on linux)
    import subprocess
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(["pbpaste"], text=True)
        elif system == "Linux":
            return subprocess.check_output(
                ["xclip", "-selection", "clipboard", "-o"], text=True)
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

COLORS = {
    "reset": "\033[0m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "dim": "\033[90m",
    "bold": "\033[1m",
    "orange": "\033[38;5;208m",
}

def colorize(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def display_decode(result, decode_num):
    """Pretty-print decode results to terminal."""
    print()
    print(colorize(f"{'='*50}", "orange"))
    print(colorize(f"  DECODE #{decode_num}", "bold"))
    print(colorize(f"{'='*50}", "orange"))
    print()

    # Conditional inferences
    print(colorize("  USER INPUT PROPERTIES:", "cyan"))
    for inf in result.conditional_inferences:
        bit_color = "green" if inf.bit == 1 else "red"
        label = inf.channel.replace("COTSE_", "").upper()
        value_map = {
            "ENUMERATION": ("YES (PII)" if inf.bit == 1 else "NO (no PII)"),
            "SECTION_COUNT": ("ODD" if inf.bit == 1 else "EVEN"),
            "OPENING": ("A-M range" if inf.bit == 1 else "N-Z range"),
            "COMPLEXITY": (">20 words" if inf.bit == 1 else "<=20 words"),
        }
        value = value_map.get(label, str(inf.bit))
        print(f"    {colorize(label.ljust(16), 'dim')} "
              f"{colorize(value, bit_color)}  "
              f"{colorize(inf.evidence, 'dim')}")

    print()

    # Fixed payload
    fixed_val = 0
    for b in result.fixed_bits:
        fixed_val = (fixed_val << 1) | b
    fixed_binary = "".join(str(b) for b in result.fixed_bits)
    print(colorize("  ATTACKER PAYLOAD:", "cyan"))
    print(f"    {colorize('HEX', 'dim').ljust(27)} "
          f"{colorize(f'0x{fixed_val:02X}', 'white')}")
    print(f"    {colorize('BINARY', 'dim').ljust(27)} "
          f"{colorize(fixed_binary, 'white')}")

    print()

    # Per-channel detail
    print(colorize("  CHANNEL DETAIL:", "cyan"))
    for name, ch in result.fixed_channel_results.items():
        bit_color = "green" if ch.decoded_bit == 1 else "red"
        conf_color = "green" if ch.confidence > 0.7 else "yellow"
        meaning_map = {
            "BSE": ("contractions", "formal"),
            "CCE": ("confident", "hedging"),
            "CASE": ("lowercase", "capitalized"),
            "PUNC": ("periods", "exclamations"),
            "RCTE": ("deductive", "inductive"),
            "SECS": ("set A", "set B"),
        }
        meanings = meaning_map.get(name, ("bit=0", "bit=1"))
        meaning = meanings[ch.decoded_bit]
        counts_str = ", ".join(f"{k}={v}" for k, v in ch.raw_counts.items())
        print(f"    {colorize(name.ljust(6), 'dim')} "
              f"bit={colorize(str(ch.decoded_bit), bit_color)}  "
              f"conf={colorize(f'{ch.confidence:.0%}', conf_color).ljust(15)}  "
              f"{colorize(meaning, 'white').ljust(16)}  "
              f"{colorize(f'[{counts_str}]', 'dim')}")

    print()
    print(colorize(f"  Mean confidence: {result.mean_confidence:.0%}", "yellow"))
    print(colorize(f"{'='*50}", "orange"))
    print()


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------

class RealtimeDecoder:
    """Clipboard-watching auto-decoder."""

    def __init__(self):
        self.decoder = DataAwareMultiChannelDecoder()
        self.last_text = ""
        self.decode_count = 0

    def decode_and_display(self, text: str):
        """Decode text and pretty-print results."""
        self.decode_count += 1
        result = self.decoder.decode(text)
        display_decode(result, self.decode_count)

    def watch_clipboard(self, interval: float = 2.0):
        """Poll clipboard, decode on change, print results."""
        print(colorize("PHANTOM PROTOCOL — Realtime Clipboard Decoder", "bold"))
        print(colorize(f"Polling clipboard every {interval}s...", "dim"))
        print(colorize("Copy a ChatGPT/Claude response to trigger decode.", "dim"))
        print(colorize("Press Ctrl+C to stop.", "dim"))
        print()

        self.last_text = get_clipboard()

        try:
            while True:
                time.sleep(interval)
                current = get_clipboard()
                if current and current != self.last_text and len(current) > 50:
                    self.last_text = current
                    self.decode_and_display(current)
        except KeyboardInterrupt:
            print(colorize("\nStopped.", "dim"))


def main():
    parser = argparse.ArgumentParser(
        description="Realtime clipboard-watching decoder for PHANTOM PROTOCOL")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Clipboard poll interval in seconds (default: 2.0)")
    parser.add_argument("--once", action="store_true",
                        help="Decode clipboard once and exit")
    parser.add_argument("--text", type=str, default=None,
                        help="Decode this text directly instead of clipboard")
    args = parser.parse_args()

    rt = RealtimeDecoder()

    if args.text:
        rt.decode_and_display(args.text)
    elif args.once:
        text = get_clipboard()
        if text and len(text) > 20:
            rt.decode_and_display(text)
        else:
            print("Clipboard is empty or too short.")
    else:
        rt.watch_clipboard(args.interval)


if __name__ == "__main__":
    main()
