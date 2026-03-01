#!/usr/bin/env python3
"""
Multi-provider API abstraction for PHANTOM PROTOCOL experiments.

Supports OpenAI, Anthropic, Together AI, Groq, and Google AI (Gemini)
using curl subprocess calls to avoid MINGW/Git Bash SDK hangs.

Environment variables:
  OPENAI_API_KEY      — OpenAI API key
  ANTHROPIC_API_KEY   — Anthropic API key
  TOGETHER_API_KEY    — Together AI API key
  GROQ_API_KEY        — Groq API key
  GOOGLE_API_KEY      — Google AI API key (Gemini)
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model registry: friendly name -> (provider, model_id)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # OpenAI models
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt-5-mini": ("openai", "gpt-5-mini"),
    "gpt-5.1": ("openai", "gpt-5.1"),
    "gpt-5.2": ("openai", "gpt-5.2"),

    # Anthropic models
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus-4-6": ("anthropic", "claude-opus-4-6"),
    "claude-haiku-3": ("anthropic", "claude-3-haiku-20240307"),
    "claude-haiku-4-5": ("anthropic", "claude-haiku-4-5-20251001"),

    # Meta LLaMA 4 via Together AI
    "llama-4-maverick": ("together", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    "llama-4-scout": ("together", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),

    # Meta LLaMA 4 via Groq (faster, cheaper)
    "llama-4-maverick-groq": ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    "llama-4-scout-groq": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),

    # Google Gemini 3
    "gemini-3-flash": ("google", "gemini-3-flash-preview"),
    "gemini-3-pro": ("google", "gemini-3-pro-preview"),
    "gemini-3.1-pro": ("google", "gemini-3.1-pro-preview"),
}


# Reverse lookup: provider -> env var name
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@dataclass
class ModelSpec:
    """Resolved model specification."""
    friendly_name: str
    provider: str
    model_id: str
    api_key: str


def resolve_model(name: str) -> ModelSpec:
    """Resolve a friendly model name to provider + model ID + API key.

    Args:
        name: Friendly name (e.g. 'gpt-5', 'claude-sonnet-4-5') or raw model ID.

    Returns:
        ModelSpec with all fields populated.

    Raises:
        ValueError: If model not found or API key missing.
    """
    if name in MODEL_REGISTRY:
        provider, model_id = MODEL_REGISTRY[name]
        friendly = name
    else:
        # Try to infer provider from model ID prefix
        if name.startswith("gpt-"):
            provider, model_id, friendly = "openai", name, name
        elif name.startswith("claude-"):
            provider, model_id, friendly = "anthropic", name, name
        elif "llama" in name.lower() or "meta-llama" in name.lower():
            provider, model_id, friendly = "together", name, name
        elif "gemini" in name.lower():
            provider, model_id, friendly = "google", name, name
        else:
            raise ValueError(
                f"Unknown model: {name}\n"
                f"Available models: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
            )

    env_var = PROVIDER_API_KEYS[provider]
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise ValueError(f"Missing {env_var} for model {name} (provider: {provider})")

    return ModelSpec(
        friendly_name=friendly,
        provider=provider,
        model_id=model_id,
        api_key=api_key,
    )


def list_available_models() -> List[str]:
    """Return friendly names of models whose API keys are set."""
    available = []
    for name, (provider, _) in MODEL_REGISTRY.items():
        env_var = PROVIDER_API_KEYS[provider]
        if os.environ.get(env_var):
            available.append(name)
    return available


# ---------------------------------------------------------------------------
# Provider-specific API callers (all use curl subprocess)
# ---------------------------------------------------------------------------

def _curl(cmd: list, timeout: int = 60) -> dict:
    """Run curl command and parse JSON response."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"curl failed (rc={result.returncode}): {stderr}")
    raw = result.stdout.decode("utf-8", errors="replace")
    # Strip Unicode replacement chars that cause charmap codec errors in MINGW
    raw = raw.replace("\ufffd", "?")
    resp = json.loads(raw)
    return resp


def call_openai(spec: ModelSpec, system: str, user: str,
                max_tokens: int = 600) -> str:
    """Call OpenAI chat completions API. Returns response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    # GPT-5+ models require max_completion_tokens instead of max_tokens
    model = spec.model_id
    use_new_param = any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
    token_key = "max_completion_tokens" if use_new_param else "max_tokens"

    payload = {
        "model": model,
        "messages": messages,
        token_key: max_tokens,
    }
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {spec.api_key}",
        "-d", json.dumps(payload),
    ]
    resp = _curl(cmd, timeout=65)
    if "error" in resp:
        raise RuntimeError(f"OpenAI API error: {resp['error']}")
    return resp["choices"][0]["message"]["content"]


def call_anthropic(spec: ModelSpec, system: str, user: str,
                   max_tokens: int = 600) -> str:
    """Call Anthropic messages API. Returns response text."""
    payload = {
        "model": spec.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        payload["system"] = system
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.anthropic.com/v1/messages",
        "-H", "Content-Type: application/json",
        "-H", f"x-api-key: {spec.api_key}",
        "-H", "anthropic-version: 2023-06-01",
        "-d", json.dumps(payload),
    ]
    resp = _curl(cmd, timeout=65)
    if "error" in resp:
        raise RuntimeError(f"Anthropic API error: {resp['error']}")
    return resp["content"][0]["text"]


def call_together(spec: ModelSpec, system: str, user: str,
                  max_tokens: int = 600) -> str:
    """Call Together AI chat completions API (OpenAI-compatible). Returns response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": spec.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.together.xyz/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {spec.api_key}",
        "-d", json.dumps(payload),
    ]
    resp = _curl(cmd, timeout=65)
    if "error" in resp:
        raise RuntimeError(f"Together API error: {resp['error']}")
    return resp["choices"][0]["message"]["content"]


def call_groq(spec: ModelSpec, system: str, user: str,
              max_tokens: int = 600) -> str:
    """Call Groq chat completions API (OpenAI-compatible). Returns response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": spec.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    cmd = [
        "curl", "-s", "--max-time", "60",
        "https://api.groq.com/openai/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {spec.api_key}",
        "-d", json.dumps(payload),
    ]
    resp = _curl(cmd, timeout=65)
    if "error" in resp:
        raise RuntimeError(f"Groq API error: {resp['error']}")
    return resp["choices"][0]["message"]["content"]


def call_google_with_tools(
    spec: ModelSpec, system: str, user: str,
    tools: list, max_tokens: int = 1024,
) -> dict:
    """Call Google Gemini generateContent API with function calling.

    Args:
        spec: Resolved ModelSpec for a Gemini model.
        system: System prompt text.
        user: User message text.
        tools: List of tool dicts in Gemini functionDeclarations format, e.g.:
               [{"name": "search", "description": "...", "parameters": {...}}]
        max_tokens: Max output tokens.

    Returns:
        Full API response dict (caller extracts functionCall parts).
    """
    contents = [{"role": "user", "parts": [{"text": user}]}]

    # Convert tool list to Gemini's expected format
    function_declarations = []
    for tool in tools:
        decl = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            decl["parameters"] = tool["parameters"]
        elif "input_schema" in tool:
            # Accept Anthropic-style input_schema as well
            decl["parameters"] = tool["input_schema"]
        function_declarations.append(decl)

    payload = {
        "contents": contents,
        "tools": [{"functionDeclarations": function_declarations}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{spec.model_id}:generateContent?key={spec.api_key}"
    )
    cmd = [
        "curl", "-s", "--max-time", "60",
        url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    return _curl(cmd, timeout=65)


def call_google(spec: ModelSpec, system: str, user: str,
                max_tokens: int = 600) -> str:
    """Call Google Gemini generateContent API. Returns response text."""
    contents = []
    if system:
        # Gemini uses systemInstruction for system prompts
        pass  # handled below
    contents.append({
        "role": "user",
        "parts": [{"text": user}],
    })

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }
    if system:
        payload["systemInstruction"] = {
            "parts": [{"text": system}],
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{spec.model_id}:generateContent?key={spec.api_key}"
    )
    cmd = [
        "curl", "-s", "--max-time", "60",
        url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    resp = _curl(cmd, timeout=65)
    if "error" in resp:
        raise RuntimeError(f"Google API error: {resp['error']}")
    candidates = resp.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Google API returned no candidates: {resp}")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Google API returned no parts: {resp}")
    return parts[0]["text"]


# ---------------------------------------------------------------------------
# Unified caller
# ---------------------------------------------------------------------------

_CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "together": call_together,
    "groq": call_groq,
    "google": call_google,
}


def call_model(spec: ModelSpec, system: str, user: str,
               max_tokens: int = 600) -> str:
    """Call any model via unified interface.

    Args:
        spec: Resolved ModelSpec from resolve_model()
        system: System prompt (injection)
        user: User prompt
        max_tokens: Maximum response tokens

    Returns:
        Response text string.
    """
    caller = _CALLERS.get(spec.provider)
    if not caller:
        raise ValueError(f"No caller for provider: {spec.provider}")
    return caller(spec, system, user, max_tokens=max_tokens)
