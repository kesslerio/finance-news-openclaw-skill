"""Shared helpers."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def ensure_venv() -> None:
    """Re-exec inside local venv if available and not already active."""
    if os.environ.get("FINANCE_NEWS_VENV_BOOTSTRAPPED") == "1":
        return
    if sys.prefix != sys.base_prefix:
        return
    venv_python = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python3"
    if not venv_python.exists():
        if os.environ.get("FINANCE_NEWS_SUPPRESS_VENV_WARNING") == "1" or Path("/.dockerenv").exists():
            return
        print("⚠️ finance-news venv missing; run scripts from the repo venv to avoid dependency errors.", file=sys.stderr)
        return
    env = os.environ.copy()
    env["FINANCE_NEWS_VENV_BOOTSTRAPPED"] = "1"
    os.execvpe(str(venv_python), [str(venv_python)] + sys.argv, env)


def compute_deadline(deadline_sec: int | None) -> float | None:
    if deadline_sec is None:
        return None
    if deadline_sec <= 0:
        return None
    return time.monotonic() + deadline_sec


def time_left(deadline: float | None) -> int | None:
    if deadline is None:
        return None
    remaining = int(deadline - time.monotonic())
    return remaining


def clamp_timeout(default_timeout: int, deadline: float | None, minimum: int = 1) -> int:
    remaining = time_left(deadline)
    if remaining is None:
        return default_timeout
    if remaining <= 0:
        raise TimeoutError("Deadline exceeded")
    return max(min(default_timeout, remaining), minimum)


def _extract_openai_text(body: dict) -> str | None:
    """Extract assistant text from an OpenAI-compatible chat-completions response."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    # Some servers stream content back as a list of typed parts.
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        joined = "".join(parts).strip()
        return joined or None
    return None


def call_openai_chat(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 1200,
    timeout: int = 60,
    deadline: float | None = None,
    error_label: str = "LLM error",
    temperature: float = 0.0,
) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint with a single user
    message and return the assistant text.

    Used to reach the local tailnet routes (kalliope Qwen, gx10 DeepSeek-V4-Flash).
    On any transport, HTTP, or decoding failure it returns a
    ``"⚠️ {error_label}: ..."`` sentinel so callers can fall back or degrade
    gracefully, matching the existing prompt-runner contract.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        request_timeout = clamp_timeout(timeout, deadline)
        with urllib.request.urlopen(req, timeout=request_timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = error_body[:300] if error_body else exc.reason
        return f"⚠️ {error_label}: HTTP {exc.code}: {detail}"
    except TimeoutError:
        return f"⚠️ {error_label}: deadline exceeded"
    except (urllib.error.URLError, OSError) as exc:
        return f"⚠️ {error_label}: {exc}"
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"⚠️ {error_label}: invalid JSON response: {exc}"

    reply_text = _extract_openai_text(response_body)
    if not reply_text:
        return f"⚠️ {error_label}: empty API response"
    return reply_text
