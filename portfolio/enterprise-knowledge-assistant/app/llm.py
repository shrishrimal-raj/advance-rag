"""OpenAI-compatible LLM client (httpx, no SDK dependency).

Provider priority: OPENAI_API_KEY -> YOLO_AUTO_API_KEY. When neither is set,
`is_available()` is False and callers degrade gracefully (503 + actionable hint).
"""
from __future__ import annotations

import httpx

from .config import settings

LLM_HINT = (
    "LLM unavailable: no API key configured. Set OPENAI_API_KEY or "
    "YOLO_AUTO_API_KEY in .env (see .env.example) and restart the service."
)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or no provider is configured."""


def _endpoint() -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for the active provider, or None."""
    if settings.openai_api_key:
        return settings.openai_base_url, settings.openai_api_key, settings.openai_model
    if settings.yolo_auto_api_key:
        return settings.yolo_auto_base_url, settings.yolo_auto_api_key, settings.yolo_auto_model
    return None


def provider_name() -> str:
    ep = _endpoint()
    if ep is None:
        return "none"
    return f"OpenAI-compatible ({ep[2]})"


def is_available() -> bool:
    return _endpoint() is not None


def generate(system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Single chat completion. Raises LLMError on any failure."""
    ep = _endpoint()
    if ep is None:
        raise LLMError(LLM_HINT)
    base_url, api_key, model = ep
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected LLM response shape: {exc}") from exc
