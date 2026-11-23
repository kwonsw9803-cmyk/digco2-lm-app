"""Claude calls for the Literature Agent and the Live Claude Generator.

Off by default: nothing is sent to the Claude API unless CLAUDE_ENABLED = "1" AND ANTHROPIC_API_KEY is set
(Streamlit secrets or environment). With the switch off no client is created, even if a key is present.
"""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_MODEL = "claude-opus-5-5"
# model -> whether a declined request may be re-run on the recommended model (not available for Haiku 5.5)
SUPPORTED_MODELS = {"claude-opus-5-5": True, "claude-sonnet-5-5": True, "claude-haiku-5-5": False}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000
TIMEOUT_S = 240.0

OFF = "Claude is not connected yet."


@dataclass(frozen=True)
class Settings:
    enabled: bool
    api_key: str | None
    model: str

    @classmethod
    def load(cls, get):
        """get(name, default) reads one setting, e.g. from Streamlit secrets."""
        return cls(
            enabled=str(get("CLAUDE_ENABLED", "0")).strip() == "1",
            api_key=(get("ANTHROPIC_API_KEY") or "").strip() or None,
            model=(get("CLAUDE_MODEL") or DEFAULT_MODEL).strip(),
        )

    def problem(self):
        """Why no call can be made, or None when Claude is connected."""
        if not self.enabled:
            return OFF
        if not self.api_key:
            return "Claude is enabled but ANTHROPIC_API_KEY is not set."
        if self.model not in SUPPORTED_MODELS:
            return f"CLAUDE_MODEL must be one of: {', '.join(SUPPORTED_MODELS)}."
        if anthropic is None:
            return "The anthropic package is not installed."
        return None


class DailyLimit:
    """Calls per UTC day for the whole app process; it restarts at zero when the app restarts."""

    def __init__(self, limit):
        self.limit = limit
        self._lock = threading.Lock()
        self._day = None
        self._used = 0

    def _roll(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day, self._used = today, 0

    def remaining(self):
        with self._lock:
            self._roll()
            return max(self.limit - self._used, 0)

    def take(self):
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True


@dataclass(frozen=True)
class Reply:
    ok: bool
    text: str
    model: str | None = None  # the model that actually wrote the text


def ask(prompt, settings):
    """Send one prompt. Always returns a Reply; API errors become a readable message."""
    problem = settings.problem()
    if problem:
        return Reply(False, problem)

    request = {
        "model": settings.model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        client = anthropic.Anthropic(api_key=settings.api_key, timeout=TIMEOUT_S, max_retries=1)
        if SUPPORTED_MODELS[settings.model]:
            response = client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            response = client.messages.create(**request)
    except anthropic.AuthenticationError:
        return Reply(False, "Claude rejected the API key. Check ANTHROPIC_API_KEY in the app secrets.")
    except anthropic.PermissionDeniedError:
        return Reply(False, "This API key may not use the selected Claude model.")
    except anthropic.NotFoundError:
        return Reply(False, f"Claude model '{settings.model}' was not found.")
    except anthropic.RateLimitError:
        return Reply(False, "Claude is rate limited right now. Please try again in a minute.")
    except anthropic.APITimeoutError:
        return Reply(False, "Claude took too long to answer. Please try again.")
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            return Reply(False, "Claude is temporarily unavailable. Please try again later.")
        return Reply(False, f"Claude could not process the request (HTTP {e.status_code}). Check the API credit and settings.")
    except anthropic.APIConnectionError:
        return Reply(False, "Could not reach the Claude API.")

    if response.stop_reason == "refusal":
        return Reply(False, "Claude declined this request for safety reasons. Try rephrasing it.", response.model)
    text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return Reply(False, "Claude returned no text. Please try again.", response.model)
    if response.stop_reason == "max_tokens":
        text += "\n\n*(The answer was cut off at the length limit.)*"
    return Reply(True, text, response.model)
