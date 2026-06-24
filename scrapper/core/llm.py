"""Shared LLM interface via litellm."""
from __future__ import annotations

import logging
import random
import re
import time

import litellm

from scrapper.core.cost import tracker

logger = logging.getLogger(__name__)

MAX_RETRIES = 6
BASE_DELAY = 5  # seconds; exponential backoff base

# OpenAI rate-limit errors often suggest a wait, e.g. "Please try again in 7.451s".
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")


def _retry_delay(attempt: int, error: Exception) -> float:
    """Backoff that honors a server-suggested wait, with jitter.

    Jitter is essential: many concurrent calls hit the limit at once, so without
    it they'd all retry simultaneously and re-trip the limit (thundering herd).
    """
    m = _RETRY_AFTER_RE.search(str(error))
    suggested = float(m.group(1)) if m else 0.0
    backoff = BASE_DELAY * (2 ** attempt)  # 5, 10, 20, 40, 80...
    return max(suggested, backoff) + random.uniform(0, BASE_DELAY)


def _record_cost(model: str, stage: str, response) -> None:
    """Record token usage and cost for a completion (best-effort)."""
    try:
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    except Exception:
        prompt_tokens = completion_tokens = 0

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception as e:
        # Unknown model pricing, etc. — track tokens but treat cost as unknown.
        logger.debug(f"Could not compute cost for {model}: {e}")
        cost = 0.0

    tracker.record(model, stage, prompt_tokens, completion_tokens, cost)


def llm_call(model: str, prompt: str, max_tokens: int = 4096, stage: str = "other") -> str:
    """Make an LLM completion call with retry on rate limits.

    `stage` is a label (e.g. "classification", "structuring") used to group
    token/cost usage in the per-run cost report.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            _record_cost(model, stage, response)
            return response.choices[0].message.content.strip()
        except litellm.exceptions.RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait = _retry_delay(attempt, e)
                logger.warning(f"Rate limited, waiting {wait:.1f}s before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise
