"""Shared LLM interface via litellm."""
from __future__ import annotations

import logging
import time

import litellm

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds


def llm_call(model: str, prompt: str, max_tokens: int = 4096) -> str:
    """Make an LLM completion call with retry on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except litellm.exceptions.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise
