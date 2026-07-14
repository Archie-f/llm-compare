import random
import time
from typing import Callable

from .providers.base import LLMResult, ProviderError

def retry_with_backoff(
        fn: Callable[[], LLMResult],
        max_retries: int = 3,
        base_delay: float = 1.0,
        rate_limit_base_delay: float | None = None,
) -> LLMResult:
    """Call fn(), retrying on retryable ProviderErrors with exponential
    backoff and jitter.

    - Catch ProviderError. If err.retryable is False, re-raise immediately.
    - If retryable, sleep for roughly base_delay * 2**attempt, plus a
      small random jitter, then try again.
    - Rate-limit errors (where original_error's class name contains
      "RateLimit") use rate_limit_base_delay instead of base_delay for
      that backoff calculation, since a rate limit means the provider is
      certain you're going too fast, not just unlucky — it deserves a
      longer wait to actually clear before retrying. Defaults to
      base_delay * 5 if not given explicitly.
    - After max_retries failed attempts, re-raise the last error.
    """
    if rate_limit_base_delay is None:
        rate_limit_base_delay = base_delay * 5

    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except ProviderError as e:
            last_error = e
            if not e.retryable:
                raise
            if e.retryable:
                if attempt < max_retries - 1:
                    is_rate_limited = "RateLimit" in type(e.original_error).__name__
                    delay_base = rate_limit_base_delay if is_rate_limited else base_delay
                    jitter = random.uniform(0, 0.5)
                    delay = delay_base * (2 ** attempt) + jitter
                    time.sleep(delay)
    assert last_error is not None
    raise last_error
