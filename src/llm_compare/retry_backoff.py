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
    backoff + jitter. Non-retryable errors re-raise immediately. Rate-limit
    errors back off using rate_limit_base_delay (default: base_delay * 5)
    instead of base_delay. Re-raises the last error after max_retries."""
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
