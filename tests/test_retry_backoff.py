import pytest

from src.llm_compare.providers.base import LLMResult, ProviderError
from src.llm_compare.retry_backoff import retry_with_backoff

SUCCESS_RESULT = LLMResult(
    provider="test", model="fake", text="success",
    tokens_in=1, tokens_out=1, latency_ms=1,
)

def test_retry_with_backoff_pass_without_retry() -> None:
    def positive() -> LLMResult:
        return SUCCESS_RESULT

    result = retry_with_backoff(positive, max_retries=3, base_delay=0.01)
    assert result == SUCCESS_RESULT

def test_retry_with_backoff_fail_with_retry() -> None:
    def negative() -> LLMResult:
        raise ProviderError(provider_name="test", original_error=Exception("test failure"), retryable=True)

    with pytest.raises(ProviderError, match='test failure') as e:
        retry_with_backoff(negative, max_retries=3, base_delay=0.01)

    assert str(e.value.original_error) == 'test failure'

def test_retry_with_backoff_pass_with_retry() -> None:
    calls: dict[str, int] = {"count": 0}
    def flaky() -> LLMResult:
        calls['count'] += 1
        if calls['count'] < 3:
            raise ProviderError(provider_name="test", original_error=Exception("rate limited"), retryable=True)
        return SUCCESS_RESULT

    result = retry_with_backoff(flaky, max_retries=5, base_delay=0.01)
    assert result == SUCCESS_RESULT
    assert calls['count'] == 3