from typing import Any, Generator

import pytest

from src.llm_compare.streaming import print_stream_and_collect_result
from src.llm_compare.providers.base import LLMResult


def fake_stream(chunks: list[str], result: LLMResult) -> Generator[str, Any, LLMResult]:
    """A fake generator standing in for ask_stream() — yields chunks, then returns result."""
    for chunk in chunks:
        yield chunk
    return result


def test_print_stream_and_collect_result_returns_final_llm_result() -> None:
    expected_result = LLMResult(
        provider="claude",
        model="claude-haiku-4-5",
        text="Hello world",
        tokens_in=5,
        tokens_out=3,
        latency_ms=120,
    )
    gen = fake_stream(["Hello ", "world"], expected_result)

    actual_result = print_stream_and_collect_result(gen)

    assert actual_result == expected_result


def test_print_stream_and_collect_result_prints_each_chunk(capsys: pytest.CaptureFixture[str]) -> None:
    result = LLMResult(
        provider="claude",
        model="claude-haiku-4-5",
        text="Hi there",
        tokens_in=2,
        tokens_out=2,
        latency_ms=50,
    )
    gen = fake_stream(["Hi ", "there"], result)

    print_stream_and_collect_result(gen)

    captured = capsys.readouterr()
    assert "Hi " in captured.out
    assert "there" in captured.out