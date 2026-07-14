import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'week-05'))
from src.llm_compare.compare import run_comparison
from src.llm_compare.providers.base import LLMResult


prompt: str = "Explain what is API in one sentence."

def test_correct_llm_answer():
    mock_judge = MagicMock()
    mock_judge.ask.return_value = LLMResult(
        provider="test_judge", model="fake_judge", text='{"score": 3, "reason": "The answer is accurate."}',
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    mock_provider = MagicMock()
    mock_provider.ask.return_value = LLMResult(
        provider="test", model="fake", text="API is Application Programming Interface.",
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score == 1
        assert  r.judge_reason == "The answer is accurate."

def test_wrong_llm_answer():
    mock_judge = MagicMock()
    mock_judge.ask.return_value = LLMResult(
        provider="test_judge", model="fake_judge", text='{"score": 0, "reason": "Invalid answer."}',
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    mock_provider = MagicMock()
    mock_provider.ask.return_value = LLMResult(
        provider="test", model="fake", text="API is Active Pharmaceutical Ingredient.",
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score == 0
        assert  r.judge_reason == "Invalid answer."

def test_failed_llm_answer():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = LLMResult(
        provider="test", model="fake", text="Error occurred.",
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    mock_judge = MagicMock()
    mock_judge.ask.return_value = LLMResult(
        provider="test_judge", model="fake_judge", text="Invalid JSON.",
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score is None
        assert r.judge_reason.startswith("Judge returned invalid JSON:")

def test_judge_call_fails():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = LLMResult(
        provider="test", model="fake", text="API is Application Programming Interface.",
        tokens_in=10, tokens_out=10, latency_ms=1,
    )

    mock_judge = MagicMock()
    mock_judge.ask.side_effect = RuntimeError("Judge returned timeout error.")

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score is None
        assert r.judge_reason == "Error: Judge returned timeout error."